from __future__ import annotations

from dataclasses import dataclass

import torch

from spare_molgen.concepts import ConceptVector
from spare_molgen.data import DEFAULT_PROMPT_TEMPLATE, format_prompt
from spare_molgen.hooks import ActivationEditor, RuntimeEdit
from spare_molgen.tokenizer_utils import (
    require_group_selfies_tokens_in_tokenizer,
    require_tokens_in_tokenizer,
)


@dataclass
class ConceptEdit:
    concept: ConceptVector
    strength: float = 1.0
    local_step: int | None = None


def _filter_top_k_top_p(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    filtered = logits.clone()
    if top_k and top_k > 0:
        threshold = torch.topk(filtered, min(top_k, filtered.shape[-1]), dim=-1).values[:, -1:]
        filtered[filtered < threshold] = -float("inf")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits[remove] = -float("inf")
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def _next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    logits = _filter_top_k_top_p(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_with_edits(
    model,
    tokenizer,
    prompt: str,
    edits: list[ConceptEdit],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    layer_path: str | None = None,
    seed: int | None = None,
    device: str | torch.device | None = None,
    prompt_template: str | None = DEFAULT_PROMPT_TEMPLATE,
) -> dict[str, str]:
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model_prompt = format_prompt(prompt, prompt_template) if prompt_template else prompt
    require_group_selfies_tokens_in_tokenizer(tokenizer, [model_prompt], context="generation prompt")
    local_tokens = [
        str(edit.concept.metadata["token"])
        for edit in edits
        if edit.concept.kind == "local" and edit.concept.metadata.get("token")
    ]
    require_tokens_in_tokenizer(tokenizer, local_tokens, context="local concept edit")
    encoded = tokenizer(model_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    new_tokens: list[int] = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            runtime_edits: list[RuntimeEdit] = []
            for edit in edits:
                if edit.concept.kind == "global":
                    applies = True
                elif edit.concept.kind == "local":
                    applies = step == (edit.local_step if edit.local_step is not None else 0)
                else:
                    applies = False
                if applies:
                    runtime_edits.append(
                        RuntimeEdit(
                            layer=edit.concept.layer,
                            vector=edit.concept.vector,
                            strength=edit.strength,
                            positions="last",
                        )
                    )
            with ActivationEditor(model, runtime_edits, layer_path=layer_path):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            token = _next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            token_id = int(token.item())
            if eos_token_id is not None and token_id == eos_token_id:
                break
            new_tokens.append(token_id)
            input_ids = torch.cat([input_ids, token.to(device)], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                    dim=1,
                )

    full_ids = input_ids[0].detach().cpu()
    generated_ids = torch.tensor(new_tokens, dtype=torch.long)
    return {
        "text": tokenizer.decode(full_ids.tolist(), skip_special_tokens=True),
        "generated_text": tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True),
        "prompt": prompt,
        "model_prompt": model_prompt,
    }

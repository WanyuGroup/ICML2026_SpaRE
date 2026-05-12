from __future__ import annotations

import argparse
import json


PUBLIC_COMMANDS = (
    "finetune",
    "collect-activations",
    "train-sae",
    "extract-local",
    "extract-global",
    "generate",
)


def _load_hf(model_name_or_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    return model, tokenizer


def cmd_finetune(args: argparse.Namespace) -> None:
    from spare_molgen.finetune import finetune_causal_lm

    finetune_causal_lm(
        model_name_or_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        prompt_column=args.prompt_column,
        target_column=args.target_column,
        template=args.template,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        limit=args.limit,
    )


def cmd_collect_activations(args: argparse.Namespace) -> None:
    from spare_molgen.activation_store import collect_activation_shards

    model, tokenizer = _load_hf(args.model)
    collect_activation_shards(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data,
        output_dir=args.output,
        layers=args.layers,
        text_column=args.text_column,
        prompt_column=args.prompt_column,
        target_column=args.target_column,
        template=args.template,
        batch_size=args.batch_size,
        max_length=args.max_length,
        selector=args.selector,
        shard_size=args.shard_size,
        limit=args.limit,
        device=args.device,
    )


def cmd_train_sae(args: argparse.Namespace) -> None:
    from spare_molgen.sae import train_sae_from_shards

    train_sae_from_shards(
        activations=args.activations,
        output=args.output,
        input_dim=args.input_dim,
        expansion_factor=args.expansion_factor,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l1=args.l1,
        optimizer=args.optimizer,
        code_normalization=args.code_normalization,
        device=args.device,
    )


def cmd_extract_local(args: argparse.Namespace) -> None:
    from spare_molgen.concepts import build_local_concept
    from spare_molgen.sae import SparseAutoencoder
    from spare_molgen.tokenizer_utils import require_tokens_in_tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    require_tokens_in_tokenizer(tokenizer, [args.token], context="local concept extraction")
    sae = SparseAutoencoder.load(args.sae, map_location=args.device or "cpu")
    if args.device:
        sae.to(args.device)
    concept = build_local_concept(
        sae=sae,
        activations=args.activations,
        name=args.name,
        layer=args.layer,
        token=args.token,
        threshold=args.threshold,
        min_fraction=args.min_fraction,
        batch_size=args.batch_size,
    )
    concept.save(args.output)


def cmd_extract_global(args: argparse.Namespace) -> None:
    from spare_molgen.concepts import build_global_concept
    from spare_molgen.sae import SparseAutoencoder

    sae = SparseAutoencoder.load(args.sae, map_location=args.device or "cpu")
    if args.device:
        sae.to(args.device)
    concept = build_global_concept(
        sae=sae,
        positive_activations=args.positive_activations,
        negative_activations=args.negative_activations,
        name=args.name,
        layer=args.layer,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    concept.save(args.output)


def cmd_generate(args: argparse.Namespace) -> None:
    from spare_molgen.concepts import ConceptVector
    from spare_molgen.generation import ConceptEdit, generate_with_edits

    model, tokenizer = _load_hf(args.model)
    concepts = [ConceptVector.load(path) for path in args.concept]
    if args.local_step is None and any(concept.kind == "local" for concept in concepts):
        raise SystemExit("Local concepts require --local-step so the edit is applied at one target step.")
    edits = [
        ConceptEdit(
            concept=concept,
            strength=args.strength[index] if index < len(args.strength) else args.strength[-1],
            local_step=args.local_step,
        )
        for index, concept in enumerate(concepts)
    ]
    result = generate_with_edits(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        edits=edits,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        layer_path=args.layer_path,
        seed=args.seed,
        device=args.device,
        prompt_template=None if args.raw_prompt else args.prompt_template,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spare",
        description=(
            "SpaRE public workflow: fine-tune, collect activations, train SAE, "
            "extract local/global control vectors, and run controlled generation."
        ),
        epilog="Public commands: " + ", ".join(PUBLIC_COMMANDS),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    finetune = sub.add_parser("finetune")
    finetune.add_argument("--model", required=True)
    finetune.add_argument("--data", required=True)
    finetune.add_argument("--output", required=True)
    finetune.add_argument("--prompt-column", default="text")
    finetune.add_argument("--target-column", default="group_selfies")
    finetune.add_argument("--template", default="Text: {prompt}\nMolecule: {target}")
    finetune.add_argument("--epochs", type=float, default=2.0)
    finetune.add_argument("--batch-size", type=int, default=64)
    finetune.add_argument("--gradient-accumulation-steps", type=int, default=1)
    finetune.add_argument("--lr", type=float, default=5e-5)
    finetune.add_argument("--max-length", type=int, default=1024)
    finetune.add_argument("--weight-decay", type=float, default=0.0)
    finetune.add_argument("--warmup-steps", type=int, default=0)
    finetune.add_argument("--limit", type=int)
    finetune.set_defaults(func=cmd_finetune)

    collect = sub.add_parser("collect-activations")
    collect.add_argument("--model", required=True)
    collect.add_argument("--data", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--layers", type=int, nargs="+", required=True)
    collect.add_argument("--text-column")
    collect.add_argument("--prompt-column", default="text")
    collect.add_argument("--target-column", default="group_selfies")
    collect.add_argument("--template", default="Text: {prompt}\nMolecule: {target}")
    collect.add_argument("--batch-size", type=int, default=4)
    collect.add_argument("--max-length", type=int, default=1024)
    collect.add_argument("--selector", choices=["all", "last"], default="all")
    collect.add_argument("--shard-size", type=int, default=8192)
    collect.add_argument("--limit", type=int)
    collect.add_argument("--device")
    collect.set_defaults(func=cmd_collect_activations)

    train_sae = sub.add_parser("train-sae")
    train_sae.add_argument("--activations", required=True)
    train_sae.add_argument("--output", required=True)
    train_sae.add_argument("--input-dim", type=int)
    train_sae.add_argument("--expansion-factor", type=int, default=40)
    train_sae.add_argument("--latent-dim", type=int)
    train_sae.add_argument("--epochs", type=int, default=8)
    train_sae.add_argument("--batch-size", type=int, default=1024)
    train_sae.add_argument("--lr", type=float, default=1e-4)
    train_sae.add_argument("--l1", type=float, default=1e-5)
    train_sae.add_argument("--optimizer", default="adamw")
    train_sae.add_argument("--code-normalization", choices=["l2", "max", "none"], default="l2")
    train_sae.add_argument("--device")
    train_sae.set_defaults(func=cmd_train_sae)

    local = sub.add_parser("extract-local")
    local.add_argument("--sae", required=True)
    local.add_argument("--tokenizer", required=True)
    local.add_argument("--token", required=True)
    local.add_argument("--activations", required=True)
    local.add_argument("--output", required=True)
    local.add_argument("--name", required=True)
    local.add_argument("--layer", type=int, required=True)
    local.add_argument("--threshold", type=float, default=0.5)
    local.add_argument("--min-fraction", type=float, default=1.0)
    local.add_argument("--batch-size", type=int, default=1024)
    local.add_argument("--device")
    local.set_defaults(func=cmd_extract_local)

    global_cmd = sub.add_parser("extract-global")
    global_cmd.add_argument("--sae", required=True)
    global_cmd.add_argument("--positive-activations", required=True)
    global_cmd.add_argument("--negative-activations", required=True)
    global_cmd.add_argument("--output", required=True)
    global_cmd.add_argument("--name", required=True)
    global_cmd.add_argument("--layer", type=int, required=True)
    global_cmd.add_argument("--threshold", type=float, default=0.5)
    global_cmd.add_argument("--batch-size", type=int, default=1024)
    global_cmd.add_argument("--device")
    global_cmd.set_defaults(func=cmd_extract_global)

    generate = sub.add_parser("generate")
    generate.add_argument("--model", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--prompt-template", default="Text: {prompt}\nMolecule:")
    generate.add_argument("--raw-prompt", action="store_true")
    generate.add_argument("--concept", nargs="+", required=True)
    generate.add_argument("--strength", type=float, nargs="+", default=[1.0])
    generate.add_argument("--local-step", type=int)
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--layer-path")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--device")
    generate.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

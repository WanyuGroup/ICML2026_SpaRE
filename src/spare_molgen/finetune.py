from __future__ import annotations

from pathlib import Path

from spare_molgen.data import (
    DEFAULT_TRAIN_TEMPLATE,
    collect_group_selfies_tokens,
    iter_text_to_molecule_examples,
)
from spare_molgen.tokenizer_utils import add_group_selfies_tokens


def finetune_causal_lm(
    model_name_or_path: str,
    data_path: str,
    output_dir: str,
    prompt_column: str = "text",
    target_column: str = "group_selfies",
    template: str = DEFAULT_TRAIN_TEMPLATE,
    epochs: float = 2.0,
    batch_size: int = 64,
    learning_rate: float = 5e-5,
    max_length: int = 1024,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    limit: int | None = None,
) -> None:
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    texts = list(
        iter_text_to_molecule_examples(
            data_path,
            prompt_column=prompt_column,
            target_column=target_column,
            template=template,
            limit=limit,
        )
    )
    if not texts:
        raise ValueError(f"No training text found in {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    group_selfies_tokens = collect_group_selfies_tokens(texts)
    added_tokens = add_group_selfies_tokens(tokenizer, group_selfies_tokens)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if added_tokens or len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        logging_steps=25,
        save_strategy="epoch",
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

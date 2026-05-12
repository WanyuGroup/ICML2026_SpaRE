from __future__ import annotations

from collections.abc import Iterable

from spare_molgen.data import collect_group_selfies_tokens


def tokenizer_vocab(tokenizer) -> set[str]:
    if hasattr(tokenizer, "get_vocab"):
        return set(tokenizer.get_vocab())
    raise TypeError("Tokenizer must expose get_vocab()")


def missing_tokens(tokenizer, tokens: Iterable[str]) -> list[str]:
    vocab = tokenizer_vocab(tokenizer)
    return sorted({token for token in tokens if token not in vocab})


def require_tokens_in_tokenizer(tokenizer, tokens: Iterable[str], context: str) -> None:
    missing = missing_tokens(tokenizer, tokens)
    if missing:
        preview = ", ".join(missing[:20])
        extra = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        raise ValueError(
            f"{context}: Group SELFIES token(s) not found in tokenizer: {preview}{extra}. "
            "Run fine-tuning with the same Group SELFIES data first so these bracket tokens "
            "are added to the tokenizer and model embeddings."
        )


def require_group_selfies_tokens_in_tokenizer(tokenizer, texts: Iterable[str], context: str) -> None:
    require_tokens_in_tokenizer(tokenizer, collect_group_selfies_tokens(texts), context=context)


def add_group_selfies_tokens(tokenizer, tokens: Iterable[str]) -> int:
    new_tokens = missing_tokens(tokenizer, tokens)
    if not new_tokens:
        return 0
    try:
        from tokenizers import AddedToken
    except ImportError:
        from transformers import AddedToken

    added_tokens = [
        AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for token in new_tokens
    ]
    return int(tokenizer.add_tokens(added_tokens))


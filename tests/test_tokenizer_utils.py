from __future__ import annotations

import pytest

from spare_molgen.tokenizer_utils import (
    add_group_selfies_tokens,
    missing_tokens,
    require_group_selfies_tokens_in_tokenizer,
    require_tokens_in_tokenizer,
)


class FakeTokenizer:
    def __init__(self, tokens):
        self.tokens = set(tokens)
        self.added = []

    def get_vocab(self):
        return {token: index for index, token in enumerate(sorted(self.tokens))}

    def add_tokens(self, tokens):
        as_strings = [getattr(token, "content", str(token)) for token in tokens]
        self.added.extend(as_strings)
        self.tokens.update(as_strings)
        return len(as_strings)


def test_add_group_selfies_tokens_adds_missing_bracket_tokens():
    tokenizer = FakeTokenizer(["[C]"])

    added = add_group_selfies_tokens(tokenizer, ["[C]", "[*]", "[=O]"])

    assert added == 2
    assert missing_tokens(tokenizer, ["[C]", "[*]", "[=O]"]) == []


def test_require_tokens_in_tokenizer_raises_for_missing_token():
    tokenizer = FakeTokenizer(["[C]"])

    with pytest.raises(ValueError, match=r"\[\*\]"):
        require_tokens_in_tokenizer(tokenizer, ["[*]"], context="local concept extraction")


def test_require_group_selfies_tokens_in_tokenizer_accepts_present_tokens():
    tokenizer = FakeTokenizer(["[C]", "[*]", "[Branch1]"])

    require_group_selfies_tokens_in_tokenizer(
        tokenizer,
        ["[C][*]", "Text without groups", "[Branch1]"],
        context="generation prompt",
    )

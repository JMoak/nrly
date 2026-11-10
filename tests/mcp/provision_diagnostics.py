"""Build a deterministic local transformer for actual MCP diagnostic acceptance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tokenizers
import torch
import transformers


def main() -> None:
    root = Path(sys.argv[1]) / "reference-model"
    words = ["[UNK]", "[PAD]", "[EOS]", "hello", "world", "yes", "no"]
    raw = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {word: index for index, word in enumerate(words)},
            unk_token="[UNK]",
        )
    )
    raw.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=len(words),
                n_positions=64,
                n_embd=8,
                n_layer=1,
                n_head=1,
                bos_token_id=2,
                eos_token_id=2,
                pad_token_id=1,
            )
        )
    model.save_pretrained(root, safe_serialization=True)
    tokenizer.save_pretrained(root)
    print(json.dumps({"backend": "huggingface", "model_id": str(root), "revision": "0" * 40}))


if __name__ == "__main__":
    main()

"""Offline tiny tokenizer + randomly-initialised Qwen3 backbone for tests / smoke runs."""
from __future__ import annotations

import torch

_CORPUS = [
    "Select the intent that best matches the utterance.",
    "alarm_set lists_remove iot_hue_lightchange play_music weather_query general_quirky",
    "state instruction criteria candidate tool route agent decision json true false null",
    "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 {}[]\":,._-/\\n",
    "the quick brown fox jumps over the lazy dog and picks the best option for the user",
]


def make_tiny_tokenizer(vocab_size: int = 400):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(_CORPUS * 4, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")


def make_tiny_backbone(tiny_cfg: dict, vocab_size: int, dtype=torch.float32):
    from transformers import Qwen3Config, AutoModel

    c = dict(tiny_cfg)
    seed = c.pop("seed", 0)
    conf = Qwen3Config(vocab_size=vocab_size, max_position_embeddings=4096, tie_word_embeddings=False, **c)
    g = torch.random.fork_rng()
    with g:
        torch.manual_seed(seed)
        model = AutoModel.from_config(conf)
    return model.to(dtype)

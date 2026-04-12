import sys
from pathlib import Path
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

@pytest.fixture
def tiny_config():
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=128,
    )

@pytest.fixture
def tiny_model(tiny_config):
    model = LlamaForCausalLM(tiny_config)
    model.eval()
    return model

@pytest.fixture
def tiny_tokenizer(tiny_config):
    vocab = {f"tok{i}": i for i in range(tiny_config.vocab_size)}
    tok_model = models.WordLevel(vocab=vocab, unk_token="tok0")
    tokenizer = Tokenizer(tok_model)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    with TemporaryDirectory() as d:
        path = Path(d) / "tokenizer.json"
        tokenizer.save(str(path))
        yield PreTrainedTokenizerFast(tokenizer_file=str(path))

import sys
from pathlib import Path
import pytest
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture(autouse=True)
def _isolate_persistence_state(tmp_path, monkeypatch):  # pyright: ignore[reportUnusedFunction]
    """Redirect persistence.DEFAULT_STATE_PATH to a tmp file for every
    test. Without this, any test that exercises a route handler via
    TestClient calls persistence.persist(mgr) and trampling the user's
    real .cache/gui_sessions.json on the dev box."""
    from gui.backend import persistence
    monkeypatch.setattr(
        persistence,
        "DEFAULT_STATE_PATH",
        tmp_path / "test_gui_sessions.json",
    )

@pytest.fixture
def tiny_config():
    return LlamaConfig(
        vocab_size=64,  # pyright: ignore[reportCallIssue]
        hidden_size=32,  # pyright: ignore[reportCallIssue]
        intermediate_size=64,  # pyright: ignore[reportCallIssue]
        num_hidden_layers=4,  # pyright: ignore[reportCallIssue]
        num_attention_heads=4,  # pyright: ignore[reportCallIssue]
        max_position_embeddings=128,  # pyright: ignore[reportCallIssue]
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

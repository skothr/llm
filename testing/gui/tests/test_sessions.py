import pytest
import torch
from gui.backend.sessions import SessionManager, SessionInfo

class TestSessionManager:
    def test_register_and_list(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("test-model", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        sessions = mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].name == "test-model"
        assert sessions[0].model_id == "test/tiny"

    def test_get_session(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        assert info.model is tiny_model
        assert info.tokenizer is tiny_tokenizer

    def test_get_missing_raises(self):
        mgr = SessionManager()
        with pytest.raises(KeyError, match="s1"):
            mgr.get("s1")

    def test_delete(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        mgr.delete("s1")
        assert mgr.list_sessions() == []

    def test_duplicate_name_raises(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        with pytest.raises(ValueError, match="already exists"):
            mgr.register("s1", tiny_model, tiny_tokenizer,
                          model_id="test/tiny", mode="eval")

    def test_validate_slug_rejects_invalid(self):
        mgr = SessionManager()
        with pytest.raises(ValueError, match="slug"):
            mgr.validate_name("Has Spaces")
        with pytest.raises(ValueError, match="slug"):
            mgr.validate_name("UPPERCASE")
        with pytest.raises(ValueError, match="slug"):
            mgr.validate_name("")

    def test_validate_slug_accepts_valid(self):
        mgr = SessionManager()
        mgr.validate_name("my-model-1")
        mgr.validate_name("baseline")

    def test_snapshot_and_undo(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        original_weight = tiny_model.model.layers[0].mlp.gate_proj.weight.clone()
        mgr.snapshot("s1")
        tiny_model.model.layers[0].mlp.gate_proj.weight.data.zero_()
        mgr.undo("s1")
        restored = tiny_model.model.layers[0].mlp.gate_proj.weight
        assert torch.allclose(restored, original_weight)

    def test_undo_without_snapshot_raises(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        with pytest.raises(ValueError, match="No snapshot"):
            mgr.undo("s1")

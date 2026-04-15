import pytest
import torch
from unittest.mock import patch
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

    def test_validate_name_rejects_invalid(self):
        mgr = SessionManager()
        with pytest.raises(ValueError, match="Invalid name"):
            mgr.validate_name("Has Spaces")
        with pytest.raises(ValueError, match="Invalid name"):
            mgr.validate_name("")
        with pytest.raises(ValueError, match="Invalid name"):
            mgr.validate_name("-starts-with-hyphen")

    def test_validate_name_accepts_valid(self):
        mgr = SessionManager()
        mgr.validate_name("my-model-1")
        mgr.validate_name("baseline")
        mgr.validate_name("TinyLlama_v1.0")
        mgr.validate_name("Model.3B~test")

    def test_snapshot_and_undo(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        original_weight = tiny_model.model.layers[0].mlp.gate_proj.weight.clone()
        mgr.snapshot("s1")
        tiny_model.model.layers[0].mlp.gate_proj.weight.data.zero_()
        mgr.undo("s1")
        restored_model = mgr.get("s1").model
        restored = restored_model.model.layers[0].mlp.gate_proj.weight
        assert torch.allclose(restored, original_weight)

    def test_multi_undo(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        w0 = tiny_model.model.layers[0].mlp.gate_proj.weight.clone()

        mgr.snapshot("s1")
        tiny_model.model.layers[0].mlp.gate_proj.weight.data.zero_()
        w1 = tiny_model.model.layers[0].mlp.gate_proj.weight.clone()

        mgr.snapshot("s1")
        tiny_model.model.layers[0].mlp.gate_proj.weight.data.fill_(1.0)

        assert mgr.get("s1").undo_depth == 2

        mgr.undo("s1")
        restored = mgr.get("s1").model.model.layers[0].mlp.gate_proj.weight
        assert torch.allclose(restored, w1)

        mgr.undo("s1")
        restored = mgr.get("s1").model.model.layers[0].mlp.gate_proj.weight
        assert torch.allclose(restored, w0)

        assert mgr.get("s1").undo_depth == 0

    def test_register_sets_eval_mode(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        tiny_model.train()  # force training mode
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        assert not info.model.training

    def test_register_disables_grad(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        for p in tiny_model.parameters():
            p.requires_grad = True  # force grads on
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        assert all(not p.requires_grad for p in info.model.parameters())

    def test_delete_calls_gc_collect(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        with patch("gui.backend.sessions.gc.collect") as mock_gc:
            mgr.delete("s1")
            mock_gc.assert_called_once()

    def test_undo_without_snapshot_raises(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        with pytest.raises(ValueError, match="No undo history"):
            mgr.undo("s1")

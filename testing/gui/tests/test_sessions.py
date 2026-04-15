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

    def test_stage_op(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        info.stage_op("zero_mlp", {"layer": 0})
        assert len(info.pending_ops) == 1
        assert info.pending_ops[0]["operation"] == "zero_mlp"
        assert info.pending_ops[0]["params"] == {"layer": 0}

    def test_undo_op_pops_last(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        info.stage_op("zero_mlp", {"layer": 0})
        info.stage_op("remove_layers", {"layer_indices": [1]})
        popped = info.undo_op()
        assert popped["operation"] == "remove_layers"
        assert len(info.pending_ops) == 1

    def test_undo_op_empty_raises(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        with pytest.raises(ValueError, match="No pending operations"):
            info.undo_op()

    def test_clear_pending(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        info.stage_op("zero_mlp", {"layer": 0})
        info.stage_op("zero_mlp", {"layer": 1})
        info.clear_pending()
        assert len(info.pending_ops) == 0

    def test_record_applied_ops(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        ops = [
            {"operation": "zero_mlp", "params": {"layer": 0}},
            {"operation": "remove_layers", "params": {"layer_indices": [1]}},
        ]
        info.record_applied(ops)
        assert len(info.applied_ops) == 2
        assert info.applied_ops[0]["operation"] == "zero_mlp"

    def test_revert_moves_applied_to_pending(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        ops = [
            {"operation": "zero_mlp", "params": {"layer": 0}},
            {"operation": "remove_layers", "params": {"layer_indices": [1]}},
        ]
        info.record_applied(ops)
        info.revert()
        assert len(info.applied_ops) == 0
        assert len(info.pending_ops) == 2
        assert len(info.op_history) == 1
        assert info.pending_ops[0]["operation"] == "zero_mlp"

    def test_revert_empty_raises(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        with pytest.raises(ValueError, match="No applied operations"):
            info.revert()

    def test_has_pending(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        assert not info.has_pending
        info.stage_op("zero_mlp", {"layer": 0})
        assert info.has_pending

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


import pytest
import torch
from unittest.mock import patch
from transformers import LlamaConfig, LlamaForCausalLM
from gui.backend.sessions import (
    SessionManager, SessionInfo, translate_to_current, validate_original_indices,
)

def _make_model(n: int) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n,
        num_attention_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model

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


class TestTranslateToCurrentIndex:
    def test_remove_layers_identity(self):
        layer_map = [0, 1, 2, 3, 4, 5, 6, 7]
        result = translate_to_current("remove_layers", {"layer_indices": [5]}, layer_map)
        assert result == {"layer_indices": [5]}

    def test_remove_layers_after_prior_removal(self):
        layer_map = [0, 1, 2, 3, 4, 6, 7]  # original 5 already removed
        result = translate_to_current("remove_layers", {"layer_indices": [6]}, layer_map)
        assert result == {"layer_indices": [5]}  # original 6 is now at current 5

    def test_keep_layers(self):
        layer_map = [0, 1, 2, 3, 4, 5, 6, 7]
        result = translate_to_current("keep_layers", {"layer_indices": [0, 1, 5, 7]}, layer_map)
        assert result == {"layer_indices": [0, 1, 5, 7]}

    def test_keep_layers_after_removal(self):
        layer_map = [0, 2, 4, 6]  # kept even layers only
        result = translate_to_current("keep_layers", {"layer_indices": [0, 4]}, layer_map)
        assert result == {"layer_indices": [0, 2]}

    def test_zero_heads(self):
        layer_map = [0, 1, 2, 3, 4, 6, 7]
        result = translate_to_current("zero_heads", {"layer": 6, "heads": [0, 2]}, layer_map)
        assert result == {"layer": 5, "heads": [0, 2]}

    def test_scale_heads(self):
        layer_map = [0, 1, 2, 3, 4, 6, 7]
        result = translate_to_current("scale_heads", {"layer": 7, "heads": [1], "factor": 0.5}, layer_map)
        assert result == {"layer": 6, "heads": [1], "factor": 0.5}

    def test_swap_layers(self):
        layer_map = [0, 2, 4, 6]
        result = translate_to_current("swap_layers", {"i": 2, "j": 6}, layer_map)
        assert result == {"i": 1, "j": 3}

    def test_swap_heads(self):
        layer_map = [0, 1, 2, 4, 5]
        result = translate_to_current("swap_heads", {"layer": 4, "h1": 0, "h2": 3}, layer_map)
        assert result == {"layer": 3, "h1": 0, "h2": 3}

    def test_zero_mlp(self):
        layer_map = [0, 1, 2, 3, 4, 6, 7]
        result = translate_to_current("zero_mlp", {"layer": 7}, layer_map)
        assert result == {"layer": 6}

    def test_zero_attention(self):
        layer_map = [0, 1, 2, 3, 4, 6, 7]
        result = translate_to_current("zero_attention", {"layer": 6}, layer_map)
        assert result == {"layer": 5}

    def test_duplicate_layer_src_translated_dst_unchanged(self):
        layer_map = [0, 1, 2, 4, 5]
        result = translate_to_current("duplicate_layer", {"src": 4, "dst": 1}, layer_map)
        assert result == {"src": 3, "dst": 1}

    def test_reorder_layers(self):
        layer_map = [0, 2, 4, 6]
        result = translate_to_current("reorder_layers", {"new_order": [6, 4, 2, 0]}, layer_map)
        assert result == {"new_order": [3, 2, 1, 0]}

    def test_missing_original_raises(self):
        layer_map = [0, 1, 2, 4, 5]  # original 3 removed
        with pytest.raises(ValueError, match="not found in current model"):
            translate_to_current("zero_mlp", {"layer": 3}, layer_map)


class TestValidateOriginalIndices:
    def test_valid_remove_layers(self):
        validate_original_indices("remove_layers", {"layer_indices": [0, 5, 7]}, 8)

    def test_remove_layers_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_original_indices("remove_layers", {"layer_indices": [8]}, 8)

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_original_indices("zero_mlp", {"layer": -1}, 8)

    def test_valid_layer_ops(self):
        for op in ("zero_heads", "scale_heads", "zero_mlp", "zero_attention", "swap_heads"):
            validate_original_indices(op, {"layer": 7}, 8)

    def test_swap_layers_both_checked(self):
        validate_original_indices("swap_layers", {"i": 0, "j": 7}, 8)
        with pytest.raises(ValueError, match="out of range"):
            validate_original_indices("swap_layers", {"i": 0, "j": 8}, 8)

    def test_duplicate_layer_src_checked(self):
        validate_original_indices("duplicate_layer", {"src": 7, "dst": 0}, 8)
        with pytest.raises(ValueError, match="out of range"):
            validate_original_indices("duplicate_layer", {"src": 8, "dst": 0}, 8)

    def test_reorder_layers_all_checked(self):
        validate_original_indices("reorder_layers", {"new_order": [7, 6, 5, 4, 3, 2, 1, 0]}, 8)
        with pytest.raises(ValueError, match="out of range"):
            validate_original_indices("reorder_layers", {"new_order": [0, 1, 2, 9]}, 8)


class TestStagingValidation:
    def test_stage_rejects_out_of_range(self):
        model = _make_model(8)
        mgr = SessionManager()
        info = mgr.register("test", model, None, model_id="test/model", mode="inspect")
        with pytest.raises(ValueError, match="out of range"):
            info.stage_op("remove_layers", {"layer_indices": [8]})

    def test_stage_accepts_valid_original(self):
        model = _make_model(8)
        mgr = SessionManager()
        info = mgr.register("test", model, None, model_id="test/model", mode="inspect")
        info.stage_op("remove_layers", {"layer_indices": [0, 7]})
        assert len(info.pending_ops) == 1

    def test_delete_op_removes_by_index(self):
        model = _make_model(4)
        mgr = SessionManager()
        info = mgr.register("test", model, None, model_id="test/model", mode="inspect")
        info.stage_op("zero_mlp", {"layer": 0})
        info.stage_op("zero_mlp", {"layer": 1})
        info.stage_op("zero_mlp", {"layer": 2})
        removed = info.delete_op(1)
        assert removed["params"]["layer"] == 1
        assert len(info.pending_ops) == 2
        assert info.pending_ops[0]["params"]["layer"] == 0
        assert info.pending_ops[1]["params"]["layer"] == 2

    def test_delete_op_out_of_range(self):
        model = _make_model(4)
        mgr = SessionManager()
        info = mgr.register("test", model, None, model_id="test/model", mode="inspect")
        with pytest.raises(IndexError, match="out of range"):
            info.delete_op(0)

    def test_num_original_layers(self):
        model = _make_model(8)
        mgr = SessionManager()
        info = mgr.register("test", model, None, model_id="test/model", mode="inspect")
        assert info.num_original_layers == 8

    def test_duplicate_remove_layers_rejected(self):
        model = _make_model(8)
        mgr = SessionManager()
        info = mgr.register("test", model, None, model_id="test/model", mode="inspect")
        info.stage_op("remove_layers", {"layer_indices": [3, 5]})
        with pytest.raises(ValueError, match="already staged for removal"):
            info.stage_op("remove_layers", {"layer_indices": [5]})

    def test_remove_layers_disjoint_allowed(self):
        model = _make_model(8)
        mgr = SessionManager()
        info = mgr.register("test", model, None, model_id="test/model", mode="inspect")
        info.stage_op("remove_layers", {"layer_indices": [3]})
        info.stage_op("remove_layers", {"layer_indices": [5]})
        assert len(info.pending_ops) == 2

    def test_op_on_removed_layer_rejected(self):
        model = _make_model(8)
        mgr = SessionManager()
        info = mgr.register("test", model, None, model_id="test/model", mode="inspect")
        info.stage_op("remove_layers", {"layer_indices": [5]})
        with pytest.raises(ValueError, match="staged for removal"):
            info.stage_op("zero_mlp", {"layer": 5})


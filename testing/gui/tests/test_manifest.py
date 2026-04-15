import json
from pathlib import Path
from gui.backend.manifest import SurgeryManifest


class TestSurgeryManifest:
    def test_from_session_state(self):
        m = SurgeryManifest(
            model_id="meta-llama/Llama-2-7b",
            original_num_layers=32,
            applied_ops=[
                {"operation": "remove_layers", "params": {"layer_indices": [0, 1]}},
                {"operation": "zero_mlp", "params": {"layer": 15}},
            ],
            layer_map=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        )
        assert m.model_id == "meta-llama/Llama-2-7b"
        assert m.original_num_layers == 32
        assert len(m.applied_ops) == 2
        assert len(m.layer_map) == 30

    def test_to_dict_roundtrip(self):
        m = SurgeryManifest(
            model_id="test/model",
            original_num_layers=8,
            applied_ops=[{"operation": "zero_mlp", "params": {"layer": 3}}],
            layer_map=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        d = m.to_dict()
        m2 = SurgeryManifest.from_dict(d)
        assert m2.model_id == m.model_id
        assert m2.original_num_layers == m.original_num_layers
        assert m2.applied_ops == m.applied_ops
        assert m2.layer_map == m.layer_map

    def test_save_and_load(self, tmp_path):
        m = SurgeryManifest(
            model_id="test/model",
            original_num_layers=4,
            applied_ops=[],
            layer_map=[0, 1, 2, 3],
        )
        path = tmp_path / "manifest.json"
        m.save(path)
        assert path.exists()

        loaded = SurgeryManifest.load(path)
        assert loaded.model_id == m.model_id
        assert loaded.layer_map == m.layer_map

    def test_save_file_is_valid_json(self, tmp_path):
        m = SurgeryManifest(
            model_id="test/model",
            original_num_layers=4,
            applied_ops=[{"operation": "swap_layers", "params": {"i": 0, "j": 3}}],
            layer_map=[3, 1, 2, 0],
        )
        path = tmp_path / "manifest.json"
        m.save(path)
        data = json.loads(path.read_text())
        assert data["model_id"] == "test/model"
        assert data["applied_ops"][0]["operation"] == "swap_layers"

    def test_empty_manifest(self):
        m = SurgeryManifest(
            model_id="test/model",
            original_num_layers=4,
            applied_ops=[],
            layer_map=[0, 1, 2, 3],
        )
        d = m.to_dict()
        assert d["applied_ops"] == []
        assert d["original_num_layers"] == 4

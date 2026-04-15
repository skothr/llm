import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SurgeryManifest:
    model_id: str
    original_num_layers: int
    applied_ops: list = field(default_factory=list)
    layer_map: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "original_num_layers": self.original_num_layers,
            "applied_ops": list(self.applied_ops),
            "layer_map": list(self.layer_map),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SurgeryManifest":
        return cls(
            model_id=data["model_id"],
            original_num_layers=data["original_num_layers"],
            applied_ops=data.get("applied_ops", []),
            layer_map=data.get("layer_map", []),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "SurgeryManifest":
        return cls.from_dict(json.loads(path.read_text()))

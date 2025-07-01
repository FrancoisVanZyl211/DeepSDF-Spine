from __future__ import annotations
from dataclasses import dataclass, asdict
import json, tempfile, uuid, os
from types import SimpleNamespace
from pathlib import Path

# ----------------------------------------------------------------------
@dataclass
class TrainSettings:
    # ---------- file paths ----------
    multi_shape_file: str = ""     # path to .npy with points
    out_model:     str = ""     # where the .pth will be saved

    # ---------- hyper-parameters ----------
    train_split: float = 0.8
    latent_dim:  int   = 64
    hidden_dim: int = 512
    dropout     : float = 0.10
    lr:          float = 1e-4
    weight_decay:float = 1e-4
    epochs:      int   = 20
    batch_size:  int   = 512
    sample_std:  float = 0.05
    n_samples:   int   = 10
    grid_N:      int   = 128
    max_xyz:     float = 1.0

    # ------------------------------------------------------------------
    # persistence helpers
    # ------------------------------------------------------------------
    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=indent))

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainSettings":
        return cls(**json.loads(Path(path).read_text()))

    def to_json_temp(self) -> str:
        tmp = os.path.join(tempfile.gettempdir(), f"ts_{uuid.uuid4().hex}.json")
        self.to_json(tmp)
        return tmp

    def to_namespace(self):
        return SimpleNamespace(**asdict(self))
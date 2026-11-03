"""Load a model from a training checkpoint (state.pt), a run directory (uses 'latest') or an exported base_v1.pt."""
from pathlib import Path

import torch

from src.model.model import DecoderLM, ModelConfig


def resolve(path):
    p = Path(path)
    if p.is_dir():
        if (p / "latest").exists():
            p = p / (p / "latest").read_text().strip()
        p = p / "state.pt"
    return p


def load_model(path, device="cpu"):
    st = torch.load(resolve(path), map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(st["config"]["model"])
    model = DecoderLM(cfg)
    model.load_state_dict(st["model"])
    model.to(device).eval()
    info = {k: st.get(k) for k in ("step", "tokens_seen")}
    return model, st["config"], info


def export_weights(run_dir, out_path, extra=None):
    """Model weights + config only (~160 MB): the frozen hand-off checkpoint (e.g. base_v1.pt)."""
    st = torch.load(resolve(run_dir), map_location="cpu", weights_only=False)
    out = {"model": st["model"], "config": st["config"], "step": st["step"], "tokens_seen": st["tokens_seen"]}
    out.update(extra or {})
    torch.save(out, out_path)
    return out_path

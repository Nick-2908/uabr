"""Handbook section 13: tests that must pass before any long training. Runs on CPU in ~1 minute."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.model import DecoderLM, ModelConfig  # noqa: E402

TINY = ModelConfig(vocab_size=500, max_seq_len=64, n_layers=2, d_model=64, n_heads=4, d_ff=256)


def test_parameter_count_40m():
    cfg = ModelConfig.from_dict(yaml.safe_load(open(ROOT / "configs/pretrain.yaml", encoding="utf-8"))["model"])
    n = DecoderLM(cfg).num_params()
    print(f"params = {n:,}")
    assert 38e6 <= n <= 42e6, f"{n:,} outside the 38M-42M contract"


def test_shapes():
    m = DecoderLM(TINY)
    x = torch.randint(0, TINY.vocab_size, (3, 17))
    logits, loss = m(x)
    assert logits.shape == (3, 17, TINY.vocab_size) and loss is None


def test_causal_mask():
    torch.manual_seed(0)
    m = DecoderLM(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, 20))
    x2 = x.clone()
    x2[0, 12:] = (x2[0, 12:] + 7) % TINY.vocab_size  # change only future tokens
    l1, _ = m(x)
    l2, _ = m(x2)
    assert torch.allclose(l1[0, :12], l2[0, :12], atol=1e-5), "earlier logits changed: mask leaks the future"
    assert not torch.allclose(l1[0, 12:], l2[0, 12:]), "later logits should change"


def test_loss_masking_ignores_pad_and_prompt():
    torch.manual_seed(0)
    m = DecoderLM(TINY)
    x = torch.randint(0, TINY.vocab_size, (2, 10))
    y = torch.randint(0, TINY.vocab_size, (2, 10))
    y_masked = y.clone()
    y_masked[:, :6] = -100  # prompt/PAD positions
    logits, loss = m(x, y_masked)
    manual = torch.nn.functional.cross_entropy(logits[:, 6:].reshape(-1, TINY.vocab_size), y[:, 6:].reshape(-1))
    assert torch.allclose(loss, manual, atol=1e-5), "masked (-100) positions contributed to the loss"


def test_tiny_batch_overfit():
    torch.manual_seed(0)
    m = DecoderLM(TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, TINY.vocab_size, (8, 32))
    y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(150):
        _, loss = m(x, y)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"overfit loss {first:.2f} -> {loss.item():.3f}")
    assert loss.item() < 0.1 * first, "tiny batch did not overfit: check labels/shift/optimizer"


def test_generation_stops_at_eos_and_respects_context():
    torch.manual_seed(0)
    m = DecoderLM(TINY)
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))
    out = m.generate(prompt, max_new_tokens=100)  # longer than context: must not crash
    assert out.shape[1] == 105
    first_tok = m.generate(prompt, max_new_tokens=1)[0, -1].item()
    out = m.generate(prompt, max_new_tokens=50, eos_id=first_tok)
    assert out.shape[1] == 6, "decoding did not stop at EOS"


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    subprocess.check_call([sys.executable, str(ROOT / "scripts/make_synthetic_data.py"), "--out_dir", str(d),
                           "--vocab_size", "12000", "--n_train", "200000", "--n_val", "20000"])
    return d


def _run(args):
    from src.train.pretrain import main
    return main(args + ["--device", "cpu"])


def _metrics(run):
    return [json.loads(l) for l in open(run / "metrics.jsonl", encoding="utf-8") if '"loss"' in l]


def test_checkpoint_resume_matches_uninterrupted(tmp_path, synthetic_data):
    cfg = str(ROOT / "configs/tiny_cpu.yaml")
    a, b = tmp_path / "uninterrupted", tmp_path / "resumed"
    _run(["--config", cfg, "--data_dir", str(synthetic_data), "--output_dir", str(a), "--max_steps", "60"])
    _run(["--config", cfg, "--data_dir", str(synthetic_data), "--output_dir", str(b), "--max_steps", "40"])
    _run(["--config", cfg, "--resume", str(b), "--output_dir", str(b), "--max_steps", "60"])
    la = {r["step"]: r["loss"] for r in _metrics(a)}
    lb = {r["step"]: r["loss"] for r in _metrics(b)}
    for s in (50, 60):
        assert abs(la[s] - lb[s]) < 1e-3, f"step {s}: uninterrupted {la[s]} vs resumed {lb[s]}"
    man = json.load(open(b / "manifest.json", encoding="utf-8"))
    assert len(man["sessions"]) == 2 and man["sessions"][1]["start_step"] == 40

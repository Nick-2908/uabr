"""verify the GPU before any training."""
import torch, platform, subprocess
print("python", platform.python_version(), "| torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name} | {p.total_memory/1e9:.1f} GB | compute capability {p.major}.{p.minor}")
    print("bf16 supported:", torch.cuda.is_bf16_supported(), "(T4/P100: use fp16 AMP)")
    try: print(subprocess.check_output(["nvidia-smi"]).decode())
    except Exception: pass
else:
    print("NO GPU")

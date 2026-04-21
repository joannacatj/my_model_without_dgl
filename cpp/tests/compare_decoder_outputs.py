"""一键运行并对比 Python/CUDA decoder 输出是否一致。"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("[RUN]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}")
    return res.stdout


def parse_shape(text):
    m = re.search(r"decoder_output shape=\s*\(([^)]*)\)", text)
    if not m:
        raise RuntimeError("cannot parse decoder_output shape")
    return tuple(int(x.strip()) for x in m.group(1).split(",") if x.strip())


def parse_values(text):
    m = re.search(r"decoder_output values=\s*(\[[^\]]*\]|[^\n]+)", text, re.S)
    if not m:
        raise RuntimeError("cannot parse decoder_output values")
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))]


def max_abs_diff(a, b):
    if len(a) != len(b):
        raise RuntimeError(f"value length mismatch: {len(a)} vs {len(b)}")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/wikics/gin_checkpoint.pth")
    p.add_argument("--config-path", default=".")
    p.add_argument("--export-dir", default="checkpoints/wikics/export_cpp")
    p.add_argument("--python-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--cuda-bin", default="/tmp/decoder_cuda_example")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-compile", action="store_true")
    args = p.parse_args()

    export_dir = Path(args.export_dir)
    manifest = export_dir / "weights_manifest.json"
    weights = export_dir / "graphdecoder_weights.bin"
    config = export_dir / "export_config.json"

    if not args.skip_export:
        run_cmd(
            [
                sys.executable,
                "tools/export_for_cpp.py",
                "--checkpoint",
                args.checkpoint,
                "--config-path",
                args.config_path,
                "--output-dir",
                str(export_dir),
            ]
        )

    py_out = run_cmd(
        [
            sys.executable,
            "cpp/tests/decoder_python_example.py",
            "--checkpoint",
            args.checkpoint,
            "--config-path",
            args.config_path,
            "--device",
            args.python_device,
        ]
    )

    if not args.skip_compile:
        run_cmd(
            [
                "nvcc",
                "-std=c++17",
                "cpp/tests/decoder_cuda_example.cu",
                "cpp/src/decoder/graph_transformer_decoder_cuda.cu",
                "-Icpp/src/decoder",
                "-o",
                args.cuda_bin,
            ]
        )

    cuda_out = run_cmd(
        [
            args.cuda_bin,
            "--manifest",
            str(manifest),
            "--weights",
            str(weights),
            "--config",
            str(config),
        ]
    )

    py_shape = parse_shape(py_out)
    cu_shape = parse_shape(cuda_out)
    py_vals = parse_values(py_out)
    cu_vals = parse_values(cuda_out)

    print(f"python shape: {py_shape}")
    print(f"cuda   shape: {cu_shape}")
    if py_shape != cu_shape:
        print("shape mismatch", file=sys.stderr)
        sys.exit(2)

    mad = max_abs_diff(py_vals, cu_vals)
    print(f"max_abs_diff={mad:.6e}")
    if mad > args.atol:
        print(f"[FAIL] max_abs_diff {mad:.6e} > atol {args.atol:.6e}", file=sys.stderr)
        sys.exit(1)

    print(f"[PASS] max_abs_diff {mad:.6e} <= atol {args.atol:.6e}")


if __name__ == "__main__":
    os_cwd = Path.cwd()
    # 确保从仓库根目录执行子命令，避免相对路径歧义。
    repo_root = Path(__file__).resolve().parents[2]
    if os_cwd != repo_root:
        import os

        os.chdir(repo_root)
    main()

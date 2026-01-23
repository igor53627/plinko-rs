"""
Benchmark GPU hint generation on Modal H100/H200.

Uploads local code directly - no GitHub needed.

Usage:
    # Run benchmark (builds + generates data + benchmarks)
    modal run scripts/modal_bench.py --gpu h100 --scale 0.1

    # List existing datasets
    modal run scripts/modal_bench.py --list-data
"""
import modal

app = modal.App("plinko-gpu-bench")

# Persistent volume for datasets
volume = modal.Volume.from_name("plinko-data", create_if_missing=True)
VOLUME_PATH = "/data"

# CUDA image with Rust + local code
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH", "CUDA_ARCH": "sm_90"})
    # Upload local code directly
    .add_local_dir(
        ".",  # Current directory (plinko-rs)
        remote_path="/app",
        ignore=["target", ".git", ".jj", "tmp", "__pycache__", ".DS_Store", "data", ".beads"]
    )
)


@app.function(
    image=cuda_image,
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def bench_h100(
    scale_percent: float = 0.1,
    seed: int = 42,
    lambda_param: int = 128,
    iterations: int = 10,
):
    """Build and run benchmark on H100."""
    import subprocess
    import os

    os.chdir("/app")

    # Build
    print("=== Building with CUDA ===")
    subprocess.run(["nvcc", "--version"], check=True)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "gen_synthetic"],
        check=True,
    )
    subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
        check=True,
    )

    # Generate data
    data_dir = f"{VOLUME_PATH}/synthetic_{scale_percent}pct_seed{seed}"
    db_path = f"{data_dir}/database.bin"

    if not os.path.exists(db_path):
        print(f"\n=== Generating {scale_percent}% data ===")
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(
            [
                "./target/release/gen_synthetic",
                "--output-dir", data_dir,
                "--scale-percent", str(scale_percent),
                "--seed", str(seed),
            ],
            check=True,
        )
        volume.commit()
    else:
        print(f"\n=== Using existing data: {db_path} ===")

    # Benchmark
    print(f"\n=== Benchmarking on H100 ===")
    result = subprocess.run(
        [
            "./target/release/bench_gpu_hints",
            "--db", db_path,
            "--lambda", str(lambda_param),
            "--iterations", str(iterations),
            "--warmup", "5",
            "--cpu-baseline",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {"gpu": "H100", "scale": scale_percent, "output": result.stdout}


@app.function(
    image=cuda_image,
    gpu="H200",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def bench_h200(
    scale_percent: float = 0.1,
    seed: int = 42,
    lambda_param: int = 128,
    iterations: int = 10,
):
    """Build and run benchmark on H200."""
    import subprocess
    import os

    os.chdir("/app")

    print("=== Building with CUDA ===")
    subprocess.run(["nvcc", "--version"], check=True)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "gen_synthetic"],
        check=True,
    )
    subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
        check=True,
    )

    data_dir = f"{VOLUME_PATH}/synthetic_{scale_percent}pct_seed{seed}"
    db_path = f"{data_dir}/database.bin"

    if not os.path.exists(db_path):
        print(f"\n=== Generating {scale_percent}% data ===")
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(
            [
                "./target/release/gen_synthetic",
                "--output-dir", data_dir,
                "--scale-percent", str(scale_percent),
                "--seed", str(seed),
            ],
            check=True,
        )
        volume.commit()

    print(f"\n=== Benchmarking on H200 ===")
    result = subprocess.run(
        [
            "./target/release/bench_gpu_hints",
            "--db", db_path,
            "--lambda", str(lambda_param),
            "--iterations", str(iterations),
            "--warmup", "5",
            "--cpu-baseline",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {"gpu": "H200", "scale": scale_percent, "output": result.stdout}


@app.function(
    image=cuda_image,
    volumes={VOLUME_PATH: volume},
)
def list_datasets():
    """List available datasets in the volume."""
    import os

    print(f"Datasets in {VOLUME_PATH}:")
    if not os.path.exists(VOLUME_PATH):
        print("  (volume empty)")
        return []

    datasets = []
    for name in sorted(os.listdir(VOLUME_PATH)):
        path = os.path.join(VOLUME_PATH, name)
        if os.path.isdir(path):
            db_file = os.path.join(path, "database.bin")
            if os.path.exists(db_file):
                size = os.path.getsize(db_file)
                print(f"  {name}: {size / 1e6:.2f} MB")
                datasets.append(name)
    return datasets


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    scale: float = 0.1,
    seed: int = 42,
    lambda_param: int = 128,
    iterations: int = 10,
    list_data: bool = False,
):
    """
    Run GPU hint generation benchmark on Modal.

    Examples:
        modal run scripts/modal_bench.py --gpu h100 --scale 0.1
        modal run scripts/modal_bench.py --gpu h200 --scale 1.0
        modal run scripts/modal_bench.py --list-data
    """
    if list_data:
        list_datasets.remote()
        return

    print(f"Plinko GPU Hint Generation Benchmark")
    print(f"=====================================")
    print(f"GPU: {gpu.upper()}")
    print(f"Scale: {scale}% of mainnet (~{scale * 0.83:.1f} MB)")
    print(f"Lambda: {lambda_param}")
    print(f"Iterations: {iterations}")
    print()

    if gpu.lower() == "h100":
        result = bench_h100.remote(scale, seed, lambda_param, iterations)
    elif gpu.lower() == "h200":
        result = bench_h200.remote(scale, seed, lambda_param, iterations)
    else:
        raise ValueError(f"Unknown GPU: {gpu}. Use 'h100' or 'h200'")

    print("\n" + "=" * 50)
    print(result["output"])

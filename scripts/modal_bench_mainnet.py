"""
Benchmark GPU hint generation on Modal H200 with mainnet v3 database.

Usage:
    modal run scripts/modal_bench_mainnet.py --gpus 2
"""
import modal

app = modal.App("plinko-mainnet-bench")

# Persistent volume with mainnet data
volume = modal.Volume.from_name("plinko-data", create_if_missing=False)
VOLUME_PATH = "/data"

# CUDA image with Rust + local code
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH", "CUDA_ARCH": "sm_90"})
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=["target", ".git", ".jj", "tmp", "__pycache__", ".DS_Store", "data", ".beads"]
    )
)


@app.function(
    image=cuda_image,
    gpu="H200:2",  # 2x H200
    volumes={VOLUME_PATH: volume},
    timeout=7200,  # 2 hours for full mainnet
)
def bench_mainnet_2xh200(
    lambda_param: int = 128,
    iterations: int = 5,
    warmup: int = 2,
    max_hints: int = None,
):
    """Build and run benchmark on 2xH200 with mainnet v3 database."""
    import subprocess
    import os
    import json

    os.chdir("/app")

    # Show GPU info
    print("=== GPU Info ===")
    subprocess.run(["nvidia-smi", "-L"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
    print()

    # Build
    print("=== Building with CUDA ===")
    subprocess.run(["nvcc", "--version"], check=True)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
        check=True,
    )

    # Check mainnet data
    db_dir = f"{VOLUME_PATH}/mainnet-v3"
    db_path = f"{db_dir}/database.bin"
    manifest_path = f"{db_dir}/manifest.json"

    if not os.path.exists(db_path):
        print(f"\n=== Generating Mainnet v3 Data (Full Scale) ===")
        # Build generator
        subprocess.run(
            ["cargo", "build", "--release", "-p", "plinko", "--bin", "gen_synthetic"],
            check=True,
        )
        
        os.makedirs(db_dir, exist_ok=True)
        # Generate full mainnet scale (100%)
        # This will produce ~83GB of data (40-byte entries)
        subprocess.run(
            [
                "./target/release/gen_synthetic",
                "--output-dir", db_dir,
                "--scale-percent", "100.0",
                "--seed", "42",
            ],
            check=True,
        )
        volume.commit()
    else:
        print(f"\n=== Using existing data: {db_path} ===")

    db_size = os.path.getsize(db_path)
    print(f"\n=== Mainnet v3 Database ===")
    print(f"Path: {db_path}")
    print(f"Size: {db_size / 1e9:.2f} GB")

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Block: {manifest.get('block_number')}")
        print(f"Entries: {manifest.get('entries', {}).get('total'):,}")
        print(f"Schema: v{manifest.get('schema_version')}")
    print()

    # Build benchmark command
    cmd = [
        "./target/release/bench_gpu_hints",
        "--db", db_path,
        "--lambda", str(lambda_param),
        "--iterations", str(iterations),
        "--warmup", str(warmup),
    ]

    if max_hints:
        cmd.extend(["--max-hints", str(max_hints)])

    print(f"=== Running Benchmark ===")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {
        "gpu": "2xH200",
        "db_size_gb": db_size / 1e9,
        "output": result.stdout,
        "returncode": result.returncode,
    }


@app.function(
    image=cuda_image,
    gpu="H200",  # 1x H200
    volumes={VOLUME_PATH: volume},
    timeout=7200,
)
def bench_mainnet_1xh200(
    lambda_param: int = 128,
    iterations: int = 5,
    warmup: int = 2,
    max_hints: int = None,
):
    """Build and run benchmark on 1xH200 with mainnet v3 database."""
    import subprocess
    import os
    import json

    os.chdir("/app")

    print("=== GPU Info ===")
    subprocess.run(["nvidia-smi", "-L"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
    print()

    print("=== Building with CUDA ===")
    subprocess.run(["nvcc", "--version"], check=True)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
        check=True,
    )

    # Check mainnet data
    db_dir = f"{VOLUME_PATH}/mainnet-v3"
    db_path = f"{db_dir}/database.bin"
    manifest_path = f"{db_dir}/manifest.json"

    if not os.path.exists(db_path):
        print(f"\n=== Generating Mainnet v3 Data (Full Scale) ===")
        subprocess.run(
            ["cargo", "build", "--release", "-p", "plinko", "--bin", "gen_synthetic"],
            check=True,
        )
        os.makedirs(db_dir, exist_ok=True)
        subprocess.run(
            [
                "./target/release/gen_synthetic",
                "--output-dir", db_dir,
                "--scale-percent", "100.0",
                "--seed", "42",
            ],
            check=True,
        )
        volume.commit()
    else:
        print(f"\n=== Using existing data: {db_path} ===")

    db_size = os.path.getsize(db_path)
    print(f"\n=== Mainnet v3 Database ===")
    print(f"Path: {db_path}")
    print(f"Size: {db_size / 1e9:.2f} GB")

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Block: {manifest.get('block_number')}")
        print(f"Entries: {manifest.get('entries', {}).get('total'):,}")
        print(f"Schema: v{manifest.get('schema_version')}")
    print()

    cmd = [
        "./target/release/bench_gpu_hints",
        "--db", db_path,
        "--lambda", str(lambda_param),
        "--iterations", str(iterations),
        "--warmup", str(warmup),
    ]

    if max_hints:
        cmd.extend(["--max-hints", str(max_hints)])

    print(f"=== Running Benchmark ===")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {
        "gpu": "1xH200",
        "db_size_gb": db_size / 1e9,
        "output": result.stdout,
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def main(
    gpus: int = 2,
    lambda_param: int = 128,
    iterations: int = 5,
    warmup: int = 2,
    max_hints: int = None,
):
    """
    Run GPU hint generation benchmark on Modal with mainnet v3 database.

    Examples:
        modal run scripts/modal_bench_mainnet.py --gpus 2
        modal run scripts/modal_bench_mainnet.py --gpus 1 --max-hints 1000000
    """
    print(f"Plinko Mainnet v3 GPU Benchmark")
    print(f"================================")
    print(f"GPUs: {gpus}x H200")
    print(f"Database: mainnet-v3 (~69 GB, 1.83B entries)")
    print(f"Lambda: {lambda_param}")
    print(f"Iterations: {iterations}")
    if max_hints:
        print(f"Max hints: {max_hints}")
    print()

    if gpus == 2:
        result = bench_mainnet_2xh200.remote(lambda_param, iterations, warmup, max_hints)
    else:
        result = bench_mainnet_1xh200.remote(lambda_param, iterations, warmup, max_hints)

    print("\n" + "=" * 60)
    print(f"GPU: {result['gpu']}")
    print(f"DB Size: {result['db_size_gb']:.2f} GB")
    print(f"Return code: {result['returncode']}")
    print()
    print(result["output"])

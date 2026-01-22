"""
Run GPU benchmark on Modal (assumes data already uploaded).

Usage:
    # First upload data (once)
    modal run scripts/modal_upload_data.py --local-path data/synthetic_0.1pct

    # Then run benchmarks (fast - just builds and runs)
    modal run scripts/modal_run_bench.py --gpu h100 --data synthetic_0.1pct
"""
import modal

app = modal.App("plinko-bench")
volume = modal.Volume.from_name("plinko-data", create_if_missing=True)

# Minimal image - just Rust + CUDA
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=["target", ".git", ".jj", "tmp", "__pycache__", ".DS_Store", "data", ".beads"]
    )
)


@app.function(image=image, gpu="H100", volumes={"/data": volume}, timeout=1800)
def bench_h100(data_name: str, lambda_param: int = 128, iterations: int = 10, chunk_size: int = None):
    """Run benchmark on H100."""
    return _run_bench("H100", data_name, lambda_param, iterations, chunk_size)


@app.function(image=image, gpu="H200", volumes={"/data": volume}, timeout=1800)
def bench_h200(data_name: str, lambda_param: int = 128, iterations: int = 10, chunk_size: int = None):
    """Run benchmark on H200."""
    return _run_bench("H200", data_name, lambda_param, iterations, chunk_size)


def _run_bench(gpu: str, data_name: str, lambda_param: int, iterations: int, chunk_size: int = None) -> dict:
    import subprocess
    import os

    os.chdir("/app")
    db_path = f"/data/{data_name}/database.bin"

    if not os.path.exists(db_path):
        print(f"ERROR: Data not found: {db_path}")
        print("\nAvailable datasets:")
        for d in os.listdir("/data"):
            print(f"  - {d}")
        raise FileNotFoundError(db_path)

    # Show data info
    size_mb = os.path.getsize(db_path) / 1e6
    print(f"Data: {db_path} ({size_mb:.1f} MB)")

    # Set up environment with CUDA paths
    env = os.environ.copy()
    env["PATH"] = f"/root/.cargo/bin:/usr/local/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_ROOT"] = "/usr/local/cuda"
    env["CUDA_PATH"] = "/usr/local/cuda"
    env["CUDA_ARCH"] = "sm_90"

    # Build
    print(f"\n=== Building ===")
    build_result = subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda"],
        capture_output=True,
        text=True,
        env=env,
    )
    if build_result.returncode != 0:
        # Print only the last part of stderr where error messages are
        stderr = build_result.stderr
        # Find actual errors (lines containing "error")
        error_lines = [l for l in stderr.split('\n') if 'error' in l.lower() or 'warning' in l.lower()]
        print("=== BUILD ERRORS/WARNINGS ===")
        print('\n'.join(error_lines[-50:]))
        print("=== LAST 3000 CHARS ===")
        print(stderr[-3000:])
        raise RuntimeError(f"Build failed with code {build_result.returncode}")
    print("Build succeeded")

    # Run
    print(f"\n=== Benchmarking on {gpu} ===")
    cmd = [
        "./target/release/bench_gpu_hints",
        "--db", db_path,
        "--lambda", str(lambda_param),
        "--iterations", str(iterations),
        "--warmup", "3",
        "--cpu-baseline",
    ]
    if chunk_size:
        cmd.extend(["--chunk-size", str(chunk_size)])
        print(f"Using mainnet chunk_size (w): {chunk_size}")

    # Run without capturing output so we can see progress in real-time
    import sys
    sys.stdout.flush()
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"Benchmark failed with code {result.returncode}")

    return {"gpu": gpu, "data": data_name, "output": f"Exit code: {result.returncode}"}


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    data: str = "synthetic_0.1pct",
    lambda_param: int = 128,
    iterations: int = 10,
    mainnet_w: bool = True,
):
    """
    Run benchmark on Modal.

    Args:
        gpu: h100 or h200
        data: Dataset name in volume (e.g., synthetic_0.1pct)
        lambda_param: Security parameter
        iterations: Benchmark iterations
        mainnet_w: Use mainnet chunk_size w=131072 (default: True)
    """
    chunk_size = 131072 if mainnet_w else None

    print(f"GPU: {gpu.upper()}")
    print(f"Data: {data}")
    print(f"Lambda: {lambda_param}")
    print(f"Chunk size (w): {chunk_size if chunk_size else 'auto'}")
    print(f"Iterations: {iterations}")
    print()

    if gpu.lower() == "h100":
        result = bench_h100.remote(data, lambda_param, iterations, chunk_size)
    elif gpu.lower() == "h200":
        result = bench_h200.remote(data, lambda_param, iterations, chunk_size)
    else:
        raise ValueError(f"Unknown GPU: {gpu}")

    print("\n" + "=" * 50)
    print(result["output"])

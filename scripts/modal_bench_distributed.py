"""
Distributed GPU Benchmark on Modal (50x H200).

Simulates full mainnet dataset (~83GB) in RAM to avoid disk I/O.
Splits hint generation across 50 workers (2% each).

Usage:
    modal run scripts/modal_bench_distributed.py
"""
import modal

app = modal.App("plinko-distributed-bench")

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

# Constants for Full Mainnet Scale
MAINNET_ENTRIES = 1_750_000_000  # ~1.75B entries
CHUNK_SIZE = 131072
LAMBDA = 128
TOTAL_HINTS = 2 * LAMBDA * CHUNK_SIZE
NUM_WORKERS = 50

# Volume to share compiled binary
bin_volume = modal.Volume.from_name("plinko-bin-vol", create_if_missing=True)

@app.function(
    image=cuda_image,
    gpu="H100", # Use H100 for fast build
    volumes={"/bin_vol": bin_volume},
    timeout=600,
)
def build_binary():
    import subprocess
    import os
    import shutil

    os.chdir("/app")
    print("Building binary...")
    subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
        check=True,
    )
    
    # Copy to volume
    print("Copying binary to volume...")
    if not os.path.exists("/bin_vol/bench_gpu_hints"):
        # Copy only if changed or missing? For now always copy to be safe
        pass
    
    shutil.copy2("target/release/bench_gpu_hints", "/bin_vol/bench_gpu_hints")
    # Ensure executable
    os.chmod("/bin_vol/bench_gpu_hints", 0o755)
    bin_volume.commit()
    print("Build complete.")

@app.function(
    image=cuda_image,
    gpu="H200",
    timeout=3600,
    memory=100000,
    volumes={"/bin_vol": bin_volume},
)
def bench_worker(worker_id: int):
    import subprocess
    import os
    import time

    # Wait for volume consistency (though build_binary runs before)
    bin_path = "/bin_vol/bench_gpu_hints"
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary not found at {bin_path}. Build failed?")

    # Calculate range
    hints_per_worker = TOTAL_HINTS // NUM_WORKERS
    start_hint = worker_id * hints_per_worker
    count = hints_per_worker
    
    if worker_id == NUM_WORKERS - 1:
        count = TOTAL_HINTS - start_hint

    print(f"[Worker {worker_id}] Processing hints {start_hint}..{start_hint + count} ({count} hints)")
    print(f"[Worker {worker_id}] Simulating {MAINNET_ENTRIES} entries (Synthetic RAM)")

    cmd = [
        bin_path,
        "--synthetic-entries", str(MAINNET_ENTRIES),
        "--chunk-size", str(CHUNK_SIZE),
        "--lambda", str(LAMBDA),
        "--hint-start", str(start_hint),
        "--hint-count", str(count),
        "--warmup", "1",
        "--iterations", "1",
    ]

    start_t = time.time()
    # Use check=True to raise exception on failure
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_t

    if result.returncode != 0:
        print(f"[Worker {worker_id}] FAILED:\n{result.stderr}")
        # Don't raise immediately to allow other workers to finish? 
        # But Modal map raises on exception anyway.
        raise Exception(f"Worker {worker_id} failed with exit code {result.returncode}")

    throughput = 0.0
    for line in result.stdout.splitlines():
        if "hints/sec" in line:
            parts = line.strip().split()
            throughput = float(parts[0])

    print(f"[Worker {worker_id}] Done in {elapsed:.2f}s. Throughput: {throughput:,.0f} hints/s")
    
    return {
        "worker": worker_id,
        "hints": count,
        "throughput": throughput,
        "elapsed": elapsed
    }

@app.local_entrypoint()
def main():
    print(f"Plinko Distributed Benchmark (50x H200)")
    print(f"=======================================")
    print(f"Total Hints: {TOTAL_HINTS:,}")
    print(f"Workers: {NUM_WORKERS}")
    
    print("Step 1: Building binary...")
    build_binary.remote()

    print("Step 2: Running workers...")
    results = list(bench_worker.map(range(NUM_WORKERS)))

    # Aggregation
    total_throughput = sum(r["throughput"] for r in results)
    max_elapsed = max(r["elapsed"] for r in results)
    total_hints = sum(r["hints"] for r in results)

    print("\n" + "=" * 50)
    print(f"AGGREGATE RESULTS")
    print(f"=================")
    print(f"Total Throughput: {total_throughput:,.0f} hints/s")
    print(f"Total Hints Gen:  {total_hints:,}")
    print(f"Wall Time (Max):  {max_elapsed:.2f} s")
    print(f"Effective Time:   {total_hints / total_throughput:.2f} s")
    print(f"Data Processed:   {MAINNET_ENTRIES * 40 / 1e9 * NUM_WORKERS:.2f} GB (Logical)")

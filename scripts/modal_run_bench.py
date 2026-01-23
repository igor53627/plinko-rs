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
mainnet_volume = modal.Volume.from_name("morphogenesis-data", create_if_missing=True)
hints_volume = modal.Volume.from_name("plinko-hints", create_if_missing=True)

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


@app.function(image=image, gpu="H100", volumes={"/data": volume}, timeout=7200)
def bench_h100(data_name: str, lambda_param: int = 128, iterations: int = 10, chunk_size: int = None, set_size: int = None, max_hints: int = None):
    """Run benchmark on H100."""
    return _run_bench("H100", data_name, lambda_param, iterations, chunk_size, set_size, max_hints)


@app.function(image=image, gpu="H200", volumes={"/data": volume}, timeout=7200)
def bench_h200(data_name: str, lambda_param: int = 128, iterations: int = 10, chunk_size: int = None, set_size: int = None, max_hints: int = None):
    """Run benchmark on H200."""
    return _run_bench("H200", data_name, lambda_param, iterations, chunk_size, set_size, max_hints)


@app.function(image=image, gpu="B200", volumes={"/data": volume}, timeout=7200)
def bench_b200(data_name: str, lambda_param: int = 128, iterations: int = 10, chunk_size: int = None, set_size: int = None, max_hints: int = None):
    """Run benchmark on B200."""
    return _run_bench("B200", data_name, lambda_param, iterations, chunk_size, set_size, max_hints)


@app.function(image=image, gpu="H200", volumes={"/mainnet": mainnet_volume}, timeout=7200, memory=65536)
def bench_mainnet_h200(lambda_param: int = 128, iterations: int = 5, chunk_size: int = 131072):
    """Run mainnet benchmark on H200."""
    return _run_mainnet_bench("H200", lambda_param, iterations, chunk_size)


@app.function(image=image, gpu="H200", volumes={"/mainnet": mainnet_volume}, timeout=7200)
def bench_mainnet_slice_h200(slice_pct: float = 1.0, lambda_param: int = 128, iterations: int = 2, chunk_size: int = 131072):
    """Run benchmark on a slice of mainnet data with full mainnet params."""
    import subprocess
    import os
    import mmap

    os.chdir("/app")
    src_path = "/mainnet/mainnet_optimized48.bin"
    slice_path = "/tmp/mainnet_slice.bin"

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Mainnet data not found: {src_path}")

    # Get file info
    file_size = os.path.getsize(src_path)
    entry_size = 48
    total_entries = file_size // entry_size
    slice_entries = int(total_entries * slice_pct / 100)
    slice_size = slice_entries * entry_size

    print(f"=== Creating {slice_pct}% slice of mainnet ===")
    print(f"Source: {src_path} ({file_size / 1e9:.1f} GB, {total_entries:,} entries)")
    print(f"Slice: {slice_size / 1e9:.2f} GB, {slice_entries:,} entries")

    # Create slice file
    with open(src_path, "rb") as src:
        with open(slice_path, "wb") as dst:
            # Read and write in chunks
            remaining = slice_size
            chunk = 100 * 1024 * 1024  # 100 MB chunks
            while remaining > 0:
                to_read = min(chunk, remaining)
                data = src.read(to_read)
                dst.write(data)
                remaining -= len(data)

    print(f"Slice created: {slice_path}")

    # Calculate mainnet set_size
    mainnet_entries = total_entries
    mainnet_set_size = (mainnet_entries + chunk_size - 1) // chunk_size
    mainnet_set_size = ((mainnet_set_size + 3) // 4) * 4  # Round to multiple of 4

    print(f"\nUsing mainnet params:")
    print(f"  Mainnet entries: {mainnet_entries:,}")
    print(f"  Mainnet set_size (c): {mainnet_set_size}")
    print(f"  Blocks per hint: ~{mainnet_set_size // 2}")

    # Build
    env = os.environ.copy()
    env["PATH"] = f"/root/.cargo/bin:/usr/local/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_ROOT"] = "/usr/local/cuda"
    env["CUDA_PATH"] = "/usr/local/cuda"
    env["CUDA_ARCH"] = "sm_90"

    print(f"\n=== Building ===")
    build_result = subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
        capture_output=True,
        text=True,
        env=env,
    )
    if build_result.returncode != 0:
        print(build_result.stderr[-3000:])
        raise RuntimeError(f"Build failed")
    print("Build succeeded")

    # Run benchmark
    print(f"\n=== Benchmarking {slice_pct}% mainnet slice on H200 ===")
    max_hints = 100000  # Limit for reasonable runtime
    cmd = [
        "./target/release/bench_gpu_hints",
        "--db", slice_path,
        "--lambda", str(lambda_param),
        "--iterations", str(iterations),
        "--warmup", "1",
        "--chunk-size", str(chunk_size),
        "--set-size", str(mainnet_set_size),
        "--max-hints", str(max_hints),
    ]

    import sys
    sys.stdout.flush()
    result = subprocess.run(cmd, env=env)

    # Cleanup
    os.remove(slice_path)

    return {"gpu": "H200", "data": f"mainnet_{slice_pct}pct", "output": f"Exit code: {result.returncode}"}


def _run_mainnet_bench(gpu: str, lambda_param: int, iterations: int, chunk_size: int) -> dict:
    import subprocess
    import os

    os.chdir("/app")
    db_path = "/mainnet/mainnet_optimized48.bin"

    if not os.path.exists(db_path):
        print(f"ERROR: Mainnet data not found: {db_path}")
        print("\nAvailable files in /mainnet:")
        for f in os.listdir("/mainnet"):
            print(f"  - {f}")
        raise FileNotFoundError(db_path)

    # Show data info
    size_gb = os.path.getsize(db_path) / 1e9
    print(f"Data: {db_path} ({size_gb:.1f} GB)")

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
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
        capture_output=True,
        text=True,
        env=env,
    )
    if build_result.returncode != 0:
        stderr = build_result.stderr
        error_lines = [l for l in stderr.split('\n') if 'error' in l.lower() or 'warning' in l.lower()]
        print("=== BUILD ERRORS/WARNINGS ===")
        print('\n'.join(error_lines[-50:]))
        print("=== LAST 3000 CHARS ===")
        print(stderr[-3000:])
        raise RuntimeError(f"Build failed with code {build_result.returncode}")
    print("Build succeeded")

    # Run
    print(f"\n=== Benchmarking MAINNET on {gpu} ===")
    # Limit hints to fit in GPU memory (H200: 141GB, DB: ~103GB, leaves ~38GB)
    # Each hint subset needs ~2KB, so max ~10M hints to be safe
    max_hints = 10_000_000
    cmd = [
        "./target/release/bench_gpu_hints",
        "--db", db_path,
        "--lambda", str(lambda_param),
        "--iterations", str(iterations),
        "--warmup", "1",
        "--chunk-size", str(chunk_size),
        "--max-hints", str(max_hints),
    ]
    print(f"Using mainnet chunk_size (w): {chunk_size}")
    print(f"Limiting to {max_hints:,} hints (extrapolating to full 33.5M)")

    import sys
    sys.stdout.flush()
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"Benchmark failed with code {result.returncode}")

    return {"gpu": gpu, "data": "mainnet", "output": f"Exit code: {result.returncode}"}


def _run_bench(gpu: str, data_name: str, lambda_param: int, iterations: int, chunk_size: int = None, set_size: int = None, max_hints: int = None) -> dict:
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
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
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
    ]
    if chunk_size:
        cmd.extend(["--chunk-size", str(chunk_size)])
        print(f"Using mainnet chunk_size (w): {chunk_size}")
    if set_size:
        cmd.extend(["--set-size", str(set_size)])
        print(f"Using mainnet set_size (c): {set_size}")
    if max_hints:
        cmd.extend(["--max-hints", str(max_hints)])
        print(f"Limiting to {max_hints:,} hints")

    # Run without capturing output so we can see progress in real-time
    import sys
    sys.stdout.flush()
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"Benchmark failed with code {result.returncode}")

    return {"gpu": gpu, "data": data_name, "output": f"Exit code: {result.returncode}"}


def _multi_gpu_worker_impl(worker_id: int, num_workers: int, total_hints: int, chunk_size: int, slice_pct: float, gpu_name: str, replicate_data: bool = False):
    """Implementation for multi-GPU worker (shared by H200 and B200).

    If replicate_data=True: Each worker gets the SAME data slice (for accurate hint benchmarking)
    If replicate_data=False: Data is SPLIT among workers (for I/O testing)
    """
    import subprocess
    import os
    import re

    os.chdir("/app")
    src_path = "/mainnet/mainnet_optimized48.bin"
    slice_path = f"/tmp/mainnet_slice_{worker_id}.bin"

    file_size = os.path.getsize(src_path)
    entry_size = 48
    total_entries = file_size // entry_size

    if replicate_data:
        # Each worker gets the SAME slice_pct of data (replicated)
        slice_entries = int(total_entries * slice_pct / 100)
        slice_size = slice_entries * entry_size
        worker_offset = 0  # All workers start from beginning

        print(f"Worker {worker_id} ({gpu_name}): Copying {slice_pct}% data ({slice_size / 1e9:.2f} GB) [REPLICATED]...")
    else:
        # Data is SPLIT among workers
        worker_slice_pct = slice_pct / num_workers
        slice_entries = int(total_entries * worker_slice_pct / 100)
        slice_size = slice_entries * entry_size
        worker_offset = worker_id * slice_size

        print(f"Worker {worker_id} ({gpu_name}): Creating {worker_slice_pct:.2f}% slice ({slice_size / 1e9:.2f} GB) at offset {worker_offset / 1e9:.2f} GB...")

    with open(src_path, "rb") as src:
        src.seek(worker_offset)
        with open(slice_path, "wb") as dst:
            remaining = slice_size
            chunk = 100 * 1024 * 1024
            while remaining > 0:
                to_read = min(chunk, remaining)
                data = src.read(to_read)
                if not data:
                    break
                dst.write(data)
                remaining -= len(data)

    # Calculate hints per worker
    hints_per_worker = total_hints // num_workers
    worker_hints = hints_per_worker

    # Calculate set_size based on the data this worker has
    worker_set_size = (slice_entries + chunk_size - 1) // chunk_size
    worker_set_size = ((worker_set_size + 3) // 4) * 4  # Round to multiple of 4

    print(f"Worker {worker_id}/{num_workers}: {worker_hints:,} hints, {slice_entries:,} entries, set_size={worker_set_size}")

    # Build
    env = os.environ.copy()
    env["PATH"] = f"/root/.cargo/bin:/usr/local/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_ROOT"] = "/usr/local/cuda"
    env["CUDA_PATH"] = "/usr/local/cuda"
    env["CUDA_ARCH"] = "sm_90"

    build_result = subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
        capture_output=True, text=True, env=env,
    )
    if build_result.returncode != 0:
        raise RuntimeError(f"Build failed: {build_result.stderr[-1000:]}")

    # Run benchmark (iterations=1, warmup=1 to get GPU loaded, then measure)
    cmd = [
        "./target/release/bench_gpu_hints",
        "--db", slice_path,
        "--lambda", "128",
        "--iterations", "1",
        "--warmup", "1",
        "--chunk-size", str(chunk_size),
        "--set-size", str(worker_set_size),
        "--max-hints", str(worker_hints),
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    output = result.stdout if result.stdout else ""

    # Parse actual GPU time from output (e.g., "Iteration 1: 28703.982 ms")
    gpu_time_ms = None
    for line in output.split('\n'):
        if 'Iteration 1:' in line:
            match = re.search(r'Iteration 1:\s+([\d.]+)\s*ms', line)
            if match:
                gpu_time_ms = float(match.group(1))
                break

    os.remove(slice_path)

    return {
        "worker_id": worker_id,
        "hints": worker_hints,
        "gpu_time_ms": gpu_time_ms,
        "output": output[-1500:]
    }


@app.function(image=image, gpu="H200", volumes={"/mainnet": mainnet_volume}, timeout=7200)
def bench_multi_gpu_worker_h200(worker_id: int, num_workers: int, total_hints: int, chunk_size: int = 131072, slice_pct: float = 10.0, replicate_data: bool = False):
    """Single H200 worker for multi-GPU benchmark."""
    return _multi_gpu_worker_impl(worker_id, num_workers, total_hints, chunk_size, slice_pct, "H200", replicate_data)


@app.function(image=image, gpu="B200", volumes={"/mainnet": mainnet_volume}, timeout=7200)
def bench_multi_gpu_worker_b200(worker_id: int, num_workers: int, total_hints: int, chunk_size: int = 131072, slice_pct: float = 10.0, replicate_data: bool = False):
    """Single B200 worker for multi-GPU benchmark."""
    return _multi_gpu_worker_impl(worker_id, num_workers, total_hints, chunk_size, slice_pct, "B200", replicate_data)


# =============================================================================
# Production Hint Generation
# =============================================================================

@app.function(image=image, gpu="H200", volumes={"/mainnet": mainnet_volume, "/hints": hints_volume}, timeout=14400)
def generate_hints_worker(run_id: str, worker_id: int, num_workers: int, chunk_size: int = 131072):
    """Production hint generation worker. Saves hints to volume."""
    import subprocess
    import os
    import re

    os.chdir("/app")
    db_path = "/mainnet/mainnet_optimized48.bin"
    output_path = f"/hints/{run_id}/hints_{worker_id:03d}.bin"

    # Create output directory
    os.makedirs(f"/hints/{run_id}", exist_ok=True)

    # Calculate hint range for this worker
    full_hints = 33554432  # 2 * 128 * 131072
    hints_per_worker = full_hints // num_workers
    hint_start = worker_id * hints_per_worker
    hint_count = hints_per_worker if worker_id < num_workers - 1 else (full_hints - hint_start)

    print(f"=== PRODUCTION HINT GENERATION ===")
    print(f"Run ID: {run_id}")
    print(f"Worker: {worker_id}/{num_workers}")
    print(f"Hint range: {hint_start:,} to {hint_start + hint_count:,} ({hint_count:,} hints)")
    print(f"Output: {output_path}")
    print()

    # Build
    env = os.environ.copy()
    env["PATH"] = f"/root/.cargo/bin:/usr/local/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_ROOT"] = "/usr/local/cuda"
    env["CUDA_PATH"] = "/usr/local/cuda"
    env["CUDA_ARCH"] = "sm_90"

    print("Building...")
    build_result = subprocess.run(
        ["cargo", "build", "--release", "-p", "plinko", "--bin", "bench_gpu_hints", "--features", "cuda,parallel"],
        capture_output=True, text=True, env=env,
    )
    if build_result.returncode != 0:
        print(f"Build failed: {build_result.stderr[-2000:]}")
        raise RuntimeError("Build failed")
    print("Build succeeded")
    print()

    # Run hint generation
    cmd = [
        "./target/release/bench_gpu_hints",
        "--db", db_path,
        "--lambda", "128",
        "--chunk-size", str(chunk_size),
        "--hint-start", str(hint_start),
        "--hint-count", str(hint_count),
        "--output", output_path,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if result.returncode != 0:
        print(f"ERROR: Hint generation failed with code {result.returncode}")
        print(output[-2000:])
        return {
            "run_id": run_id,
            "worker_id": worker_id,
            "error": f"Generation failed with code {result.returncode}",
            "output": output[-1500:]
        }

    # Parse results
    hints_generated = 0
    generation_time_ms = 0
    for line in output.split('\n'):
        if 'HINTS_GENERATED=' in line:
            hints_generated = int(line.split('=')[1])
        if 'GENERATION_TIME_MS=' in line:
            generation_time_ms = float(line.split('=')[1])

    # Verify output file
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        expected_size = hint_count * 32
        print(f"\nOutput file: {output_path}")
        print(f"File size: {file_size:,} bytes ({file_size / 1e6:.2f} MB)")
        print(f"Expected: {expected_size:,} bytes")

        if file_size != expected_size:
            print(f"ERROR: File size mismatch! Got {file_size}, expected {expected_size}")
            return {
                "run_id": run_id,
                "worker_id": worker_id,
                "error": f"Size mismatch: {file_size} != {expected_size}",
                "output": output[-1500:]
            }

        # Commit the volume to persist the file
        hints_volume.commit()
    else:
        print(f"ERROR: Output file not created!")
        print(output[-2000:])
        return {
            "run_id": run_id,
            "worker_id": worker_id,
            "error": "Output file not generated",
            "output": output[-1500:]
        }

    return {
        "run_id": run_id,
        "worker_id": worker_id,
        "hint_start": hint_start,
        "hint_count": hint_count,
        "hints_generated": hints_generated,
        "generation_time_ms": generation_time_ms,
        "output_path": output_path,
        "output": output[-1500:]
    }


@app.function(image=image, volumes={"/hints": hints_volume}, timeout=3600)
def combine_hints(run_id: str, num_workers: int):
    """Combine hint files from all workers into a single file."""
    import os

    output_dir = f"/hints/{run_id}"
    combined_path = f"/hints/{run_id}/hints_combined.bin"

    print(f"=== COMBINING HINTS ===")
    print(f"Run ID: {run_id}")
    print(f"Workers: {num_workers}")
    print()

    # Check all files exist
    missing = []
    for i in range(num_workers):
        path = f"{output_dir}/hints_{i:03d}.bin"
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print(f"ERROR: Missing {len(missing)} files:")
        for p in missing[:10]:
            print(f"  - {p}")
        return {"error": f"Missing {len(missing)} files"}

    # Combine files in order
    print(f"Combining {num_workers} files...")
    total_bytes = 0
    with open(combined_path, 'wb') as outfile:
        for i in range(num_workers):
            path = f"{output_dir}/hints_{i:03d}.bin"
            with open(path, 'rb') as infile:
                data = infile.read()
                outfile.write(data)
                total_bytes += len(data)
            if i % 10 == 0:
                print(f"  Processed {i + 1}/{num_workers} files...")

    # Verify
    total_hints = total_bytes // 32
    print()
    print(f"Combined file: {combined_path}")
    print(f"Total size: {total_bytes:,} bytes ({total_bytes / 1e9:.2f} GB)")
    print(f"Total hints: {total_hints:,}")

    hints_volume.commit()

    return {
        "run_id": run_id,
        "combined_path": combined_path,
        "total_bytes": total_bytes,
        "total_hints": total_hints
    }


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    data: str = "synthetic_0.1pct",
    lambda_param: int = 128,
    iterations: int = 10,
    mainnet_w: bool = True,
    mainnet: bool = False,
    simulate_mainnet: bool = False,
    multi_gpu: int = 0,
    replicate: bool = False,
    data_pct: float = 100.0,
    hint_pct: float = 1.0,
    # Production hint generation
    generate: bool = False,
    run_id: str = "",
    num_gpus: int = 50,
    combine_only: bool = False,
):
    """
    Run benchmark on Modal.

    Args:
        gpu: h100, h200, or b200
        data: Dataset name in volume (e.g., synthetic_0.1pct)
        lambda_param: Security parameter
        iterations: Benchmark iterations
        mainnet_w: Use mainnet chunk_size w=131072 (default: True)
        mainnet: Run full mainnet benchmark (uses morphogenesis-data volume)
        simulate_mainnet: Use mainnet set_size (16404) on smaller DB to simulate workload
        multi_gpu: Number of B200 GPUs to use in parallel (0 = disabled)
        generate: Production mode - generate and save all hints
        run_id: Unique ID for this hint generation run (auto-generated if empty)
        num_gpus: Number of H200 GPUs for production hint generation
        combine_only: Only run the combine step (for re-running after failure)
    """
    import time
    from datetime import datetime

    chunk_size = 131072 if mainnet_w else None
    set_size = 16404 if simulate_mainnet else None
    max_hints = 100000 if simulate_mainnet else None  # Limit hints for simulation

    # Production hint generation mode
    if generate or combine_only:
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        full_hints = 33554432
        hints_per_gpu = full_hints // num_gpus

        print(f"{'='*60}")
        print(f"PRODUCTION HINT GENERATION")
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"GPUs: {num_gpus}× H200")
        print(f"Total hints: {full_hints:,}")
        print(f"Hints per GPU: {hints_per_gpu:,}")
        print(f"Output: plinko-hints volume, /hints/{run_id}/")
        print(f"{'='*60}")
        print()

        if not combine_only:
            # Launch all workers
            print(f"Launching {num_gpus} workers...")
            start_time = time.time()
            futures = [
                generate_hints_worker.spawn(run_id, i, num_gpus, 131072)
                for i in range(num_gpus)
            ]

            # Wait for all to complete
            print(f"Waiting for workers to complete...")
            results = []
            for i, f in enumerate(futures):
                result = f.get()
                results.append(result)
                print(f"  Worker {result['worker_id']}: {result['hints_generated']:,} hints in {result['generation_time_ms']/1000:.1f}s")

            total_time = time.time() - start_time
            total_hints_generated = sum(r['hints_generated'] for r in results)
            max_gen_time = max(r['generation_time_ms'] for r in results)

            print()
            print(f"All workers completed in {total_time:.1f}s (wall clock)")
            print(f"Max GPU time: {max_gen_time/1000:.1f}s")
            print(f"Total hints: {total_hints_generated:,}")
            print()

        # Combine files
        print(f"Combining hint files...")
        combine_result = combine_hints.remote(run_id, num_gpus)
        
        if "error" in combine_result:
            print(f"ERROR: Combine failed: {combine_result['error']}")
            exit(1)
            
        print(f"Combined: {combine_result['total_hints']:,} hints ({combine_result['total_bytes']/1e9:.2f} GB)")
        print()
        print(f"{'='*60}")
        print(f"COMPLETE")
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"Output: modal volume get plinko-hints /hints/{run_id}/hints_combined.bin")
        return

    # Multi-GPU mode
    if multi_gpu > 0:
        gpu_type = gpu.upper()

        # GPU pricing (Modal)
        gpu_prices = {"H200": 4.89, "B200": 6.25, "H100": 3.95}
        gpu_price = gpu_prices.get(gpu_type, 4.89)

        # Calculate hints: hint_pct of full 33.5M hints
        full_hints = 33554432  # 2 * 128 * 131072
        total_hints = int(full_hints * hint_pct / 100)
        hints_per_gpu = total_hints // multi_gpu

        # Data size
        full_data_gb = 103.0  # Full mainnet ~103 GB
        data_gb = full_data_gb * data_pct / 100

        print(f"=== MULTI-GPU BENCHMARK ({multi_gpu}× {gpu_type}) ===")
        if replicate:
            print(f"Data: {data_pct}% of mainnet ({data_gb:.1f} GB) REPLICATED to each worker")
            print(f"Set size: {int(2150000000 * data_pct / 100 / 131072)} (production: 16404)")
        else:
            print(f"Data: {data_pct}% of mainnet SPLIT across {multi_gpu} workers")
            print(f"Data per worker: {data_pct/multi_gpu:.2f}% (~{data_gb/multi_gpu:.2f} GB)")
        print(f"Hints: {hint_pct}% of full = {total_hints:,} total, {hints_per_gpu:,} per GPU")
        print(f"Params: λ=128, w=131072, 759 rounds, ChaCha8")
        print()

        # Launch all workers in parallel
        start_time = time.time()
        if gpu_type == "H200":
            futures = [
                bench_multi_gpu_worker_h200.spawn(i, multi_gpu, total_hints, chunk_size or 131072, data_pct, replicate)
                for i in range(multi_gpu)
            ]
        elif gpu_type == "B200":
            futures = [
                bench_multi_gpu_worker_b200.spawn(i, multi_gpu, total_hints, chunk_size or 131072, data_pct, replicate)
                for i in range(multi_gpu)
            ]
        else:
            raise ValueError(f"Multi-GPU only supports H200 or B200, got: {gpu_type}")

        # Wait for all to complete
        results = [f.get() for f in futures]
        total_wall_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"RESULTS ({multi_gpu}× {gpu_type}) - GPU COMPUTE TIME ONLY")
        print(f"{'='*60}")

        valid_results = [r for r in results if r['gpu_time_ms'] is not None]
        for r in results:
            if r['gpu_time_ms']:
                print(f"Worker {r['worker_id']}: {r['hints']:,} hints in {r['gpu_time_ms']/1000:.2f}s (GPU only)")
            else:
                print(f"Worker {r['worker_id']}: {r['hints']:,} hints - ERROR parsing time")
                print(f"  Output: {r['output'][-500:]}")

        if valid_results:
            total_hints_processed = sum(r['hints'] for r in valid_results)
            max_gpu_time_ms = max(r['gpu_time_ms'] for r in valid_results)
            max_gpu_time_sec = max_gpu_time_ms / 1000
            avg_gpu_time_ms = sum(r['gpu_time_ms'] for r in valid_results) / len(valid_results)

            # Calculate set_size based on data
            mainnet_entries = 2150000000
            worker_entries = int(mainnet_entries * data_pct / 100) if replicate else int(mainnet_entries * data_pct / 100 / multi_gpu)
            worker_set_size = (worker_entries + 131072 - 1) // 131072
            worker_set_size = ((worker_set_size + 3) // 4) * 4

            print(f"\n{'='*60}")
            print(f"BENCHMARK SUMMARY")
            print(f"{'='*60}")
            print(f"| Parameter            | Value                           |")
            print(f"|----------------------|---------------------------------|")
            print(f"| GPU Type             | {gpu_type}                      |")
            print(f"| Number of GPUs       | {multi_gpu}                     |")
            if replicate:
                print(f"| Data (replicated)    | {data_pct}% of mainnet ({data_gb:.1f} GB each) |")
            else:
                print(f"| Data (split)         | {data_pct}% total, {data_pct/multi_gpu:.2f}% each |")
            print(f"| Total Hints          | {total_hints_processed:,} ({hint_pct}% of full) |")
            print(f"| Hints per GPU        | {hints_per_gpu:,}               |")
            print(f"| λ (security param)   | 128                             |")
            print(f"| w (chunk size)       | 131,072                         |")
            print(f"| c (set size/worker)  | {worker_set_size} (prod: 16404) |")
            print(f"| SwapOrNot rounds     | 759                             |")
            print(f"| Cipher               | ChaCha8                         |")
            print()
            print(f"| Metric               | Value                           |")
            print(f"|----------------------|---------------------------------|")
            print(f"| Max GPU time         | {max_gpu_time_sec:.2f}s         |")
            print(f"| Avg GPU time         | {avg_gpu_time_ms/1000:.2f}s     |")
            print(f"| Throughput           | {total_hints_processed / max_gpu_time_sec:,.0f} hints/sec |")

            # Extrapolate to full 33.5M hints
            extrapolated_time_sec = max_gpu_time_sec * (full_hints / total_hints_processed)

            # Also extrapolate set_size scaling if not using full data
            if data_pct < 100:
                set_size_scale = 16404 / worker_set_size
                extrapolated_time_sec *= set_size_scale
                print(f"| Set size scaling     | {set_size_scale:.1f}x (to prod 16404) |")

            print()
            print(f"| Extrapolation (100% hints + 100% data) |")
            print(f"|----------------------|---------------------------------|")
            print(f"| Full hints           | {full_hints:,}                  |")
            print(f"| Est. parallel time   | {extrapolated_time_sec:.0f}s ({extrapolated_time_sec/60:.1f} min) |")

            # Cost calculation
            gpu_hours = multi_gpu * (extrapolated_time_sec / 3600)
            cost = gpu_hours * gpu_price
            print()
            print(f"| Cost Estimate        | Value                           |")
            print(f"|----------------------|---------------------------------|")
            print(f"| GPU hours            | {gpu_hours:.2f} hr              |")
            print(f"| Price per GPU-hr     | ${gpu_price:.2f}                |")
            print(f"| Total cost           | ${cost:.2f}                     |")
            print(f"{'='*60}")
        return

    if mainnet:
        if simulate_mainnet:
            # Run on 1% slice of actual mainnet data with full mainnet params
            print(f"=== MAINNET 1% SLICE BENCHMARK ===")
            print(f"GPU: H200")
            print(f"Data: 1% of mainnet_optimized48.bin")
            print(f"Lambda: {lambda_param}")
            print(f"Chunk size (w): {chunk_size}")
            print(f"Iterations: {iterations}")
            print()
            result = bench_mainnet_slice_h200.remote(1.0, lambda_param, iterations, chunk_size)
        else:
            print(f"=== MAINNET BENCHMARK ===")
            print(f"GPU: H200")
            print(f"Data: mainnet_optimized48.bin (~88 GB)")
            print(f"Entries: 1,831,921,514")
            print(f"Lambda: {lambda_param}")
            print(f"Chunk size (w): {chunk_size}")
            print(f"Iterations: {iterations}")
            print()
            result = bench_mainnet_h200.remote(lambda_param, iterations, chunk_size)
    else:
        print(f"GPU: {gpu.upper()}")
        print(f"Data: {data}")
        print(f"Lambda: {lambda_param}")
        print(f"Chunk size (w): {chunk_size if chunk_size else 'auto'}")
        print(f"Set size (c): {set_size if set_size else 'auto'}")
        print(f"Simulate mainnet: {simulate_mainnet}")
        print(f"Iterations: {iterations}")
        print()

        if gpu.lower() == "h100":
            result = bench_h100.remote(data, lambda_param, iterations, chunk_size, set_size, max_hints)
        elif gpu.lower() == "h200":
            result = bench_h200.remote(data, lambda_param, iterations, chunk_size, set_size, max_hints)
        elif gpu.lower() == "b200":
            result = bench_b200.remote(data, lambda_param, iterations, chunk_size, set_size, max_hints)
        else:
            raise ValueError(f"Unknown GPU: {gpu}. Supported: h100, h200, b200")

    print("\n" + "=" * 50)
    print(result["output"])

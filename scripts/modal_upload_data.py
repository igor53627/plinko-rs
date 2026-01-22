"""
Upload local synthetic data to Modal volume.

Usage:
    modal run scripts/modal_upload_data.py --local-path data/synthetic_0.1pct
"""
import modal
import os

app = modal.App("plinko-upload")
volume = modal.Volume.from_name("plinko-data", create_if_missing=True)


@app.function(volumes={"/data": volume})
def upload_data(files: dict[str, bytes], remote_dir: str):
    """Upload files to volume."""
    import os

    os.makedirs(f"/data/{remote_dir}", exist_ok=True)

    for name, content in files.items():
        path = f"/data/{remote_dir}/{name}"
        with open(path, "wb") as f:
            f.write(content)
        print(f"  Uploaded: {path} ({len(content) / 1e6:.2f} MB)")

    volume.commit()
    print(f"\nDone! Data available at /data/{remote_dir}")


@app.function(volumes={"/data": volume})
def list_data():
    """List data in volume."""
    import os

    print("Modal volume contents:")
    if not os.path.exists("/data"):
        print("  (empty)")
        return

    for name in sorted(os.listdir("/data")):
        path = f"/data/{name}"
        if os.path.isdir(path):
            files = os.listdir(path)
            total = sum(os.path.getsize(f"/data/{name}/{f}") for f in files)
            print(f"  {name}/: {total / 1e6:.2f} MB ({len(files)} files)")


@app.local_entrypoint()
def main(local_path: str = None, list_only: bool = False):
    """
    Upload local data to Modal volume.

    Examples:
        modal run scripts/modal_upload_data.py --local-path data/synthetic_0.1pct
        modal run scripts/modal_upload_data.py --list-only
    """
    if list_only or local_path is None:
        list_data.remote()
        return

    if not os.path.isdir(local_path):
        raise ValueError(f"Not a directory: {local_path}")

    # Read all files
    files = {}
    for name in os.listdir(local_path):
        filepath = os.path.join(local_path, name)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                files[name] = f.read()
            print(f"Reading: {name} ({len(files[name]) / 1e6:.2f} MB)")

    # Get remote dir name from local path
    remote_dir = os.path.basename(local_path.rstrip("/"))

    print(f"\nUploading to Modal volume: /data/{remote_dir}")
    upload_data.remote(files, remote_dir)

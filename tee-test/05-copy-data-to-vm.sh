#!/bin/bash
# Copy database and binary into the running VM
# Run this AFTER the VM is booted and SSH is accessible

set -e
SSH_PORT=2222
VM_HOST="localhost"

# Default paths - override with environment variables if needed
DB_PATH="${PLINKO_DB_PATH:-/mnt/mainnet/plinko/database.bin}"
HINTS_BIN="${PLINKO_HINTS_BIN:-../target/release/plinko_hints}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Copying data to SEV-SNP VM ==="
echo

echo "1. Waiting for VM SSH to be ready..."
for i in {1..30}; do
    if ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no -p $SSH_PORT ubuntu@$VM_HOST "echo ok" 2>/dev/null; then
        echo "   SSH ready!"
        break
    fi
    echo "   Waiting... ($i/30)"
    sleep 5
done

echo
echo "2. Creating /mnt/plinko directory in VM..."
ssh -p $SSH_PORT ubuntu@$VM_HOST "sudo mkdir -p /mnt/plinko && sudo chown ubuntu:ubuntu /mnt/plinko"

echo
echo "3. Copying plinko_hints binary..."
if [[ -f "$HINTS_BIN" ]]; then
    scp -P $SSH_PORT "$HINTS_BIN" ubuntu@$VM_HOST:/mnt/plinko/
else
    echo "   Warning: $HINTS_BIN not found, skipping"
fi

echo
echo "4. Copying run-benchmark.sh..."
scp -P $SSH_PORT "$SCRIPT_DIR/run-benchmark.sh" ubuntu@$VM_HOST:/mnt/plinko/

echo
echo "5. Copying database.bin (73GB - this will take a while)..."
echo "   Source: $DB_PATH"
echo "   Dest: ubuntu@$VM_HOST:/mnt/plinko/database.bin"
if [[ -f "$DB_PATH" ]]; then
    time rsync -avP --progress -e "ssh -p $SSH_PORT" "$DB_PATH" ubuntu@$VM_HOST:/mnt/plinko/
else
    echo "   Warning: $DB_PATH not found"
    echo "   Set PLINKO_DB_PATH environment variable to the database location"
fi

echo
echo "=== Copy Complete ==="
echo
echo "Now SSH into the VM and run the benchmark:"
echo "  ssh -p $SSH_PORT ubuntu@$VM_HOST"
echo "  cd /mnt/plinko && ./run-benchmark.sh"

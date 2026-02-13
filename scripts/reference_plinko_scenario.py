#!/usr/bin/env python3
"""Run a deterministic Plinko e2e scenario against an external reference repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def deterministic_entry(index: int, entry_size: int, salt: str) -> bytes:
    material = f"{salt}:{index}".encode("ascii")
    return hashlib.sha256(material).digest()[:entry_size]


def hex_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_query_round(client, server, expected_db, indices):
    queries = client.query(indices)
    responses = server.answer(queries)
    results = client.extract(responses)
    client.replenish_hints()

    expected = [expected_db[idx] for idx in indices]
    if results != expected:
        for idx, got, want in zip(indices, results, expected):
            if got != want:
                raise AssertionError(
                    f"mismatch at index={idx}: got={got.hex()} want={want.hex()}"
                )

    return {
        "indices": indices,
        "result_sha256": [hex_digest(item) for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Path to rms24-plinko-spec checkout")
    parser.add_argument("--output", help="Optional path to write JSON summary")
    args = parser.parse_args()

    repo_dir = Path(args.repo).resolve()
    pir_dir = repo_dir / "pir"
    if not (pir_dir / "plinko").exists():
        raise FileNotFoundError(
            f"expected '{repo_dir}/pir/plinko' to exist (invalid reference repo path)"
        )

    # Import the Plinko package directly from pir/ to avoid package-level side effects.
    sys.path.insert(0, str(pir_dir))
    from plinko import Client, Params, Server  # pylint: disable=import-error

    params = Params(
        num_entries=64,
        entry_size=16,
        security_param=40,
        num_backup_hints=32,
    )
    expected_db = [
        deterministic_entry(i, params.entry_size, "db-v1")
        for i in range(params.num_entries)
    ]

    server = Server(expected_db.copy(), params)
    client = Client(params)
    client.generate_hints(server.stream_database())

    round_one = run_query_round(
        client=client,
        server=server,
        expected_db=expected_db,
        indices=[0, 5, 17, 33, 63],
    )

    update_map = {
        5: deterministic_entry(1005, params.entry_size, "update-v1"),
        42: deterministic_entry(1042, params.entry_size, "update-v1"),
    }
    updates = server.update_entries(update_map)
    client.update_hints(updates)
    for idx, new_value in update_map.items():
        expected_db[idx] = new_value

    round_two = run_query_round(
        client=client,
        server=server,
        expected_db=expected_db,
        indices=[5, 42, 1, 33],
    )

    # Re-query a cached item to exercise the decoy path.
    round_three = run_query_round(
        client=client,
        server=server,
        expected_db=expected_db,
        indices=[5, 17],
    )

    summary = {
        "params": {
            "num_entries": params.num_entries,
            "entry_size": params.entry_size,
            "block_size": params.block_size,
            "num_blocks": params.num_blocks,
            "num_reg_hints": params.num_reg_hints,
            "num_backup_hints": params.num_backup_hints,
        },
        "rounds": [round_one, round_two, round_three],
        "updates": sorted(update_map.keys()),
        "remaining_queries": client.remaining_queries(),
    }

    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

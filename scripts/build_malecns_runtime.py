"""Build deterministic MaleCNS runtime CSR assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.malecns_assets import build_runtime_assets, load_runtime_assets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "malecns" / "v1.0",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "malecns" / "runtime",
    )
    parser.add_argument("--minimum-weight", type=int, default=5)
    args = parser.parse_args()
    started = time.perf_counter()
    output = build_runtime_assets(
        args.source_dir,
        args.output_root,
        minimum_weight=args.minimum_weight,
    )
    assets = load_runtime_assets(output)
    result = {
        "output": str(output),
        "model_id": assets.model_id,
        "manifest_sha256": assets.manifest_sha256,
        "nodes": assets.node_count,
        "edges": assets.edge_count,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

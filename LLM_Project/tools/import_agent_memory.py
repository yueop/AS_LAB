from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import existing metrics_history.jsonl files into persistent organ-agent memory."
    )
    parser.add_argument(
        "--metrics-glob",
        default="outputs/**/metrics_history.jsonl",
        help="Glob for source metrics history files.",
    )
    parser.add_argument(
        "--agent-memory-dir",
        type=Path,
        default=Path("model_comparison") / "agent_memory",
        help="Destination directory for {target_organ}_metrics_history.jsonl.",
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("configs") / "model_registry.json",
        help="Current model registry used to filter obsolete or wrong-organ records.",
    )
    parser.add_argument(
        "--include-incompatible",
        action="store_true",
        help="Do not filter records by current registry model/target support.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be imported without writing memory files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = _load_registry(args.model_registry)
    source_paths = sorted(
        Path(path)
        for path in glob.glob(args.metrics_glob, recursive=True)
        if Path(path).is_file()
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[Any, ...]] = set()
    skipped = {
        "empty_or_invalid": 0,
        "incompatible": 0,
        "duplicate": 0,
    }

    for path in source_paths:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped["empty_or_invalid"] += 1
                    continue

                target_organ = _normalize_organ(str(record.get("target_organ") or ""))
                model_name = str(record.get("model_name") or "")
                dsc = record.get("dsc")
                if not target_organ or not model_name or dsc is None:
                    skipped["empty_or_invalid"] += 1
                    continue

                if not args.include_incompatible and not _record_is_compatible(
                    registry, model_name, target_organ
                ):
                    skipped["incompatible"] += 1
                    continue

                key = (
                    record.get("sample_id"),
                    target_organ,
                    model_name,
                    dsc,
                    record.get("iou"),
                    record.get("error"),
                )
                if key in seen:
                    skipped["duplicate"] += 1
                    continue
                seen.add(key)

                cleaned = dict(record)
                cleaned["target_organ"] = target_organ
                grouped.setdefault(target_organ, []).append(cleaned)

    if not args.dry_run:
        args.agent_memory_dir.mkdir(parents=True, exist_ok=True)
        for target_organ, records in sorted(grouped.items()):
            destination = args.agent_memory_dir / f"{target_organ}_metrics_history.jsonl"
            with destination.open("w", encoding="utf-8") as file:
                for record in records:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source_files": len(source_paths),
        "agent_memory_dir": str(args.agent_memory_dir),
        "dry_run": bool(args.dry_run),
        "imported_by_target": {
            target: len(records)
            for target, records in sorted(grouped.items())
        },
        "skipped": skipped,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _load_registry(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        items = json.load(file)
    registry: dict[str, set[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        model_name = str(item.get("name") or "")
        if not model_name:
            continue
        targets = {_normalize_organ(str(item.get("target_organ") or ""))}
        aliases = item.get("target_organs") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        targets.update(_normalize_organ(str(alias)) for alias in aliases if alias)
        registry[model_name] = {target for target in targets if target}
    return registry


def _record_is_compatible(
    registry: dict[str, set[str]],
    model_name: str,
    target_organ: str,
) -> bool:
    supported_targets = registry.get(model_name)
    if not supported_targets:
        return False
    if target_organ == "lung" and supported_targets & {"lung", "lungs"}:
        return True
    if target_organ == "heart" and supported_targets & {"heart", "cardiac", "cardiac_silhouette"}:
        return True
    return target_organ in supported_targets


def _normalize_organ(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_").strip()


if __name__ == "__main__":
    main()

"""Which detection classes are measured, and which are only drawn."""
from __future__ import annotations

import json
from pathlib import Path

ROLES = {"measure", "qr", "card", "skip", "draw"}
DEFAULT_NAMES = ("ColorCard", "HalfCones", "Hops", "QR")
MAP_PATH_NAME = "classmap.json"


def default_rows() -> list[dict]:
    return [
        {"source": "ColorCard", "display": "ColorCard", "role": "card"},
        {"source": "HalfCones", "display": "HalfCones", "role": "draw"},
        {"source": "Hops", "display": "Hops", "role": "measure"},
        {"source": "QR", "display": "QR", "role": "qr"},
    ]


def is_default_names(names: list[str]) -> bool:
    return set(names) == set(DEFAULT_NAMES)


def guess_role(name: str) -> str:
    text = name.lower()
    if "qr" in text:
        return "qr"
    if "card" in text or "chart" in text or "color" in text:
        return "card"
    if "hop" in text:
        return "measure"
    return "skip"


def suggest_rows(names: dict[int, str]) -> list[dict]:
    """One row per model class, in index order. Only the first QR and color card are kept."""
    rows = []
    seen_qr = False
    seen_card = False
    for key in sorted(names):
        name = str(names[key])
        role = guess_role(name)
        if role == "qr":
            if seen_qr:
                role = "skip"
            seen_qr = True
        elif role == "card":
            if seen_card:
                role = "skip"
            seen_card = True
        rows.append({"index": int(key), "source": name, "display": name, "role": role})
    return rows


def with_indices(rows: list[dict], names: dict[int, str]) -> list[dict]:
    index_of = {str(value): int(key) for key, value in names.items()}
    stamped = []
    for row in rows:
        item = dict(row)
        source = str(item.get("source") or "")
        if source in index_of:
            item["index"] = index_of[source]
        stamped.append(item)
    return sorted(stamped, key=lambda item: item.get("index", 10**9))


def normalize_rows(rows: list[dict], sources: list[str]) -> list[dict]:
    """Keep one QR class and one color card. Unknown roles become Skip."""
    cleaned = []
    seen_qr = False
    seen_card = False
    by_source = {str(row.get("source")): row for row in rows}
    for source in sources:
        row = by_source.get(source, {})
        role = str(row.get("role") or "skip")
        if role not in ROLES:
            role = "skip"
        if role == "qr":
            if seen_qr:
                role = "skip"
            seen_qr = True
        elif role == "card":
            if seen_card:
                role = "skip"
            seen_card = True
        display = str(row.get("display") or source).strip() or source
        item = {"source": source, "display": display, "role": role}
        raw_index = row.get("index")
        if isinstance(raw_index, int) and not isinstance(raw_index, bool):
            item["index"] = raw_index
        cleaned.append(item)
    return cleaned


def measure_names(rows: list[dict]) -> list[str]:
    return [row["source"] for row in rows if row["role"] == "measure"]


def role_source(rows: list[dict], role: str) -> str | None:
    for row in rows:
        if row["role"] == role:
            return row["source"]
    return None


def display_for(rows: list[dict], source: str) -> str:
    for row in rows:
        if row["source"] == source:
            return row["display"]
    return source


class MapStore:
    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        maps = payload.get("maps") if isinstance(payload, dict) else None
        return maps if isinstance(maps, list) else []

    def find(self, model_path: str, sources: list[str]) -> list[dict] | None:
        for item in self._read():
            if item.get("path") == model_path and item.get("classes") == sources:
                rows = item.get("rows")
                if isinstance(rows, list):
                    return normalize_rows(rows, sources)
        return None

    def save(self, model_path: str, sources: list[str], rows: list[dict]) -> None:
        maps = [item for item in self._read() if not (item.get("path") == model_path and item.get("classes") == sources)]
        maps.append({"path": model_path, "classes": sources, "rows": normalize_rows(rows, sources)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"maps": maps}, indent=2), encoding="utf-8")

"""Find tray folders and count the photos inside them."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class FolderNode:
    path: Path
    direct_images: list[Path] = field(default_factory=list)
    children: list["FolderNode"] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.direct_images) + sum(child.total for child in self.children)


def node_dict(node: FolderNode) -> dict:
    return {
        "name": node.path.name,
        "path": str(node.path),
        "total": node.total,
        "images": [{"name": path.name, "path": str(path)} for path in node.direct_images],
        "children": [node_dict(child) for child in node.children],
    }


def scan_selection(paths: list[Path]) -> FolderNode | None:
    """Build a tree from whole folders and from individually chosen photos."""
    folders: list[Path] = []
    loose: dict[Path, list[Path]] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            folders.append(path.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            loose.setdefault(path.parent.resolve(), []).append(path.resolve())
    chosen_dirs = set(folders)
    nodes: list[FolderNode] = []
    for folder in folders:
        if any(folder != other and other in folder.parents for other in chosen_dirs):
            continue
        node = scan(folder)
        if node is not None:
            nodes.append(node)
    for parent, images in loose.items():
        if any(parent == folder or folder in parent.parents for folder in chosen_dirs):
            continue
        unique = sorted(set(images), key=lambda item: item.name.lower())
        if unique:
            nodes.append(FolderNode(parent, unique, []))
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    return FolderNode(nodes[0].path.parent, [], nodes)


def scan(root: Path) -> FolderNode | None:
    """Return the folder tree, dropping branches that contain no photos."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    node = _scan(root)
    if node.total == 0:
        return None
    return node


def _scan(folder: Path) -> FolderNode:
    images = []
    children = []
    try:
        entries = list(folder.iterdir())
    except OSError:
        return FolderNode(folder)
    for entry in entries:
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        if entry.name.endswith(("_HopBox", "_complete", "_Results")):
            continue
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
            images.append(entry)
        elif entry.is_dir():
            child = _scan(entry)
            if child.total > 0:
                children.append(child)
    images.sort(key=lambda path: path.name.lower())
    children.sort(key=lambda node: node.path.name.lower())
    return FolderNode(folder, images, children)

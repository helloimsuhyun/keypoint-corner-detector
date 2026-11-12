"""Discover and navigate capture directories by natural/numeric scene order.

No Open3D dependency, so navigation rules are independently testable.
"""
from __future__ import annotations

import re
from pathlib import Path


def scene_sort_key(path: Path) -> tuple:
    """Sort scene9 < scene10 rather than lexicographic scene10 < scene9."""
    chunks = re.split(r"(\d+)", path.name.casefold())
    return tuple((1, int(part)) if part.isdigit() else (0, part) for part in chunks), path.name.casefold()


def list_scene_folders(current_scene: str | Path) -> list[Path]:
    """List valid sibling scenes of the same numbered family as current_scene.

    Numbered scenes are grouped by their literal prefix, so a folder like
    demo_corner won't get mixed into a scene_0001 → scene_0002 sequence.
    A non-numbered scene can still navigate all valid sibling folders.
    """
    current = Path(current_scene).expanduser().resolve()
    if not (current / "points.npz").is_file():
        raise FileNotFoundError(f"Scene has no points.npz: {current}")
    match = re.fullmatch(r"(.*?)(\d+)", current.name)
    prefix = match.group(1) if match else None
    candidates = [p for p in current.parent.iterdir()
                  if p.is_dir() and (p / "points.npz").is_file()]
    if prefix is not None:
        candidates = [p for p in candidates
                      if re.fullmatch(re.escape(prefix) + r"\d+", p.name)]
    if current not in candidates:
        candidates.append(current)
    return sorted(candidates, key=scene_sort_key)


def adjacent_scene(scenes: list[Path], current: str | Path, offset: int) -> Path | None:
    """Return the immediately preceding/following existing scene, or None at ends."""
    if offset not in (-1, 1):
        raise ValueError("offset must be -1 or 1")
    pos = scenes.index(Path(current).expanduser().resolve())
    new_pos = pos + offset
    if 0 <= new_pos < len(scenes):
        return scenes[new_pos]
    return None

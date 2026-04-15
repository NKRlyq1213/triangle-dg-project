from __future__ import annotations

from pathlib import Path

from utils.io import ensure_dir


def project_root(anchor: str | Path) -> Path:
    path = Path(anchor).resolve()
    current = path if path.is_dir() else path.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    raise FileNotFoundError("Could not locate project root from anchor path.")


def _relative_output_parts(*parts: str | Path) -> tuple[str, ...]:
    relative = Path(*(str(part).strip() for part in parts if str(part).strip()))
    if relative.is_absolute():
        raise ValueError("output path segments must be relative.")
    if ".." in relative.parts:
        raise ValueError("output path segments must not contain '..'.")
    return tuple(part for part in relative.parts if part not in ("", "."))


def _grouped_output_dir(anchor: str | Path, root_name: str, *parts: str | Path) -> Path:
    root = project_root(anchor)
    relative_parts = _relative_output_parts(*parts)
    path = root / root_name
    if relative_parts:
        path = path.joinpath(*relative_parts)
    return ensure_dir(path)


def experiments_output_dir(anchor: str | Path, *parts: str | Path) -> Path:
    return _grouped_output_dir(anchor, "experiments_outputs", *parts)


def experiments_scratch_dir(anchor: str | Path, *parts: str | Path) -> Path:
    return _grouped_output_dir(anchor, "experiments_outputs", "scratch", *parts)


def photo_output_dir(anchor: str | Path, *parts: str | Path) -> Path:
    return _grouped_output_dir(anchor, "photo", *parts)

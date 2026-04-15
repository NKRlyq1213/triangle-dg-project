from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from experiments.output_paths import (
    experiments_output_dir,
    experiments_scratch_dir,
    photo_output_dir,
    project_root,
)


def _make_anchor() -> Path:
    root = Path.cwd() / ".pytest_local_tmp" / f"output-paths-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    anchor = root / "cli" / "script.py"
    anchor.parent.mkdir()
    anchor.write_text("print('demo')\n", encoding="utf-8")
    return anchor


def test_project_root_resolves_from_file_anchor() -> None:
    anchor = _make_anchor()
    try:
        assert project_root(anchor) == anchor.parents[1]
    finally:
        shutil.rmtree(anchor.parents[1], ignore_errors=True)


def test_experiments_output_dir_creates_grouped_directory() -> None:
    anchor = _make_anchor()
    try:
        output_dir = experiments_output_dir(anchor, "field_h_convergence")
        assert output_dir == anchor.parents[1] / "experiments_outputs" / "field_h_convergence"
        assert output_dir.is_dir()
    finally:
        shutil.rmtree(anchor.parents[1], ignore_errors=True)


def test_experiments_scratch_dir_is_namespaced_under_scratch() -> None:
    anchor = _make_anchor()
    try:
        output_dir = experiments_scratch_dir(anchor, "profiling")
        assert output_dir == anchor.parents[1] / "experiments_outputs" / "scratch" / "profiling"
        assert output_dir.is_dir()
    finally:
        shutil.rmtree(anchor.parents[1], ignore_errors=True)


def test_photo_output_dir_supports_nested_parts() -> None:
    anchor = _make_anchor()
    try:
        output_dir = photo_output_dir(anchor, "visualization", "mesh")
        assert output_dir == anchor.parents[1] / "photo" / "visualization" / "mesh"
        assert output_dir.is_dir()
    finally:
        shutil.rmtree(anchor.parents[1], ignore_errors=True)


@pytest.mark.parametrize("part", ("..", "../escape", r"..\escape", "C:/absolute"))
def test_output_dirs_reject_unsafe_segments(part: str) -> None:
    anchor = _make_anchor()
    try:
        with pytest.raises(ValueError):
            experiments_output_dir(anchor, part)
    finally:
        shutil.rmtree(anchor.parents[1], ignore_errors=True)

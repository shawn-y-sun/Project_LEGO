# =============================================================================
# module: persistence.py
# Purpose: Utilities for persisting and restoring candidate models to disk.
# Key Types/Classes: None
# Key Functions: ensure_segment_dirs, save_index, load_index, save_cm, load_cm, get_segment_dirs
# Dependencies: json, pathlib.Path, typing, cloudpickle
# =============================================================================
"""Persistence helpers for saving and loading candidate models."""

from __future__ import annotations

import json
import cloudpickle
from pathlib import Path
from typing import Dict, List, Optional, Any

from .cm import CM


INDEX_FILENAME = "index.json"


def get_segment_dirs(segment_id: str, base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Return the directory mapping for a segment's persistence layout.

    Parameters
    ----------
    segment_id : str
        Unique identifier for the segment.
    base_dir : Path, optional
        Base directory under which the ``Segment`` folder resides. When ``None``,
        the current working directory is used.

    Returns
    -------
    Dict[str, Path]
        Mapping with keys ``segments_root``, ``segment_dir``, ``cms_dir``,
        ``selected_dir``, and ``passed_dir``.
    """
    base = base_dir or Path.cwd()
    segments_root = base / "Segment"
    segment_dir = segments_root / segment_id
    cms_dir = segment_dir / "cms"
    selected_dir = cms_dir / "selected_cms"
    passed_dir = cms_dir / "passed_cms"
    return {
        "segments_root": segments_root,
        "segment_dir": segment_dir,
        "cms_dir": cms_dir,
        "selected_dir": selected_dir,
        "passed_dir": passed_dir,
    }


def ensure_segment_dirs(segment_id: str, base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Create the required directory structure for persisting CMs.

    Parameters
    ----------
    segment_id : str
        Unique identifier for the segment.
    base_dir : Path, optional
        Base directory under which the ``Segment`` folder resides. When ``None``,
        the current working directory is used.

    Returns
    -------
    Dict[str, Path]
        Same mapping as :func:`get_segment_dirs`, guaranteeing each path exists.
    """
    dirs = get_segment_dirs(segment_id, base_dir)
    for path in dirs.values():
        if path.suffix:
            # Skip filename-like entries; all values here are directories.
            continue
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_index(directory: Path, entries: List[Dict[str, Any]], overwrite: bool) -> None:
    """
    Write an index.json file describing persisted candidate models.

    Parameters
    ----------
    directory : Path
        Directory in which to write ``index.json``.
    entries : list of dict
        Records describing each persisted CM.
    overwrite : bool
        When ``True``, replaces any existing index file. When ``False`` and the
        index already exists, a :class:`FileExistsError` is raised.

    Raises
    ------
    FileExistsError
        If ``overwrite`` is ``False`` and an index file already exists.
    """
    index_path = directory / INDEX_FILENAME
    if index_path.exists() and not overwrite:
        raise FileExistsError(f"Index file already exists at {index_path}.")

    with index_path.open("w", encoding="utf-8") as index_file:
        json.dump(entries, index_file, indent=2)


def load_index(directory: Path) -> List[Dict[str, Any]]:
    """
    Load and return the index.json file from a CM directory.

    Parameters
    ----------
    directory : Path
        Directory containing ``index.json``.

    Returns
    -------
    list of dict
        Parsed index records.

    Raises
    ------
    FileNotFoundError
        If ``index.json`` does not exist in the directory.
    """
    index_path = directory / INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found at {index_path}.")

    with index_path.open("r", encoding="utf-8") as index_file:
        return json.load(index_file)


def save_cm(cm: CM, destination: Path, overwrite: bool) -> None:
    """
    Persist a candidate model to disk as a pickle.

    Parameters
    ----------
    cm : CM
        Candidate model to persist.
    destination : Path
        Target pickle file path.
    overwrite : bool
        When ``True``, existing files are overwritten. When ``False``, a
        :class:`FileExistsError` is raised if the destination already exists.

    Raises
    ------
    FileExistsError
        If ``destination`` exists and ``overwrite`` is ``False``.
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(f"CM file already exists at {destination}.")

    # ``cloudpickle`` tolerates dynamically defined callables that may be
    # attached to models (e.g., user-defined transforms), reducing failures
    # from attribute lookups during standard pickle serialization.
    with destination.open("wb") as cm_file:
        cloudpickle.dump(cm, cm_file)


def load_cm(source: Path) -> CM:
    """
    Load a candidate model pickle from disk.

    Parameters
    ----------
    source : Path
        Pickle file to load.

    Returns
    -------
    CM
        The unpickled candidate model instance.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    """
    if not source.exists():
        raise FileNotFoundError(f"CM file not found at {source}.")

    with source.open("rb") as cm_file:
        return cloudpickle.load(cm_file)

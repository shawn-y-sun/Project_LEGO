# =============================================================================
# module: persistence.py
# Purpose: Persistence helpers for saving and loading candidate models, indexes, and search metadata.
# Key Types/Classes: None
# Key Functions: ensure_segment_dirs, get_segment_dirs, sanitize_segment_id, generate_search_id,
#                get_search_paths, save_index, load_index, save_cm, load_cm
# Dependencies: json, datetime, cloudpickle, pathlib.Path, typing, internal cm
# =============================================================================
"""Persistence helpers for saving and loading candidate models and search metadata."""

from __future__ import annotations

import json
from datetime import datetime
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
        ``selected_dir``, ``passed_dir``, and ``log_dir``.
    """
    base = base_dir or Path.cwd()
    segments_root = base / "Segment"
    segment_dir = segments_root / segment_id
    cms_dir = segment_dir / "cms"
    selected_dir = cms_dir / "selected_cms"
    passed_dir = cms_dir / "passed_cms"
    log_dir = segment_dir / "log"
    return {
        "segments_root": segments_root,
        "segment_dir": segment_dir,
        "cms_dir": cms_dir,
        "selected_dir": selected_dir,
        "passed_dir": passed_dir,
        "log_dir": log_dir,
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


def sanitize_segment_id(segment_id: Any) -> str:
    """
    Sanitize a segment identifier for filesystem-safe usage.

    Parameters
    ----------
    segment_id : Any
        Identifier to sanitize.

    Returns
    -------
    str
        Segment identifier containing only letters, digits, underscores, or hyphens.

    Examples
    --------
    >>> sanitize_segment_id("Seg 1")
    'Seg_1'
    """

    raw = str(segment_id)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)


def generate_search_id(segment_id: Any) -> str:
    """
    Generate a timestamped search identifier scoped to a segment.

    The identifier follows ``search_<segment_id>_<YYYYMMDD_HHMMSS>`` and uses a
    sanitized ``segment_id`` suitable for filesystem paths.

    Parameters
    ----------
    segment_id : Any
        Segment identifier to embed in the search ID.

    Returns
    -------
    str
        Unique search identifier incorporating the sanitized segment label.

    Examples
    --------
    >>> generate_search_id("CNIBusiness")  # doctest: +SKIP
    'search_CNIBusiness_20250101_120000'
    """

    sanitized = sanitize_segment_id(segment_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"search_{sanitized}_{timestamp}"


def get_search_paths(segment_id: Any, search_id: str, base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Return search-scoped directories and file paths for a segment search run.

    Parameters
    ----------
    segment_id : Any
        Identifier for the segment whose search artifacts should be located.
    search_id : str
        Concrete search identifier (``search_<segment>_<YYYYMMDD_HHMMSS>``).
    base_dir : Path, optional
        Base working directory; defaults to the current working directory when omitted.

    Returns
    -------
    Dict[str, Path]
        Mapping that includes ``cms_root``, ``search_cms_dir``, ``passed_cms_dir``,
        ``selected_cms_dir``, ``log_dir``, ``log_file``, ``progress_file``, and
        ``config_file``.

    Examples
    --------
    >>> paths = get_search_paths("seg1", "search_seg1_20250101_120000")
    >>> paths["search_cms_dir"].name
    'search_seg1_20250101_120000'
    """

    base = base_dir or Path.cwd()
    dirs = ensure_segment_dirs(str(segment_id), base)
    cms_root = dirs["cms_dir"]
    search_cms_dir = cms_root / search_id
    passed_cms_dir = search_cms_dir / "passed_cms"
    selected_cms_dir = search_cms_dir / "selected_cms"
    log_dir = dirs["log_dir"]
    log_file = log_dir / f"{search_id}.log"
    progress_file = log_dir / f"{search_id}.progress"
    config_file = search_cms_dir / "config.json"

    # Ensure search-scoped directories exist so early logging/saving succeeds.
    search_cms_dir.mkdir(parents=True, exist_ok=True)
    passed_cms_dir.mkdir(parents=True, exist_ok=True)
    selected_cms_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cms_root": cms_root,
        "search_cms_dir": search_cms_dir,
        "passed_cms_dir": passed_cms_dir,
        "selected_cms_dir": selected_cms_dir,
        "log_dir": log_dir,
        "log_file": log_file,
        "progress_file": progress_file,
        "config_file": config_file,
    }


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

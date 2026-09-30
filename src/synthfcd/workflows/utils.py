"""
Utility functions for workflows.

This module groups the lower-level building blocks used by
:class:`synthfcd.workflows.workflow.SimulationWorkflow`:

- thin ANTs IO wrappers (reading, writing, and numpy <-> ANTs conversion);
- data exploration helpers that resolve a subject's segmentation mask and
  modality images from a discovered "anchor" file and load them into numpy;
- a logger factory honoring the workflow log-file/level settings;
- a helper that flattens drawn parameter dictionaries into a tabular form;
- serial/parallel execution helpers built around a pickleable
  ``process_single`` worker that runs the simulation for a single subject.
"""

from __future__ import annotations

__all__ = [
    "ants_image_read",
    "ants_image_write",
    "build_filename",
    "execute",
    "find_matching_files",
    "flatten_params",
    "get_logger",
    "get_spacing",
    "load_subject_data",
    "numpy_to_ants",
    "params_to_dataframe",
    "resolve_input_paths",
    "write_outputs",
]

import json
import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import ants
import numpy as np

from synthfcd.utils._aliases import (
    _LogLevelType,
    _SimulationResultType,
)
from synthfcd.utils._validators import (
    validate_obj_type,
    validate_spacing,
)
from synthfcd.utils.data import get_ext, resolve_path

if TYPE_CHECKING:
    import pandas as pd


# Relative tolerance used when comparing voxel spacings across images.
_SPACING_RTOL: float = 1e-3


# ----------------------------------------------------------------#
# ANTs IO helpers
# ----------------------------------------------------------------#
def ants_image_read(
    path: str | Path,
    *,
    pixeltype: str = "float",
) -> ants.ANTsImage:
    """
    Read an image from disk as an ``ANTsImage``.

    Images are automatically reoriented for internal consistency using the
    ANTs flag `reorient=True`.

    Args:
        path (str | Path):
            Path to the image file.
        pixeltype (str, optional):
            The pixel type to read the image as. Defaults to ``"float"``.

    Returns:
        ants.ANTsImage:
            The loaded (and optionally reoriented) image.

    Raises:
        FileNotFoundError:
            If ``path`` does not exist.
        ValueError:
            If ``orientation`` is not a valid 3-character orientation code.
    """
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    return ants.image_read(
        str(path),
        pixeltype=pixeltype,
        reorient=True,
    )


def ants_image_write(image: ants.ANTsImage, path: str | Path) -> None:
    """
    Write an ``ANTsImage`` to disk, creating parent directories as needed.

    Args:
        image (ants.ANTsImage):
            The image to write.
        path (str | Path):
            The destination file path.
    """
    validate_obj_type(image, "image", ants.ANTsImage)
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(image, str(path))


def numpy_to_ants(
    array: np.ndarray,
    reference: ants.ANTsImage,
) -> ants.ANTsImage:
    """
    Convert a numpy array into an ``ANTsImage`` using a reference for the header.

    The spacing, origin, and direction of ``reference`` are copied onto the new
    image so the result lives in the same physical space as the source data.

    Args:
        array (np.ndarray):
            The array holding the voxel data.
        reference (ants.ANTsImage):
            The image whose header (spacing, origin, direction) is reused.

    Returns:
        ants.ANTsImage:
            The image wrapping ``array`` with the reference's header.

    Raises:
        ValueError:
            If ``array`` and ``reference`` do not share the same shape.
    """
    validate_obj_type(reference, "reference", ants.ANTsImage)
    array = np.asarray(array)
    if array.shape != reference.shape:
        raise ValueError(
            f"`array` shape {array.shape} does not match the reference image "
            f"shape {reference.shape}."
        )

    # ANTs does not support float64; downcast for safety while preserving ints.
    if array.dtype == np.float64:
        array = array.astype(np.float32)

    return ants.from_numpy(
        array,
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


def get_spacing(image: ants.ANTsImage) -> tuple[float, float, float]:
    """
    Return the voxel spacing of an ``ANTsImage`` as a tuple of floats.

    Args:
        image (ants.ANTsImage):
            The image to read the spacing from.

    Returns:
        tuple[float, float, float]:
            The voxel spacing along each axis.
    """
    validate_obj_type(image, "image", ants.ANTsImage)
    return tuple(float(s) for s in image.spacing)  # type: ignore[return-value]


def build_filename(filename: str, suffix: str, target_ext: str | None = None) -> str:
    """
    Build a filename with a suffix and optional target extension.

    Args:
        filename (str):
            The filename to build.
        suffix (str):
            The suffix to add to the filename.
        target_ext (str | None, optional):
            The target extension to add to the filename. Defaults to ``None``.

    Returns:
        str:
            The built filename.
    """
    validate_obj_type(filename, "filename", str)
    validate_obj_type(suffix, "suffix", str)
    if target_ext is not None:
        validate_obj_type(target_ext, "target_ext", str)
    ext = get_ext(filename)
    stem = filename[: -len(ext)] if ext else filename
    return f"{stem}_{suffix}{ext if target_ext is None else target_ext}"


# ----------------------------------------------------------------#
# Data exploration and saving helpers
# ----------------------------------------------------------------#


def find_matching_files(
    directory: str | Path,
    suffix: str,
    description: str,
    raise_if_missing: bool = True,
    raise_if_many: bool = True,
) -> Path | None:
    """
    Find a single NIfTI file in a directory whose name ends with a given suffix.

    Matching is non-recursive (direct children only) and restricted to NIfTI
    files (``.nii`` / ``.nii.gz``) whose name ends with ``{suffix}`` right before
    the extension, e.g. suffix ``"_synthseg"`` matches ``sub-01_synthseg.nii.gz``.

    Args:
        directory (str | Path):
            The directory to search (non-recursively).
        suffix (str):
            The suffix that a file name must end with (before the extension).
        description (str):
            A human-readable description used in error/skip messages.
        raise_if_missing (bool):
            Whether to raise when there are no matches. If ``False``, returns
            ``None`` instead.
        raise_if_many (bool):
            Whether to raise when there is more than one match. If ``False``, the
            first match (sorted) is returned.

    Returns:
        Path | None:
            The single matching file path, or ``None`` when missing and
            ``raise_if_missing`` is ``False``.

    Raises:
        FileNotFoundError:
            If no match is found and ``raise_if_missing`` is ``True``.
        ValueError:
            If multiple matches are found and ``raise_if_many`` is ``True``.
    """
    directory = resolve_path(directory)
    found = sorted(
        p
        for p in directory.glob(f"*{suffix}.nii*")
        if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))
    )
    if len(found) == 0:
        if raise_if_missing:
            raise FileNotFoundError(f"No file found for {description}.")
        return None
    if len(found) > 1 and raise_if_many:
        listed = ", ".join(p.name for p in found)
        raise ValueError(
            f"Multiple files found for {description}: {listed}. "
            "Set `raise_if_many=False` to use the first match."
        )
    return found[0]


def resolve_input_paths(
    anchor: str | Path,
    *,
    seg_mask_suffix: str,
    modality_suffixes: list[str],
    raise_if_missing: bool = True,
    raise_if_many: bool = True,
) -> dict[str, Any] | None:
    """
    Resolve a subject's segmentation mask and modality image paths.

    Given a discovered ``anchor`` file (the "active" file produced by the data
    explorer), this looks into its parent directory for:

    - the segmentation mask, matched by ``seg_mask_suffix``;
    - one modality image per entry in ``modality_suffixes``.

    Args:
        anchor (str | Path):
            The discovered anchor file. Its parent directory is searched.
        seg_mask_suffix (str):
            Suffix used to locate the segmentation mask.
        modality_suffixes (list[str]):
            Suffixes used to locate each modality image, in order.
        raise_if_missing (bool, optional):
            Whether to raise if a required file is missing. If ``False`` and any
            required file is missing, ``None`` is returned (subject skipped).
            Defaults to ``True``.
        raise_if_many (bool, optional):
            Whether to raise if multiple files match a suffix. If ``False``, the
            first match (sorted) is used. Defaults to ``True``.

    Returns:
        dict[str, Any] | None:
            A dictionary with keys ``"seg_mask_path"`` and ``"image_paths"``, or
            ``None`` if the subject is skipped due to missing files.
    """
    anchor = resolve_path(anchor)

    seg_mask_path = find_matching_files(
        directory=anchor.parent,
        suffix=seg_mask_suffix,
        description=f"segmentation mask (suffix '{seg_mask_suffix}') in {anchor.parent}",
        raise_if_missing=raise_if_missing,
        raise_if_many=raise_if_many,
    )
    if seg_mask_path is None:
        return None

    image_paths: list[Path] = []
    for suffix in modality_suffixes:
        image_path = find_matching_files(
            directory=anchor.parent,
            suffix=suffix,
            description=f"modality (suffix '{suffix}') in {anchor.parent}",
            raise_if_missing=raise_if_missing,
            raise_if_many=raise_if_many,
        )
        if image_path is None:
            return None
        image_paths.append(image_path)

    return {"seg_mask_path": seg_mask_path, "image_paths": image_paths}


def load_subject_data(
    seg_mask_path: str | Path,
    image_paths: list[str | Path],
) -> dict[str, Any]:
    """
    Load a subject's segmentation mask and modality images into numpy arrays.

    The segmentation mask defines the anatomical reference: its spacing is used as
    the subject's spacing and every modality image is checked to match it (within
    a small tolerance). The loaded ``ANTsImage`` headers are returned alongside the
    arrays so outputs can be written back into the same physical space.

    Args:
        seg_mask_path (str | Path):
            Path to the segmentation mask.
        image_paths (list[str | Path]):
            Paths to the modality images, in order.

    Returns:
        dict[str, Any]:
            A dictionary with keys:
            - ``"images"``: list of modality images as ``float32`` arrays.
            - ``"segmentation_mask"``: the mask as an ``int32`` array.
            - ``"spacing"``: the voxel spacing tuple.
            - ``"image_refs"``: list of modality ``ANTsImage`` headers.
            - ``"seg_ref"``: the segmentation ``ANTsImage`` header.

    Raises:
        ValueError:
            If a modality image spacing or shape does not match the mask.
    """
    seg_ref = ants_image_read(seg_mask_path)
    spacing = get_spacing(seg_ref)
    validate_spacing(spacing, "segmentation mask")
    seg_mask = np.rint(seg_ref.numpy()).astype(np.int32)

    images: list[np.ndarray] = []
    image_refs: list[ants.ANTsImage] = []
    for path in image_paths:
        ref = ants_image_read(path)
        img_spacing = get_spacing(ref)
        if not np.allclose(img_spacing, spacing, rtol=_SPACING_RTOL):
            raise ValueError(
                f"Spacing mismatch between segmentation mask {spacing} and image "
                f"{Path(path).name} {img_spacing}."
            )
        if ref.shape != seg_ref.shape:
            raise ValueError(
                f"Shape mismatch between segmentation mask {seg_ref.shape} and "
                f"image {Path(path).name} {ref.shape}."
            )
        images.append(ref.numpy().astype(np.float32))
        image_refs.append(ref)

    return {
        "images": images,
        "segmentation_mask": seg_mask,
        "spacing": spacing,
        "image_refs": image_refs,
        "seg_ref": seg_ref,
    }


def write_outputs(
    result: _SimulationResultType,
    entry: dict[str, Any],
    data: dict[str, Any],
) -> list[str]:
    """
    Write the simulation outputs for a single entry to disk.

    Args:
        result (dict[str, Any]):
            The pipeline result (``out_images``, ``out_seg_mask``, ``stats``,
            ``extras``).
        entry (dict[str, Any]):
            The staged entry record carrying the resolved ``out_paths``, which contains
            only the paths of the enabled outputs.
        data (dict[str, Any]):
            The loaded entry data carrying the reference headers.

    Returns:
        list[str]:
            The paths written, as strings.
    """
    out_paths = entry.get("out_paths", None)
    if out_paths is None:
        raise ValueError("`out_paths` not found in entry")
    written: list[str] = []

    # The presence of a key in `out_paths` indicates the output is enabled
    # (see `SimpleFCDWorkflow._build_output_paths`).
    if "out_images" in out_paths:
        for arr, ref, path in zip(
            result["out_images"],
            data["image_refs"],
            out_paths["out_images"],
            strict=True,
        ):
            ants_image_write(numpy_to_ants(arr, ref), path)
            written.append(str(path))

    if "out_seg_mask" in out_paths:
        seg = numpy_to_ants(result["out_seg_mask"], data["seg_ref"])
        ants_image_write(seg, out_paths["out_seg_mask"])
        written.append(str(out_paths["out_seg_mask"]))

    if "out_lesion_mask" in out_paths:
        lesion = result.get("extras", {}).get("out_target")
        if lesion is None:
            raise ValueError(
                "Lesion mask output was requested but the pipeline result has no "
                "`extras['out_target']`."
            )
        lesion_img = numpy_to_ants(lesion.astype(np.int32), data["seg_ref"])
        ants_image_write(lesion_img, out_paths["out_lesion_mask"])
        written.append(str(out_paths["out_lesion_mask"]))

    if "out_stats" in out_paths:
        stats_path = resolve_path(out_paths["out_stats"])
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(result.get("stats", {}), fh, indent=2, default=str)
        written.append(str(stats_path))

    return written


# ----------------------------------------------------------------#
# Logging
# ----------------------------------------------------------------#
def get_logger(
    name: str,
    *,
    log_file: str | Path | None = None,
    log_level: _LogLevelType = "INFO",
) -> logging.Logger:
    """
    Create (or retrieve) a logger configured for the workflow.

    A console handler is always attached; a file handler is added when
    ``log_file`` is provided. Handlers are only attached once per logger name to
    avoid duplicate records across repeated calls.

    Args:
        name (str):
            The logger name.
        log_file (str | Path | None, optional):
            Path to a log file. If ``None``, only console logging is used.
            Defaults to ``None``.
        log_level (str, optional):
            The logging level. Defaults to ``"INFO"``.

    Returns:
        logging.Logger:
            The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    have_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not have_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = resolve_path(log_file)
        existing_files = {
            Path(h.baseFilename)
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        }
        if log_file not in existing_files:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# ----------------------------------------------------------------#
# Parameter tracking
# ----------------------------------------------------------------#
def flatten_params(value: Any, *, prefix: str = "") -> dict[str, Any]:
    """
    Flatten a nested parameter structure into dotted-path -> value pairs.

    Dictionaries contribute their keys to the path, while list/tuple elements
    contribute their index wrapped in brackets, for example
    ``"intensity_params.[0].boundary_blurring.n_iters"``.

    Args:
        value (Any):
            The (possibly nested) value to flatten.
        prefix (str, optional):
            Prefix prepended to the produced paths. Used during recursion.
            Defaults to ``""``.

    Returns:
        dict[str, Any]:
            Mapping from dotted path to terminal (non-container) value.
    """
    flat: dict[str, Any] = {}

    if isinstance(value, dict):
        for key, item in value.items():
            path = str(key) if prefix == "" else f"{prefix}.{key}"
            flat.update(flatten_params(item, prefix=path))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            path = f"[{i}]" if prefix == "" else f"{prefix}.[{i}]"
            flat.update(flatten_params(item, prefix=path))
    else:
        flat[prefix] = value

    return flat


def params_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Build a DataFrame from staged subject records.

    Each record is expected to expose an identifier under ``"id"`` and the drawn
    parameter dictionary under ``"params"``. The parameters are flattened so each
    terminal value becomes its own column (see :func:`flatten_params`).

    Args:
        records (list[dict[str, Any]]):
            The staged subject records.

    Returns:
        pandas.DataFrame:
            One row per record, with an ``"id"`` column followed by the flattened
            parameter columns.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {"id": record.get("id")}
        row.update(flatten_params(record.get("params", {})))
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------#
# Execution
# ----------------------------------------------------------------#
def _resolve_num_workers(num_workers: int | Literal["auto"]) -> int:
    """
    Resolve the requested worker count to a concrete positive integer.
    """
    if num_workers == "auto":
        return os.cpu_count() or 1
    return max(1, int(num_workers))


def execute(
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    items: Iterable[dict[str, Any]],
    *,
    num_workers: int | Literal["auto"] = 1,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """
    Execute a per-item worker function serially or across processes.

    Args:
        fn (Callable[[dict], dict]):
            The worker callable applied to each item. Must be pickleable when
            ``num_workers > 1`` (e.g. ``functools.partial(process_single,
            verbose=..., label_enum=...)``).
        items (Iterable[dict]):
            The staged subject records to process.
        num_workers (int | Literal["auto"], optional):
            Number of worker processes. ``1`` runs serially; ``"auto"`` uses the
            CPU count. Defaults to ``1``.
        logger (logging.Logger | None, optional):
            Logger for per-item progress messages. Defaults to ``None``.

    Returns:
        list[dict[str, Any]]:
            The status dictionaries returned by ``fn`` for each item.
    """
    items = list(items)
    workers = _resolve_num_workers(num_workers)

    def _log(status: dict[str, Any]) -> None:
        if logger is None:
            return
        if status.get("status") == "SUCCESS":
            logger.info("SUCCESS: %s", status.get("id"))
        else:
            logger.error("FAILURE: %s (%s)", status.get("id"), status.get("error"))

    results: list[dict[str, Any]] = []

    if workers == 1 or len(items) <= 1:
        for item in items:
            status = fn(item)
            _log(status)
            results.append(status)
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            status = future.result()
            _log(status)
            results.append(status)

    return results

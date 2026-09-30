"""
Concrete workflow using the simple FCD pipeline.
"""

from __future__ import annotations

__all__ = [
    "SimpleFCDWorkflow",
]

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

from nifti_finder.explorers import FileFinder

from synthfcd.pipelines.simple_fcd import simple_fcd_simulator
from synthfcd.presets.simple_fcd import draw_simple_fcd_params
from synthfcd.utils import get_rng, resolve_path
from synthfcd.utils._aliases import (
    _DataExplorerType,
    _LabelEnumType,
    _LogLevelType,
    _NiftiFinderConfigType,
    _RNGType,
    _SimulationResultType,
    _sMRIType,
)
from synthfcd.utils._base import _LabelEnum
from synthfcd.utils._validators import validate_class_type, validate_literal_str
from synthfcd.utils.seg_labels import SynthSegLabel
from synthfcd.workflows.workflow import SimulationWorkflow

from .explorer import get_data_explorer
from .utils import (
    build_filename,
    execute,
    get_logger,
    load_subject_data,
    params_to_dataframe,
    resolve_input_paths,
    validate_obj_type,
    write_outputs,
)


class SimpleFCDWorkflow(SimulationWorkflow):
    def __init__(
        self,
        *,
        explorer: _DataExplorerType | _NiftiFinderConfigType | None = None,
        seg_mask_suffix: str = "_synthseg",
        modality_suffixes: list[str] | str | None = None,
        modality_names: list[_sMRIType] | _sMRIType | None = None,
        raise_if_missing: bool = True,
        raise_if_many: bool = True,
        param_presets: str = "default",
        num_workers: int | Literal["auto"] = 1,
        log_file: str | None = None,
        log_level: _LogLevelType = "INFO",
        verbose: bool = False,
        out_dir: str | None = None,
        save_images: bool = True,
        save_seg_mask: bool = True,
        save_lesion_mask: bool = True,
        save_stats: bool = True,
        save_params_csv: bool = True,
        out_image_suffix: str = "synthfcd",
        out_seg_mask_suffix: str = "synthfcd_seg",
        out_lesion_mask_suffix: str = "synthfcd_lesion_seg",
        out_stats_suffix: str = "synthfcd_stats",
        out_params_csv_name: str = "synthfcd_params.csv",
        random_seed: int | None = None,
        random_state: _RNGType | None = None,
        label_enum: _LabelEnumType = SynthSegLabel,
    ) -> None:
        # Configure the data explorer
        if explorer is not None:
            self._explorer = self._get_explorer(explorer)
        else:
            self._explorer = None

        # Get modality suffixes to use in file searching and naming for use
        # in `draw_simple_fcd_params`. At least one modality is required, as the
        # simulation operates on (and draws per-sequence parameters for) images.
        self._modality_suffixes, self._modality_names = (
            self._get_modality_suffixes_and_names(modality_suffixes, modality_names)
        )
        if len(self._modality_suffixes) == 0:
            raise ValueError(
                "`modality_suffixes` and `modality_names` must be provided "
                "(at least one modality is required to run the simulation)."
            )

        # Configure logging
        self._logger = get_logger(
            self.__class__.__name__, log_file=log_file, log_level=log_level
        )

        # Configure execution
        is_valid_int = isinstance(num_workers, int) and not isinstance(
            num_workers, bool
        )
        if not (is_valid_int or num_workers == "auto"):
            raise ValueError("`num_workers` must be an integer or `auto`")
        self._num_workers: int | Literal["auto"] = num_workers

        # Configure deterministic execution
        self._rng = get_rng(random_seed, random_state)

        # Validate boolean flags
        for name, value in (
            ("raise_if_missing", raise_if_missing),
            ("raise_if_many", raise_if_many),
            ("verbose", verbose),
            ("save_images", save_images),
            ("save_seg_mask", save_seg_mask),
            ("save_lesion_mask", save_lesion_mask),
            ("save_stats", save_stats),
            ("save_params_csv", save_params_csv),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean; got {type(value)}")

        if not any(
            (save_images, save_seg_mask, save_lesion_mask, save_stats, save_params_csv)
        ):
            raise ValueError(
                "At least one of `save_images`, `save_seg_mask`, `save_lesion_mask`, "
                "`save_stats`, or `save_params_csv` must be True."
            )

        self._raise_if_missing = raise_if_missing
        self._raise_if_many = raise_if_many
        self._verbose = verbose
        self._save_images = save_images
        self._save_seg_mask = save_seg_mask
        self._save_lesion_mask = save_lesion_mask
        self._save_stats = save_stats
        self._save_params_csv = save_params_csv

        # Configure outputs
        if out_dir is not None:
            validate_obj_type(out_dir, "out_dir", (str, Path))
        self._out_dir = resolve_path(out_dir) if out_dir is not None else None

        for name, value in (
            ("seg_mask_suffix", seg_mask_suffix),
            ("out_image_suffix", out_image_suffix),
            ("out_seg_mask_suffix", out_seg_mask_suffix),
            ("out_lesion_mask_suffix", out_lesion_mask_suffix),
            ("out_stats_suffix", out_stats_suffix),
            ("out_params_csv_name", out_params_csv_name),
        ):
            if not isinstance(value, str):
                raise TypeError(f"`{name}` must be a string; got {type(value)}")
        self._seg_mask_suffix = seg_mask_suffix
        self._out_image_suffix = out_image_suffix
        self._out_seg_mask_suffix = out_seg_mask_suffix
        self._out_lesion_mask_suffix = out_lesion_mask_suffix
        self._out_stats_suffix = out_stats_suffix
        self._out_params_csv_name = out_params_csv_name

        # Configure the label enumeration
        validate_class_type(label_enum, "label_enum", _LabelEnum)
        self._label_enum = label_enum

        # Configure the parameter presets bundle forwarded to `draw_simple_fcd_params`.
        # Value validation is delegated to `draw_simple_fcd_params` (single source of truth).
        validate_obj_type(param_presets, "param_presets", str)
        self._param_presets = param_presets

    def _get_explorer(
        self,
        value: _DataExplorerType | _NiftiFinderConfigType,
    ) -> _DataExplorerType:
        """
        Get the data explorer.
        """
        validate_obj_type(value, "explorer", (FileFinder, dict))

        if isinstance(value, FileFinder):
            return value

        allowed_keys = ("patterns", "levels", "filters")
        if any(key not in allowed_keys for key in value.keys()):
            raise ValueError(
                f"Invalid keys for `explorer`: {value.keys()}; should contain only: {allowed_keys}"
            )

        return get_data_explorer(
            patterns=value.get("patterns", "*.nii*"),
            levels=value.get("levels", None),
            filters=value.get("filters", None),
        )

    def _get_modality_suffixes_and_names(
        self,
        suffix_values: list[str] | str | None,
        name_values: list[_sMRIType] | _sMRIType | None,
    ) -> tuple[list[str], list[_sMRIType]]:
        """
        Get the modality suffixes and names.
        """
        if suffix_values is None and name_values is None:
            return [], []
        if any(v is None for v in (suffix_values, name_values)):
            raise ValueError(
                "Cannot provide only one of `modality_suffixes` and `modality_names`"
            )

        validate_obj_type(suffix_values, "suffix_values", (list, str))
        if isinstance(suffix_values, str):
            suffix_values = [suffix_values]

        validate_obj_type(name_values, "name_values", (list, str))
        if isinstance(name_values, str):
            name_values = [name_values]

        suffix_values = cast(list, suffix_values)
        name_values = cast(list, name_values)

        if len(suffix_values) != len(name_values):
            raise ValueError(
                "`modality_suffixes` and `modality_names` must be the same length"
            )

        for i, (suffix, name) in enumerate(
            zip(suffix_values, name_values, strict=True)
        ):
            validate_obj_type(suffix, f"modality_suffixes[{i}]", str)
            validate_literal_str(name, f"modality_names[{i}]", ("T1like", "T2like"))

        return suffix_values, name_values

    def _discover_anchors(
        self, inputs: str | Path | list[str | Path]
    ) -> list[tuple[Path, Path]]:
        """
        Discover the anchor files to process and their base directory.

        Each input is resolved as follows:
        - A directory is scanned with the configured explorer; every matching file
          is an anchor and the directory itself is its base.
        - A file is treated directly as an anchor (the explorer is ignored) and its
          parent directory is its base.

        Args:
            inputs (str | Path | list[str | Path]):
                One or more directories to explore and/or files to process.

        Returns:
            list[tuple[Path, Path]]:
                ``(anchor, base)`` pairs, where ``base`` is used to mirror the
                output directory structure.

        Raises:
            ValueError:
                If ``inputs`` is ``None`` or empty.
            FileNotFoundError:
                If an input path does not exist.
        """
        if inputs is None:
            raise ValueError(
                "`inputs` must be one or more directories to explore or files to "
                "process; got None."
            )
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) == 0:
            raise ValueError("`inputs` must be a non-empty list")

        anchors: list[tuple[Path, Path]] = []
        for item in inputs:
            path = resolve_path(item)
            if path.is_dir():
                if self._explorer is None:
                    raise ValueError(
                        "cannot discover directories when explorer is not configured"
                    )
                found = self._explorer.list(path, sort=True, unique=True)
                self._logger.info("Found %d file(s) under %s", len(found), path)
                anchors.extend((Path(f), path) for f in found)
            elif path.is_file():
                anchors.append((path, path.parent))
            else:
                raise FileNotFoundError(f"Input path does not exist: {path}")

        return anchors

    def _discover_files(self, anchor_path: Path) -> dict[str, Any] | None:
        """
        Discover all input files for a given anchor.
        """
        paths = resolve_input_paths(
            anchor_path,
            seg_mask_suffix=self._seg_mask_suffix,
            modality_suffixes=self._modality_suffixes,
            raise_if_missing=self._raise_if_missing,
            raise_if_many=self._raise_if_many,
        )
        if paths is None:
            self._logger.info("Skipping %s due to missing files", anchor_path)
            return None
        return paths

    def _build_output_paths(
        self, anchor_path: Path, base: Path, paths: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build the output image, segmentation, stats, and parameter CSV paths for a given
        anchor.
        """
        if not isinstance(paths, dict) or set(paths.keys()) != set(
            {"image_paths", "seg_mask_path"}
        ):
            raise ValueError(
                f"Invalid `paths`; expected a dictionary containing only "
                f"'image_paths', 'seg_mask_path' keys, got {paths.keys()}"
            )

        if self._out_dir is None:
            out_dir = anchor_path.parent
        else:
            out_root = resolve_path(self._out_dir)
            try:
                rel = anchor_path.parent.relative_to(base)
            except ValueError:
                rel = Path(anchor_path.parent.name)
            out_dir = out_root / rel

        output_paths: dict[str, Path | list[Path]] = {}
        if self._save_images:
            output_paths["out_images"] = [
                out_dir / build_filename(Path(p).name, self._out_image_suffix)
                for p in paths["image_paths"]
            ]
        if self._save_seg_mask:
            output_paths["out_seg_mask"] = out_dir / build_filename(
                Path(paths["seg_mask_path"]).name, self._out_seg_mask_suffix
            )
        if self._save_lesion_mask:
            output_paths["out_lesion_mask"] = out_dir / build_filename(
                anchor_path.name, self._out_lesion_mask_suffix
            )
        if self._save_stats:
            output_paths["out_stats"] = out_dir / build_filename(
                anchor_path.name, self._out_stats_suffix, target_ext=".json"
            )

        return output_paths

    def _stage(self, anchors: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
        """
        Stage the input entries.
        """
        entries: list[dict[str, Any]] = []
        for anchor in anchors:
            anchor_path, base = anchor
            paths = self._discover_files(anchor_path)
            if paths is None:
                continue
            output_paths = self._build_output_paths(anchor_path, base, deepcopy(paths))
            params = draw_simple_fcd_params(
                sequences=self._modality_names,
                presets=self._param_presets,
                random_state=self._rng,
                label_enum=self._label_enum,
            )
            entries.append(
                {
                    "id": str(anchor_path),
                    "anchor": str(anchor_path),
                    "seg_mask": str(paths["seg_mask_path"]),
                    "images": [str(p) for p in paths["image_paths"]],
                    "out_paths": output_paths,
                    "params": params,
                }
            )
        return entries

    @staticmethod
    def process_single(
        entry: dict[str, Any], verbose: bool, label_enum: _LabelEnumType
    ) -> dict[str, Any]:
        """
        Run the simulation pipeline for a single staged subject.

        This is the unit of work for both serial and parallel execution. It loads the
        subject data, runs the pipeline with the drawn parameters, writes the enabled
        outputs, and reports success/failure. Exceptions are captured and returned
        rather than raised so a single failing subject does not abort the workflow.

        Args:
            entry (dict[str, Any]):
                A staged subject record with keys ``"id"``, ``"seg_mask"``,
                ``"images"``, ``"out_paths"``, and ``"params"`` (the latter holding
                ``growth_params``, ``deformation_params``, ``intensity_params``, and
                the ``FCD_type`` metadata).
            verbose (bool):
                Whether to print verbose output.
            label_enum (type[_LabelEnum]):
                The label enumeration to use.

        Returns:
            dict[str, Any]:
                A status dictionary with keys ``"id"``, ``"status"`` (``"SUCCESS"``
                or ``"FAILURE"``), ``"stats"``, ``"outputs"``, and ``"error"``.
        """
        entry_id = entry.get("id")
        try:
            data = load_subject_data(
                seg_mask_path=entry["seg_mask"],
                image_paths=entry["images"],
            )

            # Pass the simulator's parameters explicitly; the drawn ``params`` also
            # carries ``FCD_type`` (preset metadata kept for tracking), which is not
            # a simulator argument and is therefore intentionally not forwarded.
            params = entry["params"]
            result = cast(
                _SimulationResultType,
                simple_fcd_simulator(
                    images=data["images"],
                    seg_mask=data["segmentation_mask"],
                    spacing=data["spacing"],
                    growth_params=params["growth_params"],
                    deformation_params=params["deformation_params"],
                    intensity_params=params["intensity_params"],
                    verbose=verbose,
                    label_enum=label_enum,
                ),
            )

            written = write_outputs(result, entry, data)

            return {
                "id": entry_id,
                "status": "SUCCESS",
                "stats": result.get("stats", {}),
                "outputs": written,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - surfaced in the returned status
            return {
                "id": entry_id,
                "status": "FAILURE",
                "stats": {},
                "outputs": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def run(
        self,
        inputs: str | Path | list[str | Path],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run the workflow end-to-end over the provided inputs.

        ``inputs`` are directories for data exploration and/or specific anchor
        files. Directories are scanned with the configured explorer; files are used
        directly (the explorer is ignored) and their parents are searched for the
        segmentation mask and modality images. When ``out_dir`` is provided it is
        used as a base directory and the input structure is mirrored beneath it;
        otherwise outputs are written next to the source files.

        Args:
            inputs (str | Path | list[str | Path]):
                One or more directories to explore and/or files to process.
            dry_run (bool, optional):
                If ``True``, stage subjects (resolve files, draw parameters, build
                output paths, optionally save the parameter CSV) without running the
                simulation. Defaults to ``False``.

        Returns:
            dict[str, Any]:
                A summary with keys ``"entries"`` (the staged records),
                ``"results"`` (per-subject status dictionaries; empty on dry runs),
                ``"n_success"``, and ``"n_failure"``.
        """
        self._logger.info("Starting workflow...")
        self._logger.info(
            "pipeline=%s | params=%s | workers=%s | out_dir=%s",
            "simple_fcd_simulator",
            "draw_simple_fcd_params",
            self._num_workers,
            self._out_dir,
        )

        # Explore data and resolve anchors
        anchors = self._discover_anchors(inputs)
        if len(anchors) == 0:
            self._logger.warning("No inputs discovered; nothing to do.")
            return {"entries": [], "results": [], "n_success": 0, "n_failure": 0}

        # Stage data: resolve siblings, draw parameters, build output paths
        entries = self._stage(anchors)
        self._logger.info("Staged %d subject(s).", len(entries))

        # Track parameters; optionally persist the parameter table
        if self._save_params_csv and len(entries) > 0:
            df = params_to_dataframe(entries)
            if self._out_dir is None:
                self._logger.warning(
                    "`save_params_csv=True` but `out_dir` is None; "
                    "saving parameter table to the base directory of the first anchor."
                )
                csv_path = anchors[0][1] / self._out_params_csv_name
            else:
                csv_path = resolve_path(self._out_dir) / self._out_params_csv_name
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False, mode="w")
            self._logger.info("Wrote parameter table to %s", csv_path)

        # Dry run: stop before any simulation/IO
        if dry_run:
            self._logger.info("Dry run complete; skipping execution.")
            return {
                "entries": entries,
                "results": [],
                "n_success": 0,
                "n_failure": 0,
            }

        # Execute (serial or parallel). `process_single` is a staticmethod and
        # `label_enum` is a class, so the partial is pickleable for worker processes.
        worker = partial(
            self.process_single,
            verbose=self._verbose,
            label_enum=self._label_enum,
        )
        results = execute(
            worker,
            entries,
            num_workers=self._num_workers,
            logger=self._logger,
        )

        # Final stats
        n_success = sum(1 for r in results if r.get("status") == "SUCCESS")
        n_failure = len(results) - n_success
        self._logger.info(
            "Workflow complete: %d succeeded, %d failed.", n_success, n_failure
        )

        return {
            "entries": entries,
            "results": results,
            "n_success": n_success,
            "n_failure": n_failure,
        }

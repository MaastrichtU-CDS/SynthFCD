"""
Tests for ``SimpleFCDWorkflow`` (file discovery, staging, execution, IO).

The simulation pipeline is covered in ``tests/test_pipelines_simple_fcd.py``. Here the
simulator is replaced by a lightweight fake so workflow plumbing stays fast and
deterministic.
"""

from __future__ import annotations

import json
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Any

import ants
import numpy as np
import pytest

from synthfcd.utils.seg_labels import SynthSegLabel
from synthfcd.workflows.simple_fcd import SimpleFCDWorkflow
from synthfcd.workflows.utils import execute as _real_execute
from synthfcd.workflows.utils import flatten_params
from tests.utils import make_nested_seg_mask

SHAPE = (16, 16, 16)
SPACING = (1.0, 1.0, 1.0)


def _write_nifti(array: np.ndarray, path: Path, spacing: tuple = SPACING) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(
        ants.from_numpy(array.astype(np.float32), spacing=spacing), str(path)
    )


def _make_subject(
    subject_dir: Path,
    name: str = "sub",
    *,
    seg_suffix: str = "_synthseg",
    modalities: tuple[tuple[str, str], ...] = (
        ("_T1w", "T1like"),
        ("_FLAIR", "T2like"),
    ),
    spacing: tuple = SPACING,
) -> dict[str, Path]:
    seg = make_nested_seg_mask(SHAPE, wm_radius=4, gm_radius=7)
    seg_path = subject_dir / f"{name}{seg_suffix}.nii.gz"
    _write_nifti(seg, seg_path, spacing=spacing)
    paths: dict[str, Path] = {"seg": seg_path}
    tissue = (seg > 0).astype(np.float32) * 100.0
    for suffix, _ in modalities:
        img_path = subject_dir / f"{name}{suffix}.nii.gz"
        _write_nifti(tissue, img_path, spacing=spacing)
        paths[suffix] = img_path
    return paths


def _fake_simulator(
    images: list[np.ndarray],
    seg_mask: np.ndarray,
    spacing: tuple[float, float, float],
    **kwargs: Any,
) -> dict[str, Any]:
    target = seg_mask > 0
    return {
        "out_images": [img + 1.0 for img in images],
        "out_seg_mask": seg_mask.copy(),
        "stats": {"n_images": len(images), "spacing": list(spacing)},
        "extras": {"out_target": target, "orig_target": target},
    }


def _pid_worker(entry: dict[str, Any]) -> dict[str, Any]:
    """
    Pickleable stand-in for ``process_single`` that reports the worker PID.
    """
    return {
        "id": entry["id"],
        "status": "SUCCESS",
        "stats": {},
        "outputs": [],
        "error": None,
        "pid": os.getpid(),
    }


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    _make_subject(root / "sub-01", name="sub-01")
    _make_subject(root / "sub-02", name="sub-02")
    return root


def _make_workflow(**overrides: Any) -> SimpleFCDWorkflow:
    kwargs: dict[str, Any] = dict(
        explorer={"patterns": "*_synthseg.nii.gz"},
        seg_mask_suffix="_synthseg",
        modality_suffixes=["_T1w", "_FLAIR"],
        modality_names=["T1like", "T2like"],
        random_seed=0,
        num_workers=1,
        save_params_csv=False,
    )
    kwargs.update(overrides)
    return SimpleFCDWorkflow(**kwargs)


class TestWorkflowInit:
    """
    Construction-time validation of ``SimpleFCDWorkflow``.
    """

    def test_requires_modalities(self) -> None:
        with pytest.raises(ValueError, match="at least one modality"):
            SimpleFCDWorkflow(explorer={"patterns": "*"})

    def test_one_of_suffix_or_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot provide only one"):
            SimpleFCDWorkflow(
                explorer={"patterns": "*"},
                modality_suffixes=["_T1w"],
                modality_names=None,
            )

    def test_modality_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            SimpleFCDWorkflow(
                explorer={"patterns": "*"},
                modality_suffixes=["_T1w", "_FLAIR"],
                modality_names=["T1like"],
            )

    def test_bad_modality_name(self) -> None:
        with pytest.raises(ValueError):
            SimpleFCDWorkflow(
                explorer={"patterns": "*"},
                modality_suffixes=["_T1w"],
                modality_names=["Wrong"],  # type: ignore[arg-type]
            )

    def test_bad_num_workers(self) -> None:
        with pytest.raises(ValueError, match="num_workers"):
            _make_workflow(num_workers="lots")

    @pytest.mark.parametrize("flag", ["raise_if_missing", "verbose", "save_images"])
    def test_bad_bool_flag(self, flag: str) -> None:
        with pytest.raises(TypeError):
            _make_workflow(**{flag: "yes"})

    def test_all_saves_false_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            _make_workflow(
                save_images=False,
                save_seg_mask=False,
                save_lesion_mask=False,
                save_stats=False,
                save_params_csv=False,
            )

    def test_bad_suffix_type(self) -> None:
        with pytest.raises(TypeError):
            _make_workflow(out_image_suffix=123)

    def test_invalid_explorer_keys(self) -> None:
        with pytest.raises(ValueError, match="Invalid keys"):
            _make_workflow(explorer={"bad_key": "x"})

    def test_label_enum_validation(self) -> None:
        with pytest.raises(TypeError):
            _make_workflow(label_enum=object)


class TestWorkflowRun:
    """
    End-to-end ``run``: staging, parameter drawing, execution, and outputs.
    """

    def test_dry_run_stages_and_writes_csv(
        self, data_root: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        wf = _make_workflow(out_dir=str(out_dir), save_params_csv=True)
        summary = wf.run(data_root, dry_run=True)
        assert len(summary["entries"]) == 2
        assert summary["results"] == []
        csv = out_dir / "synthfcd_params.csv"
        assert csv.exists()
        assert not (out_dir / "synthfcd_params.csv.csv").exists()
        import pandas as pd

        assert "FCD_type" in pd.read_csv(csv).columns
        e = summary["entries"][0]
        assert set(e) == {"id", "anchor", "seg_mask", "images", "out_paths", "params"}
        assert e["params"]["FCD_type"] in (
            "FCD_type_Ia",
            "FCD_type_Ib",
            "FCD_type_Ic",
            "FCD_type_IIa",
            "FCD_type_IIb",
        )

    def test_param_presets_forwarded_to_draw(
        self, data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def _fake_draw(
            *,
            sequences: list[str],
            presets: str,
            random_state=None,
            label_enum=None,
            **kw: Any,
        ) -> dict[str, Any]:
            captured["presets"] = presets
            captured["sequences"] = sequences
            return {
                "FCD_type": "FCD_type_IIa",
                "growth_params": {},
                "deformation_params": {},
                "intensity_params": [{} for _ in sequences],
            }

        monkeypatch.setattr(
            "synthfcd.workflows.simple_fcd.draw_simple_fcd_params", _fake_draw
        )
        wf = _make_workflow(param_presets="minimal")
        wf.run(data_root, dry_run=True)
        assert captured["presets"] == "minimal"
        assert captured["sequences"] == ["T1like", "T2like"]

    def test_fcd_type_not_passed_to_simulator(
        self, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        received: dict[str, Any] = {}

        def _capturing_sim(
            images: list[np.ndarray],
            seg_mask: np.ndarray,
            spacing: tuple[float, float, float],
            **kwargs: Any,
        ) -> dict[str, Any]:
            received.update(kwargs)
            return _fake_simulator(images, seg_mask, spacing, **kwargs)

        monkeypatch.setattr(
            "synthfcd.workflows.simple_fcd.simple_fcd_simulator", _capturing_sim
        )
        wf = _make_workflow(out_dir=str(tmp_path / "out"))
        summary = wf.run(data_root)
        assert "FCD_type" not in received
        assert {"growth_params", "deformation_params", "intensity_params"} <= set(
            received
        )
        assert "FCD_type" in summary["entries"][0]["params"]

    def test_serial_run_writes_outputs(
        self, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.workflows.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        out_dir = tmp_path / "out"
        wf = _make_workflow(out_dir=str(out_dir), save_params_csv=True)
        summary = wf.run(data_root)
        assert summary["n_success"] == 2
        assert summary["n_failure"] == 0
        for r in summary["results"]:
            assert r["status"] == "SUCCESS"
            assert len(r["outputs"]) == 5  # 2 images + seg + lesion + stats
            for p in r["outputs"]:
                assert Path(p).exists()
        t1 = out_dir / "sub-01" / "sub-01_T1w_synthfcd.nii.gz"
        assert t1.exists()
        csv = out_dir / "synthfcd_params.csv"
        assert csv.exists()
        with open(out_dir / "sub-01" / "sub-01_synthseg_synthfcd_stats.json") as fh:
            assert json.load(fh)["n_images"] == 2

    def test_only_enabled_outputs_written(
        self, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.workflows.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        out_dir = tmp_path / "out"
        wf = _make_workflow(
            out_dir=str(out_dir),
            save_images=False,
            save_stats=False,
            save_params_csv=False,
        )
        summary = wf.run(data_root)
        sub = out_dir / "sub-01"
        written = {Path(p).name for p in summary["results"][0]["outputs"]}
        assert written == {
            "sub-01_synthseg_synthfcd_seg.nii.gz",
            "sub-01_synthseg_synthfcd_lesion_seg.nii.gz",
        }
        assert (sub / "sub-01_synthseg_synthfcd_seg.nii.gz").exists()
        assert (sub / "sub-01_synthseg_synthfcd_lesion_seg.nii.gz").exists()
        assert not (sub / "sub-01_T1w_synthfcd.nii.gz").exists()
        assert not (sub / "sub-01_FLAIR_synthfcd.nii.gz").exists()
        assert not (sub / "sub-01_synthseg_synthfcd_stats.json").exists()
        assert not (out_dir / "synthfcd_params.csv").exists()

    def test_writes_next_to_source_without_out_dir(
        self, data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.workflows.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        wf = _make_workflow(out_dir=None, save_params_csv=False)
        summary = wf.run(data_root)
        assert summary["n_success"] == 2
        assert (data_root / "sub-01" / "sub-01_T1w_synthfcd.nii.gz").exists()

    def test_file_input_ignores_explorer(
        self, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.workflows.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        anchor = data_root / "sub-01" / "sub-01_synthseg.nii.gz"
        wf = _make_workflow(explorer=None, out_dir=str(tmp_path / "out"))
        summary = wf.run(anchor)
        assert len(summary["entries"]) == 1
        assert summary["n_success"] == 1

    def test_directory_without_explorer_raises(self, data_root: Path) -> None:
        wf = _make_workflow(explorer=None)
        with pytest.raises(ValueError, match="explorer is not configured"):
            wf.run(data_root)

    def test_nonexistent_input_raises(self, tmp_path: Path) -> None:
        wf = _make_workflow()
        with pytest.raises(FileNotFoundError):
            wf.run(tmp_path / "nope")

    def test_run_skips_missing_subject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.workflows.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        root = tmp_path / "data"
        _make_subject(root / "good", name="good")
        _make_subject(root / "bad", name="bad", modalities=(("_T1w", "T1like"),))
        wf = _make_workflow(
            out_dir=str(tmp_path / "out"),
            raise_if_missing=False,
            save_params_csv=False,
        )
        summary = wf.run(root)
        assert len(summary["entries"]) == 1
        assert summary["n_success"] == 1

    def test_no_inputs_discovered(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        wf = _make_workflow()
        summary = wf.run(empty)
        assert summary == {"entries": [], "results": [], "n_success": 0, "n_failure": 0}

    def test_simulator_failure_is_captured(
        self, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("sim failed")

        monkeypatch.setattr("synthfcd.workflows.simple_fcd.simple_fcd_simulator", _boom)
        wf = _make_workflow(out_dir=str(tmp_path / "out"))
        summary = wf.run(data_root)
        assert summary["n_success"] == 0
        assert summary["n_failure"] == 2
        assert all(r["status"] == "FAILURE" for r in summary["results"])
        assert all("sim failed" in r["error"] for r in summary["results"])

    def test_determinism_same_seed(self, data_root: Path) -> None:
        e1 = _make_workflow(random_seed=123).run(data_root, dry_run=True)["entries"]
        e2 = _make_workflow(random_seed=123).run(data_root, dry_run=True)["entries"]
        for a, b in zip(e1, e2, strict=True):
            assert a["id"] == b["id"]
            assert flatten_params(a["params"]) == flatten_params(b["params"])

    def test_different_seed_changes_params(self, data_root: Path) -> None:
        e1 = _make_workflow(random_seed=1).run(data_root, dry_run=True)["entries"]
        e2 = _make_workflow(random_seed=2).run(data_root, dry_run=True)["entries"]
        assert flatten_params(e1[0]["params"]) != flatten_params(e2[0]["params"])

    def test_no_seed_changes_params(self, data_root: Path) -> None:
        e1 = _make_workflow(random_seed=None).run(data_root, dry_run=True)["entries"]
        e2 = _make_workflow(random_seed=None).run(data_root, dry_run=True)["entries"]
        assert flatten_params(e1[0]["params"]) != flatten_params(e2[0]["params"])

    def test_num_workers_respected(
        self, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concurrent.futures import ProcessPoolExecutor as RealPool

        seen: dict[str, Any] = {}

        def wrapping_execute(
            fn: Any,
            items: Any,
            *,
            num_workers: int | str = 1,
            logger: Any = None,
        ) -> list[dict[str, Any]]:
            seen["requested"] = num_workers
            seen["pool_max_workers"] = None
            results = _real_execute(
                _pid_worker, items, num_workers=num_workers, logger=logger  # type: ignore[arg-type]
            )
            seen["pids"] = {r["pid"] for r in results}
            return results

        class _RecordingPool(RealPool):
            def __init__(
                self, *args: Any, max_workers: int | None = None, **kwargs: Any
            ) -> None:
                seen["pool_max_workers"] = max_workers
                super().__init__(*args, max_workers=max_workers, **kwargs)

        monkeypatch.setattr("synthfcd.workflows.simple_fcd.execute", wrapping_execute)
        monkeypatch.setattr(
            "synthfcd.workflows.utils.ProcessPoolExecutor", _RecordingPool
        )
        out_dir = str(tmp_path / "out")

        _make_workflow(out_dir=out_dir, num_workers=1).run(data_root)
        assert seen["requested"] == 1
        assert seen["pool_max_workers"] is None
        assert seen["pids"] == {os.getpid()}

        _make_workflow(out_dir=out_dir, num_workers=2).run(data_root)
        assert seen["requested"] == 2
        assert seen["pool_max_workers"] == 2
        assert len(seen["pids"]) == 2
        assert os.getpid() not in seen["pids"]


class TestParallelPicklability:
    """
    Parallel execution requires pickleable workers and staged entries.
    """

    def test_worker_partial_is_pickleable(self) -> None:
        worker = partial(
            SimpleFCDWorkflow.process_single, verbose=False, label_enum=SynthSegLabel
        )
        restored = pickle.loads(pickle.dumps(worker))
        assert restored.keywords["label_enum"] is SynthSegLabel

    def test_staged_entry_is_pickleable(self, data_root: Path) -> None:
        wf = _make_workflow()
        entries = wf.run(data_root, dry_run=True)["entries"]
        restored = pickle.loads(pickle.dumps(entries[0]))
        assert restored["id"] == entries[0]["id"]

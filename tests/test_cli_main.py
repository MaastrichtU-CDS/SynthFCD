"""
Tests for the ``synthfcd`` CLI entry point (``synthfcd.cli.main``).

Parsing and ``main`` error handling are unit-tested here. End-to-end runs reuse the
existing CLI command coverage, with the heavy simulator replaced by a fast fake so the
tests exercise CLI plumbing rather than the simulation itself.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import ants
import numpy as np
import pytest

from synthfcd.cli import build_parser as exported_build_parser
from synthfcd.cli import main as exported_main
from synthfcd.cli.main import build_parser, main
from tests.utils import write_config

# ``synthfcd.cli.main`` is the re-exported ``main`` function, so the module has to
# be loaded by name rather than attribute lookup on the package.
_main_mod = import_module("synthfcd.cli.main")

SHAPE = (16, 16, 16)
SPACING = (1.0, 1.0, 1.0)


def _write_nifti(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(
        ants.from_numpy(array.astype(np.float32), spacing=SPACING), str(path)
    )


def _make_seg(shape: tuple[int, int, int] = SHAPE) -> np.ndarray:
    seg = np.zeros(shape, dtype=np.int32)
    seg[4:12, 4:12, 4:8] = 2
    seg[4:12, 4:12, 8:12] = 1006
    return seg


def _make_subject(subject_dir: Path, name: str) -> None:
    _write_nifti(_make_seg(), subject_dir / f"{name}_synthseg.nii.gz")
    img = (_make_seg() > 0).astype(np.float32) * 100.0
    _write_nifti(img, subject_dir / f"{name}_T1w.nii.gz")
    _write_nifti(img, subject_dir / f"{name}_FLAIR.nii.gz")


def base_config(data_root: Path, out_dir: Path) -> dict[str, Any]:
    """
    A valid simple_fcd config for the given data root and output dir.
    """
    return {
        "workflow": "simple_fcd",
        "settings": {
            "explorer": {"patterns": "*_synthseg.nii.gz"},
            "seg_mask_suffix": "_synthseg",
            "modality_suffixes": ["_T1w", "_FLAIR"],
            "modality_names": ["T1like", "T2like"],
            "out_dir": str(out_dir),
            "num_workers": 1,
            "random_seed": 0,
            "save_params_csv": True,
        },
        "inputs": str(data_root),
    }


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    _make_subject(root / "sub-01", name="sub-01")
    _make_subject(root / "sub-02", name="sub-02")
    return root


@pytest.fixture
def fake_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(images, seg_mask, spacing, **kwargs):
        target = seg_mask == 1006
        return {
            "out_images": [img + 1.0 for img in images],
            "out_seg_mask": seg_mask.copy(),
            "stats": {"n_images": len(images)},
            "extras": {"out_target": target, "orig_target": target},
        }

    monkeypatch.setattr("synthfcd.workflows.simple_fcd.simple_fcd_simulator", _fake)


# ----------------------------------------------------------------#
# Argument parser
# ----------------------------------------------------------------#
class TestBuildParser:
    def test_package_reexports_parser(self) -> None:
        assert exported_build_parser is build_parser
        assert exported_main is main

    def test_prog_name(self) -> None:
        assert build_parser().prog == "synthfcd"

    def test_parses_positional_config_and_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cfg.yaml", "--dry-run"])
        assert args.config == "cfg.yaml"
        assert args.dry_run is True

    def test_dry_run_defaults_to_false(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cfg.yaml"])
        assert args.config == "cfg.yaml"
        assert args.dry_run is False

    def test_config_is_required(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_rejects_extra_positionals(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["cfg.yaml", "extra.yaml"])

    def test_help_exits_zero(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["-h"])
        assert exc_info.value.code == 0


# ----------------------------------------------------------------#
# main() dispatch and error handling
# ----------------------------------------------------------------#
class _FakeWorkflow:
    def __init__(self, summary: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._summary = summary if summary is not None else {"n_failure": 0}

    def run(self, *, inputs: Any, dry_run: bool = False) -> dict[str, Any]:
        self.calls.append({"inputs": inputs, "dry_run": dry_run})
        return self._summary


class TestMainErrorHandling:
    def test_missing_config_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main([str(tmp_path / "missing.yaml")])
        assert code == 1
        assert "error" in capsys.readouterr().err.lower()

    def test_non_mapping_config_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        code = main([str(path)])
        assert code == 1
        assert "mapping" in capsys.readouterr().err.lower()

    def test_unknown_workflow_returns_one(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = write_config(tmp_path, {"workflow": "Nope", "inputs": "/data"})
        code = main([str(path)])
        assert code == 1
        assert "unknown workflow" in capsys.readouterr().err.lower()

    def test_invalid_settings_returns_one(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = write_config(
            tmp_path,
            {
                "workflow": "simple_fcd",
                "settings": {"not_a_real_arg": True},
                "inputs": "/data",
            },
        )
        code = main([str(path)])
        assert code == 1
        assert "invalid `settings`" in capsys.readouterr().err.lower()


class TestMainDispatch:
    def test_success_returns_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = write_config(tmp_path, {"workflow": "simple_fcd", "inputs": "/data"})
        fake = _FakeWorkflow({"n_failure": 0})
        monkeypatch.setattr(_main_mod, "create_workflow", lambda cfg: fake)

        assert main([str(path)]) == 0
        assert fake.calls == [{"inputs": "/data", "dry_run": False}]

    def test_forwards_dry_run_and_inputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = write_config(
            tmp_path, {"workflow": "simple_fcd", "inputs": ["/a", "/b"]}
        )
        fake = _FakeWorkflow()
        monkeypatch.setattr(_main_mod, "create_workflow", lambda cfg: fake)

        assert main([str(path), "--dry-run"]) == 0
        assert fake.calls == [{"inputs": ["/a", "/b"], "dry_run": True}]

    def test_failures_return_two(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = write_config(tmp_path, {"workflow": "simple_fcd", "inputs": "/data"})
        fake = _FakeWorkflow({"n_failure": 3})
        monkeypatch.setattr(_main_mod, "create_workflow", lambda cfg: fake)

        assert main([str(path)]) == 2


# ----------------------------------------------------------------#
# End-to-end CLI runs
# ----------------------------------------------------------------#
class TestWorkflowCommand:
    def test_serial_run_writes_outputs(
        self,
        data_root: Path,
        tmp_path: Path,
        fake_simulator: None,
    ) -> None:
        out_dir = tmp_path / "out"
        cfg = write_config(tmp_path, base_config(data_root, out_dir))

        code = main([str(cfg)])

        assert code == 0
        produced = list(out_dir.rglob("*synthfcd*.nii.gz"))
        assert len(produced) > 0
        assert (out_dir / "synthfcd_params.csv").exists()

    def test_dry_run_flag(
        self,
        data_root: Path,
        tmp_path: Path,
        fake_simulator: None,
    ) -> None:
        out_dir = tmp_path / "out"
        cfg = write_config(tmp_path, base_config(data_root, out_dir))

        code = main([str(cfg), "--dry-run"])

        assert code == 0
        assert list(out_dir.rglob("*synthfcd*.nii.gz")) == []

    def test_failure_yields_exit_code_two(
        self,
        data_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("simulation failed")

        monkeypatch.setattr("synthfcd.workflows.simple_fcd.simple_fcd_simulator", _boom)
        out_dir = tmp_path / "out"
        cfg = write_config(tmp_path, base_config(data_root, out_dir))

        code = main([str(cfg)])

        assert code == 2

    def test_no_inputs_discovered_succeeds(
        self,
        tmp_path: Path,
        fake_simulator: None,
    ) -> None:
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        out_dir = tmp_path / "out"
        cfg = write_config(tmp_path, base_config(empty_root, out_dir))

        code = main([str(cfg)])

        assert code == 0
        assert list(out_dir.rglob("*synthfcd*.nii.gz")) == []

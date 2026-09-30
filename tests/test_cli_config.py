"""
Tests for ``synthfcd.cli.config``: YAML loading, validation, and workflow construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synthfcd.cli.config import (
    LABEL_ENUM_REGISTRY,
    WORKFLOW_REGISTRY,
    ConfigError,
    create_workflow,
    load_workflow_config,
)
from synthfcd.utils._aliases import _WorkflowConfigType
from synthfcd.utils.seg_labels import SynthSegLabel
from synthfcd.workflows import SimpleFCDWorkflow
from tests.utils import DummyLabel, write_config


def _minimal_settings() -> dict[str, Any]:
    return {
        "modality_suffixes": ["_T1w"],
        "modality_names": ["T1like"],
    }


def _config(
    *,
    workflow: str = "simple_fcd",
    settings: dict[str, Any] | None = None,
    inputs: str | list[str] = "/data",
) -> _WorkflowConfigType:
    return {
        "workflow": workflow,
        "settings": _minimal_settings() if settings is None else settings,
        "inputs": inputs,
    }


# ----------------------------------------------------------------#
# Registries and ConfigError
# ----------------------------------------------------------------#
class TestRegistries:
    def test_workflow_registry_contains_simple_fcd(self) -> None:
        assert WORKFLOW_REGISTRY["simple_fcd"] is SimpleFCDWorkflow

    def test_label_enum_registry_contains_synthseg(self) -> None:
        assert LABEL_ENUM_REGISTRY["SynthSegLabel"] is SynthSegLabel


class TestConfigError:
    def test_is_value_error(self) -> None:
        assert issubclass(ConfigError, ValueError)


# ----------------------------------------------------------------#
# Config loading and validation
# ----------------------------------------------------------------#
class TestLoadWorkflowConfig:
    def test_valid_config(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "workflow": "simple_fcd",
                "settings": {"num_workers": 2},
                "inputs": ["/data/a", "/data/b"],
            },
        )
        config = load_workflow_config(path)
        assert config == {
            "workflow": "simple_fcd",
            "settings": {"num_workers": 2},
            "inputs": ["/data/a", "/data/b"],
        }

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"workflow": "simple_fcd", "inputs": "/data"})
        config = load_workflow_config(str(path))
        assert config["workflow"] == "simple_fcd"
        assert config["inputs"] == "/data"

    def test_string_inputs(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"workflow": "simple_fcd", "inputs": "/data"})
        config = load_workflow_config(path)
        assert config["inputs"] == "/data"

    def test_settings_default_to_empty(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"workflow": "simple_fcd", "inputs": "/data"})
        config = load_workflow_config(path)
        assert config["settings"] == {}

    def test_null_settings_become_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "workflow: simple_fcd\nsettings:\ninputs: /data\n",
            encoding="utf-8",
        )
        config = load_workflow_config(path)
        assert config["settings"] == {}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_workflow_config(tmp_path / "missing.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("workflow: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="parse YAML"):
            load_workflow_config(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping at the top level"):
            load_workflow_config(path)

    def test_non_mapping_top_level(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping at the top level"):
            load_workflow_config(path)

    def test_scalar_top_level(self, tmp_path: Path) -> None:
        path = tmp_path / "scalar.yaml"
        path.write_text("just a string\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping at the top level"):
            load_workflow_config(path)

    def test_missing_workflow_key(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"inputs": "/data"})
        with pytest.raises(ConfigError, match="`workflow`"):
            load_workflow_config(path)

    def test_workflow_not_a_string(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"workflow": 123, "inputs": "/data"})
        with pytest.raises(ConfigError, match="must be a string"):
            load_workflow_config(path)

    def test_settings_not_a_mapping(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "workflow": "simple_fcd",
                "settings": [1, 2],
                "inputs": "/data",
            },
        )
        with pytest.raises(ConfigError, match="`settings` must be a mapping"):
            load_workflow_config(path)

    def test_missing_inputs_key(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"workflow": "simple_fcd", "settings": {}})
        with pytest.raises(ConfigError, match="`inputs` key"):
            load_workflow_config(path)

    def test_inputs_wrong_type(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"workflow": "simple_fcd", "inputs": 123})
        with pytest.raises(ConfigError, match="`inputs` must be"):
            load_workflow_config(path)

    def test_inputs_list_must_be_strings(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"workflow": "simple_fcd", "inputs": [1, 2]})
        with pytest.raises(ConfigError, match="`inputs` must be"):
            load_workflow_config(path)

    def test_unknown_top_level_key(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "workflow": "simple_fcd",
                "inputs": "/data",
                "extra": 1,
            },
        )
        with pytest.raises(ConfigError, match="Unknown top-level"):
            load_workflow_config(path)

    def test_dry_run_is_not_a_config_key(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "workflow": "simple_fcd",
                "inputs": "/data",
                "dry_run": True,
            },
        )
        with pytest.raises(ConfigError, match="Unknown top-level"):
            load_workflow_config(path)

    def test_kwargs_is_not_a_config_key(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "workflow": "simple_fcd",
                "inputs": "/data",
                "kwargs": {"num_workers": 1},
            },
        )
        with pytest.raises(ConfigError, match="Unknown top-level"):
            load_workflow_config(path)


# ----------------------------------------------------------------#
# Workflow creation
# ----------------------------------------------------------------#
class TestCreateWorkflow:
    def test_creates_instance(self) -> None:
        workflow = create_workflow(_config())
        assert isinstance(workflow, SimpleFCDWorkflow)

    def test_class_name_is_unknown(self) -> None:
        with pytest.raises(ConfigError, match="Unknown workflow"):
            create_workflow(_config(workflow="SimpleFCDWorkflow"))

    def test_unknown_workflow(self) -> None:
        with pytest.raises(ConfigError, match="Unknown workflow"):
            create_workflow(_config(workflow="DoesNotExist"))

    def test_invalid_settings(self) -> None:
        with pytest.raises(ConfigError, match="Invalid `settings`"):
            create_workflow(_config(settings={"not_a_real_arg": True}))

    def test_does_not_mutate_config_settings(self) -> None:
        settings: dict[str, Any] = {
            **_minimal_settings(),
            "label_enum": "SynthSegLabel",
        }
        config = _config(settings=settings)
        create_workflow(config)
        assert config["settings"]["label_enum"] == "SynthSegLabel"

    def test_resolves_label_enum_name(self) -> None:
        workflow = create_workflow(
            _config(settings={**_minimal_settings(), "label_enum": "SynthSegLabel"})
        )
        assert workflow._label_enum is SynthSegLabel

    def test_unknown_label_enum_name(self) -> None:
        with pytest.raises(ConfigError, match="Unknown label_enum"):
            create_workflow(
                _config(settings={**_minimal_settings(), "label_enum": "NopeLabel"})
            )

    def test_unknown_label_enum_type(self) -> None:
        with pytest.raises(ConfigError, match="Unknown label_enum"):
            create_workflow(
                _config(settings={**_minimal_settings(), "label_enum": 123})
            )

    def test_accepts_label_enum_class(self) -> None:
        workflow = create_workflow(
            _config(settings={**_minimal_settings(), "label_enum": SynthSegLabel})
        )
        assert workflow._label_enum is SynthSegLabel

    def test_accepts_custom_label_enum_class(self) -> None:
        workflow = create_workflow(
            _config(settings={**_minimal_settings(), "label_enum": DummyLabel})
        )
        assert workflow._label_enum is DummyLabel

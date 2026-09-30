"""
Configuration loading for the synthfcd command-line interface.

A workflow run is described by a small YAML file with three top-level keys::

    workflow: simple_fcd             # name of the workflow (see WORKFLOW_REGISTRY)
    settings:                        # constructor keyword arguments (optional)
        seg_mask_suffix: _synthseg
        modality_suffixes: [_T1w, _FLAIR]
        modality_names: [T1like, T2like]
        out_dir: /path/to/out
        num_workers: 4
    inputs: /path/to/data            # str or list of paths forwarded to run(inputs=...)

This module parses and validates that structure and exposes a registry that maps
workflow names to their classes. The ``dry_run`` toggle is intentionally not part
of the config; it is only provided via the CLI ``--dry-run`` flag.
"""

from __future__ import annotations

__all__ = [
    "LABEL_ENUM_REGISTRY",
    "WORKFLOW_REGISTRY",
    "ConfigError",
    "create_workflow",
    "load_workflow_config",
]

from pathlib import Path
from typing import Any

import yaml

from synthfcd.utils import resolve_path
from synthfcd.utils._aliases import _WorkflowConfigType
from synthfcd.utils._base import _LabelEnum
from synthfcd.utils.seg_labels import SynthSegLabel
from synthfcd.workflows import SimpleFCDWorkflow

# Registry of workflows that can be launched from the CLI, keyed by a snake_case name.
WORKFLOW_REGISTRY: dict[str, type] = {
    "simple_fcd": SimpleFCDWorkflow,
}

# Label enumerations selectable from a config by name. A YAML file can only carry
# strings, so a config's ``label_enum`` is resolved to a class via this registry.
LABEL_ENUM_REGISTRY: dict[str, type[_LabelEnum]] = {
    "SynthSegLabel": SynthSegLabel,
}


class ConfigError(ValueError):
    """
    Raised when a CLI configuration file is missing, malformed, or invalid.
    """


def load_workflow_config(path: str | Path) -> _WorkflowConfigType:
    """
    Load a YAML workflow config and return a complete ``_WorkflowConfigType``.

    Omitted or null ``settings`` become an empty mapping so the returned
    config always has every key.

    Args:
        path (str | Path):
            Path to the YAML configuration file.

    Returns:
        _WorkflowConfigType:
            The parsed configuration with ``workflow``, ``settings``, and
            ``inputs`` all present.

    Raises:
        ConfigError:
            If the file is missing, is not valid YAML, is not a mapping, is
            missing required keys, or has keys of the wrong type.
    """
    config_path = resolve_path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config must be a mapping at the top level; got {type(raw).__name__}."
        )

    if "workflow" not in raw:
        raise ConfigError("Config must define a top-level `workflow` key.")
    workflow = raw["workflow"]
    if not isinstance(workflow, str):
        raise ConfigError(
            f"`workflow` must be a string; got {type(workflow).__name__}."
        )

    if "settings" not in raw or raw["settings"] is None:
        settings: dict[str, Any] = {}
    else:
        settings = raw["settings"]
        if not isinstance(settings, dict):
            raise ConfigError(
                f"`settings` must be a mapping; got {type(settings).__name__}."
            )

    if "inputs" not in raw:
        raise ConfigError("Config must define a top-level `inputs` key.")
    inputs = raw["inputs"]
    if isinstance(inputs, str):
        parsed_inputs: str | list[str] = inputs
    elif isinstance(inputs, list) and all(isinstance(item, str) for item in inputs):
        parsed_inputs = inputs
    else:
        raise ConfigError(
            "`inputs` must be a path string or a list of paths; "
            f"got {type(inputs).__name__}."
        )

    unknown = set(raw) - {"workflow", "settings", "inputs"}
    if unknown:
        raise ConfigError(
            f"Unknown top-level config key(s): {sorted(unknown)}. "
            "Expected only `workflow`, `settings`, and `inputs`."
        )

    return {
        "workflow": workflow,
        "settings": settings,
        "inputs": parsed_inputs,
    }


def create_workflow(config: _WorkflowConfigType) -> Any:
    """
    Instantiate the workflow described by a configuration.

    Args:
        config (_WorkflowConfigType):
            The parsed configuration.

    Returns:
        Any:
            An instance of the requested workflow.

    Raises:
        ConfigError:
            If the workflow name is not registered or the constructor rejects the
            provided settings.
    """
    workflow = config["workflow"]
    if workflow not in WORKFLOW_REGISTRY:
        available = sorted(WORKFLOW_REGISTRY)
        raise ConfigError(
            f"Unknown workflow '{workflow}'. Available workflows: {available}."
        )

    workflow_cls = WORKFLOW_REGISTRY[workflow]
    settings = dict(config["settings"])
    if "label_enum" in settings:
        settings["label_enum"] = _resolve_label_enum(settings["label_enum"])

    try:
        return workflow_cls(**settings)
    except TypeError as exc:
        raise ConfigError(
            f"Invalid `settings` for workflow '{workflow}': {exc}"
        ) from exc


def _resolve_label_enum(value: Any) -> type[_LabelEnum]:
    """
    Resolve a config ``label_enum`` value to a label enumeration class.

    A config sourced from YAML provides ``label_enum`` as a registered name; an
    already-resolved ``_LabelEnum`` subclass is also accepted for programmatic use.

    Args:
        value (Any):
            The ``label_enum`` value from the config settings.

    Returns:
        type[_LabelEnum]:
            The resolved label enumeration class.

    Raises:
        ConfigError:
            If ``value`` is neither a registered name nor a ``_LabelEnum`` subclass.
    """
    if isinstance(value, type) and issubclass(value, _LabelEnum):
        return value
    if isinstance(value, str) and value in LABEL_ENUM_REGISTRY:
        return LABEL_ENUM_REGISTRY[value]
    available = sorted(LABEL_ENUM_REGISTRY)
    raise ConfigError(
        f"Unknown label_enum {value!r}. Available label enumerations: {available}."
    )

"""
Command-line entry point for synthfcd.

Usage::

    synthfcd path/to/config.yaml [--dry-run]

The CLI reads a YAML configuration (see ``synthfcd.cli.config``), builds the
requested workflow, and runs it over the inputs declared under the ``inputs``
key. The CLI currently only runs workflows; subcommands can be introduced later
if it grows.
"""

from __future__ import annotations

__all__ = ["build_parser", "main"]

import argparse
import sys
from collections.abc import Sequence

from synthfcd.cli.config import (
    ConfigError,
    create_workflow,
    load_workflow_config,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser.

    Returns:
        argparse.ArgumentParser:
            The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="synthfcd",
        description=(
            "Simulation-based focal cortical dysplasia lesion generation. Builds a "
            "workflow from a YAML configuration and runs it over the inputs declared "
            "under the `inputs` key."
        ),
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage subjects without running the simulation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the synthfcd command-line interface.

    Args:
        argv (Sequence[str] | None, optional):
            Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        int:
            Process exit code (``0`` on success, ``2`` if any subject failed,
            ``1`` for configuration errors).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config)
        workflow = create_workflow(config)
    except ConfigError as exc:
        print(f"synthfcd: error: {exc}", file=sys.stderr)
        return 1

    # ``dry_run`` is a CLI-only concern; it is never read from the config.
    summary = workflow.run(inputs=config["inputs"], dry_run=args.dry_run)
    return 2 if summary.get("n_failure") else 0


if __name__ == "__main__":
    raise SystemExit(main())

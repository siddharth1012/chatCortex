"""Console script for the registry pack: ``chatcortex-registry validate``.

``main`` is the ``[project.scripts]`` entry point. It never raises —
validation results come back as a :class:`RegistryReport` — and exits 0
only when the file is a fully valid pack, so shell checks can gate on it.
"""

import argparse
from typing import List, Optional

from .package import RegistryReport, validate_registry_file


def _render_report(report: RegistryReport) -> List[str]:
    """Human-readable lines; issue/duplicate order follows the report."""
    lines = [f"components validated: {report.component_count}"]
    for duplicate in report.duplicate_names:
        lines.append(f"duplicate component: '{duplicate}'")
    lines.extend(f"error: {issue}" for issue in report.issues)
    lines.append(f"result: {'VALID' if report.valid else 'INVALID'}")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point returning the process exit code (0 valid, 1 invalid)."""
    parser = argparse.ArgumentParser(
        prog="chatcortex-registry",
        description="Registry pack tooling for ChatCortex.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser(
        "validate",
        help="Validate a registry JSON pack and print a structured report.",
    )
    validate.add_argument("file", help="Path to the registry JSON file.")

    args = parser.parse_args(argv)
    report = validate_registry_file(args.file)

    print(f"chatcortex-registry validate {args.file}")
    for line in _render_report(report):
        print(f"  {line}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

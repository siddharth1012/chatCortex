"""Registry packs: save, validate, and round-trip registry JSON files.

The document shape is the loader's shipped one — ``{"components": [...]}``
with exactly the required per-component fields — plus nothing. A pack
written by :func:`save_registry` is guaranteed reloadable:
:meth:`CapabilityRegistry.to_dict` emits the loader's shape and the save
*proves* the round-trip by loading the file back and comparing registries.

:func:`validate_registry_file` never raises: it returns a
:class:`RegistryReport` with a component count, one actionable issue per
failing row (message names row and field, loader style), and every
duplicate name — so a broken pack can be inspected offline, e.g. through
the ``chatcortex-registry validate`` console command.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Union

from .capability_registry import CapabilityRegistry
from .loader import load_registry
from .metadata import validate_json_payload


def save_registry(
    registry: CapabilityRegistry,
    path: Union[str, Path],
) -> Path:
    """Write ``registry`` to ``path`` in the loader's JSON document shape.

    The file gets the exact 0.x shipped shape — ``{"components": [...]}``
    from :meth:`CapabilityRegistry.to_dict`, no additional fields — and
    the save validates its own round-trip: the freshly written file is
    loaded back and must reproduce the same component sequence, or a
    ``RuntimeError`` is raised. Every detail a ``load_registry`` call
    would consume is verifiably present in the file.

    Args:
        registry: Populated registry to package.
        path: Destination file path (parent directory must exist).

    Returns:
        The ``Path`` written, for chaining.

    Raises:
        RuntimeError: If the written file does not round-trip to an
            equivalent registry.
    """
    file_path = Path(path)
    document = registry.to_dict()
    file_path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )

    reloaded = load_registry(file_path)
    if reloaded.list_all() != registry.list_all():  # frozen dataclass equality
        raise RuntimeError(
            f"save_registry round-trip check failed for {file_path}: "
            "the written file does not reload to the same registry"
        )
    return file_path


@dataclass
class RegistryReport:
    """Structured result of validating a registry pack file.

    Attributes:
        component_count: Rows that pass full validation (equals the
            loaded component count when the pack is valid).
        issues: One message per failing row or document-level problem,
            in loader style so row and field are named, e.g.
            ``components[1].reliability_score out of range``.
        duplicate_names: Component names appearing in more than one row,
            each reported once (first-occurrence order).
    """

    component_count: int = 0
    issues: List[str] = field(default_factory=list)
    duplicate_names: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True when the file reloads into a complete, duplicate-free registry."""
        return not self.issues and not self.duplicate_names

    def to_dict(self) -> dict:
        return {
            "component_count": self.component_count,
            "issues": list(self.issues),
            "duplicate_names": list(self.duplicate_names),
            "valid": self.valid,
        }


def validate_registry_file(path: Union[str, Path]) -> RegistryReport:
    """Validate a registry pack and report what is wrong, without raising.

    Unlike :func:`load_registry` — which fails fast on the first row —
    this walks every row, collecting one issue per failing row and *all*
    duplicate names, so a corrupt pack can be repaired in one pass.

    Args:
        path: Path to the JSON document described in
            :mod:`chatcortex.registry.loader`.

    Returns:
        A :class:`RegistryReport`; ``report.valid`` is True exactly when
        :func:`load_registry` would succeed on the same file.
    """
    file_path = Path(path)
    report = RegistryReport()

    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        report.issues.append(f"Cannot read registry file {file_path}: {exc}")
        return report

    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        report.issues.append(f"Registry file {file_path} is not valid JSON: {exc}")
        return report

    if not isinstance(document, dict) or not isinstance(document.get("components"), list):
        report.issues.append(
            f"Registry file {file_path} must contain a top-level object "
            "with a 'components' list"
        )
        return report

    rows = document["components"]
    name_occurrences: dict = {}
    loadable = True

    for row, payload in enumerate(rows):
        if not isinstance(payload, dict):
            report.issues.append(
                f"components[{row}] must be an object, {type(payload).__name__} given"
            )
            loadable = False
            continue
        try:
            metadata_kwargs = validate_json_payload(payload)
        except (TypeError, ValueError) as exc:
            report.issues.append(f"components[{row}].{exc}")
            loadable = False
            continue

        name = metadata_kwargs["name"]
        if name in name_occurrences:
            report.duplicate_names.append(name)
        else:
            name_occurrences[name] = row
            report.component_count += 1

    # load_registry also rejects duplicates; mirror that in `valid`.
    if report.duplicate_names:
        loadable = False
    if not loadable:
        report.issues.append(
            f"Registry file {file_path} is not loadable; see issues above"
        )
    return report

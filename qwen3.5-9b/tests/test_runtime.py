"""Catch dependencies hidden by an older, more permissive shared environment."""
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement


def test_pinned_runtime_matches_installed_packages():
    root = Path(__file__).resolve().parents[1]
    for line in (root / "requirements/runtime.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = version(requirement.name)
        assert installed in requirement.specifier, (
            f"{requirement.name}=={installed} does not match {requirement}"
        )

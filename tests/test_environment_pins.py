"""The pinned environment is declared in two files; keep them honest.

`pyproject.toml`'s `analysis`/`test` extras select what gets installed and
`constraints-analysis.txt` bounds the resolution, so the same version numbers
appear twice. Nothing stops them drifting apart, and the failure is silent --
whichever file you happened to read would look authoritative.

These tests also pin the structural claim behind the extras: a constraints file
only bounds packages something else selected, so every direct dependency has to
be *requested* somewhere. matplotlib was constrained but never requested, so
the documented install produced an environment with no matplotlib in it.

Pure metadata checks; no imports of the pinned packages, no data needed.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CONSTRAINTS = REPO_ROOT / "constraints-analysis.txt"

PIN = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([^\s;#]+)")


def normalise(name: str) -> str:
    """PEP 503 names: scikit_image, scikit-image and SciKit-Image are one."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> str:
    return normalise(re.split(r"[\[<>=!~;\s]", spec, maxsplit=1)[0])


def parse_constraints() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in CONSTRAINTS.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = PIN.match(line)
        if match:
            pins[normalise(match.group(1))] = match.group(2)
    return pins


@unittest.skipIf(sys.version_info < (3, 11), "tomllib needs 3.11")
class EnvironmentPinTests(unittest.TestCase):
    def setUp(self) -> None:
        import tomllib

        with PYPROJECT.open("rb") as handle:
            self.pyproject = tomllib.load(handle)
        self.project = self.pyproject["project"]
        self.extras = self.project.get("optional-dependencies", {})
        self.constraints = parse_constraints()

    def extra_pins(self, *names: str) -> dict[str, str]:
        pins: dict[str, str] = {}
        for name in names:
            for spec in self.extras.get(name, []):
                match = PIN.match(spec)
                if match:
                    pins[normalise(match.group(1))] = match.group(2)
        return pins

    def test_every_runtime_dependency_is_requested_by_the_analysis_extra(self) -> None:
        """Constraining a package nobody asked for installs nothing. The base
        dependencies are unpinned, so the extra is what makes the documented
        install reproduce anything at all."""
        base = {requirement_name(spec) for spec in self.project["dependencies"]}
        requested = set(self.extra_pins("analysis"))
        missing = sorted(base - requested)
        self.assertEqual(
            missing, [],
            f"declared as runtime dependencies but not pinned by the analysis "
            f"extra, so `pip install -e '.[analysis,test]'` resolves them to "
            f"whatever is current: {missing}",
        )

    def test_extras_and_constraints_agree(self) -> None:
        declared = self.extra_pins("analysis", "test")
        disagreements = {
            name: (version, self.constraints[name])
            for name, version in declared.items()
            if name in self.constraints and self.constraints[name] != version
        }
        self.assertEqual(
            disagreements, {},
            "pyproject.toml and constraints-analysis.txt disagree "
            "(package: pyproject, constraints)",
        )

    def test_every_pinned_extra_appears_in_the_constraints_file(self) -> None:
        declared = set(self.extra_pins("analysis", "test"))
        missing = sorted(declared - set(self.constraints))
        self.assertEqual(
            missing, [],
            f"pinned in pyproject.toml but absent from constraints-analysis.txt, "
            f"which is documented as the record of the analysis environment: "
            f"{missing}",
        )

    def test_constraints_file_pins_exactly(self) -> None:
        """A range here would defeat the point of the file."""
        for line in CONSTRAINTS.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            self.assertRegex(
                stripped, PIN,
                f"constraints-analysis.txt must pin with '==': {stripped!r}",
            )


if __name__ == "__main__":
    unittest.main()

"""Check that the version in pyproject.toml is higher than the latest release on PyPI.

Merging into devel-tiny triggers a PyPI release (see pypi-release.yml), which fails
if the version number has not been bumped.
"""

import json
import sys
import tomllib
import urllib.request
from pathlib import Path

from packaging.version import Version

PACKAGE = "feectools"


def local_version(pyproject: Path) -> Version:
    with pyproject.open("rb") as f:
        return Version(tomllib.load(f)["project"]["version"])


def latest_pypi_version(package: str) -> Version:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=30) as response:
        releases = json.load(response)["releases"]
    return max(Version(v) for v, files in releases.items() if files)


def main() -> int:
    pyproject = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pyproject.toml")
    local = local_version(pyproject)
    pypi = latest_pypi_version(PACKAGE)
    print(f"Version in {pyproject}: {local}")
    print(f"Latest {PACKAGE} version on PyPI: {pypi}")

    if local <= pypi:
        print(
            f"::error file={pyproject}::Version {local} must be higher than the latest PyPI release {pypi}. "
            "Bump the version in pyproject.toml before merging into devel-tiny."
        )
        return 1

    print("OK: version is higher than the latest PyPI release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

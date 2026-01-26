#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
def extract_version() -> str:
    """
    Returns either the version of the installed package (e.g. "0.17.1") or the
    one found in pyproject.toml (e.g. "0.17.1-dev (at /project/location)").

    Returns
    -------
    str
        The package version.

    See also
    --------
    https://stackoverflow.com/a/76206192

    """
    from contextlib import suppress
    from pathlib    import Path

    with suppress(FileNotFoundError, StopIteration):
        root_dir = Path(__file__).parent.parent
        with open(root_dir / "pyproject.toml", encoding="utf-8") as pyproject_toml:
            version_line = next(line for line in pyproject_toml if line.startswith("version"))
            version = version_line.split("=")[1].strip("'\"\n ")
            return f"{version}-dev (at {root_dir})"

    import importlib.metadata
    return importlib.metadata.version(__package__)


__version__ = extract_version()

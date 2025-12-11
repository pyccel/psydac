import logging
import os
import sys
from shutil import which
from subprocess import PIPE, STDOUT  # nosec B404
from subprocess import run as sub_run
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyCommand(build_py):
    """Custom build command to pyccelise _kernels files in the build directory."""

    # Rewrite the build_module function to copy each module in the build
    # repository and pyccelise the modules ending with _kernels
    def build_module(self, module, module_file, package):
        outfile, copied = super().build_module(module, module_file, package)

        # This part check if the module is pyccelisable and pyccelise it in
        # case
        if module.endswith('_kernels'):
            self.announce(f"\nPyccelising [{module}] ...", level=logging.INFO)
            pyccel = sub_run([which('pyccel'), 'compile', outfile, '--language', 'fortran', '--openmp'],
                              stdout=PIPE, stderr=STDOUT,
                              text=True, shell=False, check=True) # nosec B603
            self.announce(pyccel.stdout, level=logging.INFO)

        return outfile, copied

    def run(self):
        super().run()

        # Remove __pyccel__ directories
        sub_run([which('pyccel'), 'clean', self.build_lib], shell=False, check=True) # nosec B603, B607

        # Remove useless .lock files
        for path, subdirs, files in os.walk(self.build_lib):
            for name in files:
                if name == '.lock_acquisition.lock':
                    os.remove(os.path.join(path, name))


#==============================================================================
# Workaround to make requirements with URLs compatible with PyPI
# See https://github.com/biobuddies/helicopyter/pull/43
#==============================================================================
from io import StringIO
from typing import TextIO
import setuptools
from packaging.metadata import Metadata
from setuptools._core_metadata import _write_requirements

def write_pypi_compatible_requirements(self: Metadata, final_file: TextIO) -> None:
    """Mark requirements with URLs as external."""
    initial_file = StringIO()
    _write_requirements(self, initial_file)
    initial_file.seek(0)
    for initial_line in initial_file:
        final_line = initial_line
        metadata = Metadata.from_email(initial_line, validate=False)
        if metadata.requires_dist and metadata.requires_dist[0].url:
            final_line = initial_line.replace('Requires-Dist:', 'Requires-External:')
        final_file.write(final_line)

setuptools._core_metadata._write_requirements = write_pypi_compatible_requirements

#==============================================================================
def get_version():
    """ Get the package version from psydac/version.py """
    sys.path.insert(0, 'psydac')
    from version import __version__
    return __version__

#==============================================================================
setup(
    version = get_version(),
    cmdclass={
        'build_py': BuildPyCommand,
    },
#    install_requires=[
#       'igakit @ https://github.com/dalcinl/igakit/archive/refs/heads/master.zip',
#    ]
)

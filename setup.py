import logging
import os
import shutil
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
            self.announce(f"\nPyccelise module: {module}", level=logging.INFO)
            sub_run([shutil.which('pyccel'), outfile,
                     '--language', 'fortran',
                     '--openmp'],
                    shell=False, check=True) # nosec B603

        return outfile, copied

    def run(self):
        super().run()

        # Remove __pyccel__ directories
        sub_run(['pyccel-clean', self.build_lib], shell=False, check=True) # nosec B603, B607

        # Remove useless .lock files
        for path, subdirs, files in os.walk(self.build_lib):
            for name in files:
                if name == '.lock_acquisition.lock':
                    os.remove(os.path.join(path, name))


setup(
    cmdclass={
        'build_py': BuildPyCommand,
    },
)

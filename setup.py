import setuptools.command.build_py
import distutils.command.build_py as orig
import distutils.log
import setuptools
from subprocess import run as sub_run
import shutil
import os


class BuildPyCommand(setuptools.command.build_py.build_py):
    """Custom build command to pyccelise _kernels files in the build directory."""

    # Copy the setuptools.command.build_py.build_py.finalize_options to recreate the __updated_files variable
    def finalize_options(self):
        orig.build_py.finalize_options(self)
        self.package_data = self.distribution.package_data
        self.exclude_package_data = self.distribution.exclude_package_data or {}
        if 'data_files' in self.__dict__:
            del self.__dict__['data_files']
        self.__updated_files = []

    # Rewrite the build_module function to copy each module in the build repository and pyccelise the modules ending with _kernels
    def build_module(self, module, module_file, package):
        # This part is copied from distutils.command.build_py.build_module
        if isinstance(package, str):
            package = package.split('.')
        elif not isinstance(package, (list, tuple)):
            raise TypeError(
                "'package' must be a string (dot-separated), list, or tuple"
            )
        outfile = self.get_module_outfile(self.build_lib, package, module)
        dirname = os.path.dirname(outfile)
        self.mkpath(dirname)
        outfile, copied = self.copy_file(module_file, outfile, preserve_mode=0)


        # This part check if the module is pyccelisable and pyccelise it in case
        if module.endswith('_kernels'):
            self.announce(
                '\nPyccelise module: %s' % str(module),
                level=distutils.log.INFO)
            sub_run([shutil.which('pyccel'), outfile, '--language', 'fortran', '--openmp'], shell=False)

        # This part is copy from setuptools.command.build_py.build_module
        if copied:
            self.__updated_files.append(outfile)

        return outfile, copied

    def run(self):
        setuptools.command.build_py.build_py.run(self)

        # Remove __pyccel__ directories
        sub_run(['pyccel-clean', self.build_lib], shell=False)

        # Remove useless .lock files
        for path, subdirs, files in os.walk(self.build_lib):
            for name in files:
                if name == '.lock_acquisition.lock':
                    os.remove(os.path.join(path, name))


setuptools.setup(
    cmdclass={
        'build_py': BuildPyCommand,
    },
)

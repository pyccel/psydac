import setuptools.command.build_py
import distutils.command.build_py as orig
import setuptools
import distutils.log
import distutils.cmd
import os

class BuildPyCommand(setuptools.command.build_py.build_py):
    """Custom build command."""

    def finalize_options(self):
        orig.build_py.finalize_options(self)
        self.package_data = self.distribution.package_data
        self.exclude_package_data = self.distribution.exclude_package_data or {}
        if 'data_files' in self.__dict__:
            del self.__dict__['data_files']
        self.__updated_files = []

    def build_module(self, module, module_file, package):
        if isinstance(package, str):
            package = package.split('.')
        elif not isinstance(package, (list, tuple)):
            raise TypeError(
                "'package' must be a string (dot-separated), list, or tuple"
            )

            # Now put the module source file into the "build" area -- this is
            # easy, we just copy it somewhere under self.build_lib (the build
            # directory for Python source).
        outfile = self.get_module_outfile(self.build_lib, package, module)
        dir = os.path.dirname(outfile)
        self.mkpath(dir)
        outfile, copied = self.copy_file(module_file, outfile, preserve_mode=0)

        if module.endswith('_kernels'):
            command ='pyccel ' +outfile
            os.system(command)

        if copied:
            self.__updated_files.append(outfile)

        return outfile, copied
    def run(self):
        setuptools.command.build_py.build_py.run(self)


setuptools.setup(
    cmdclass={
        'build_py': BuildPyCommand,
    },
)

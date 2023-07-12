import setuptools.command.build_py
import distutils.command.build_py as orig
import distutils.log
import setuptools
from subprocess import run as sub_run
import shutil
import os

# OLD VERSION ON HOW TO PYCCELYSE KERNELS FILES AT INSTALLATION
#
# class PyccelCommand(distutils.cmd.Command):
#   """A custom command to run Pyccel on all kernels source files."""
#
#   description = 'run Pyccel on kernels source files'
#   user_options = [
#       # The format is (long option, short option, description).
#       ('language=', None, 'language to pyccelise the kernels files'),
#   ]
#
#   def initialize_options(self):
#     """Set default values for options."""
#     # Each user option must be listed here with their default value.
#     self.language = ''
#
#   def finalize_options(self):
#     """Post-process options."""
#     if self.language:
#       assert self.language in ['fortran', 'c'], (
#           'Language is of the good format' % self.language)
#
#   def run(self):
#     """Run command."""
#
#     psydac_path = os.getcwd()
#
#     if self.language:
#         language_param = '--language ' +  self.language
#     else:
#         language_param = '--language fortran'
#
#     command1 = 'pyccel ' + psydac_path + '/psydac/linalg/stencil2coo_kernels.py '         + language_param
#     command2 = 'pyccel ' + psydac_path + '/psydac/api/ast/transpose_kernels.py ' + language_param
#     command3 = 'pyccel ' + psydac_path + '/psydac/core/field_evaluation_kernels.py '           + language_param
#     command4 = 'pyccel ' + psydac_path + '/psydac/core/bsplines_kernels.py '   + language_param
#
#     self.announce(
#         'Running commands: %s' % str(command1),
#         level=distutils.log.INFO)
#     os.system(command1)
#     self.announce(
#         'Running commands: %s' % str(command2),
#         level=distutils.log.INFO)
#     os.system(command2)
#     self.announce(
#         'Running commands: %s' % str(command3),
#         level=distutils.log.INFO)
#     os.system(command3)
#     self.announce(
#         'Running commands: %s' % str(command4),
#         level=distutils.log.INFO)
#     os.system(command4)
#
#
# class BuildPyCommand(setuptools.command.build_py.build_py):
#   """Custom build command."""
#
#   def run(self):
#     self.run_command('pyccel')
#     setuptools.command.build_py.build_py.run(self)
#
#
# setuptools.setup(
#     cmdclass={
#         'pyccel': PyccelCommand,
#         'build_py': BuildPyCommand,
#     },
#     # Usual setup() args.
#     # ...
# )

# NEW VERSION ON HOW TO PYCCELYSE KERNELS FILES AT INSTALLATION
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
        dir = os.path.dirname(outfile)
        self.mkpath(dir)
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


setuptools.setup(
    cmdclass={
        'build_py': BuildPyCommand,
    },
)

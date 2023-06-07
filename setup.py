import setuptools.command.build_py
import distutils.cmd
import distutils.log
import setuptools
import subprocess
import os


class PyccelCommand(distutils.cmd.Command):
  """A custom command to run Pyccel on all kernels source files."""

  description = 'run Pyccel on kernels source files'
  user_options = [
      # The format is (long option, short option, description).
      ('language=', None, 'language to pyccelise the kernels files'),
  ]

  def initialize_options(self):
    """Set default values for options."""
    # Each user option must be listed here with their default value.
    self.language = ''

  def finalize_options(self):
    """Post-process options."""
    if self.language:
      assert self.language in ['fortran', 'c'], (
          'Language is of the good format' % self.language)

  def run(self):
    """Run command."""

    psydac_path = os.getcwd()

    if self.language:
        language_param = '--language ' +  self.language
    else:
        language_param = '--language fortran'

    command1 = 'pyccel ' + psydac_path + '/psydac/linalg/kernels.py '         + language_param
    command2 = 'pyccel ' + psydac_path + '/psydac/api/ast/linalg_kernels.py ' + language_param
    command3 = 'pyccel ' + psydac_path + '/psydac/core/kernels.py '           + language_param
    command4 = 'pyccel ' + psydac_path + '/psydac/core/bsplines_pyccel.py '   + language_param

    subprocess.run(command1, check=True, shell=True)
    subprocess.run(command2, check=True, shell=True)
    subprocess.run(command3, check=True, shell=True)
    subprocess.run(command4, check=True, shell=True)

class BuildPyCommand(setuptools.command.build_py.build_py):
  """Custom build command."""

  def run(self):
    self.run_command('pyccel')
    setuptools.command.build_py.build_py.run(self)


setuptools.setup(
    cmdclass={
        'pyccel': PyccelCommand,
        'build_py': BuildPyCommand,
    },
    # Usual setup() args.
    # ...
)

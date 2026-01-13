#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import sys

from psydac import __version__ as psydac_version
from psydac import __path__ as psydac_path

__all__ = (
    'add_help_flag',
    'add_version_flag',
)

#------------------------------------------------------------------------------
def add_help_flag(parser):
    """
    Add `-h/--help` flag to argument parser.

    Add `-h/--help` flag to argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    message = 'Show this help message and exit.'
    parser.add_argument('-h', '--help', action='help', help=message)

#------------------------------------------------------------------------------
def add_version_flag(parser):
    """
    Add `-V/--version` flag to argument parser.

    Add `-V/--version` flag to argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    version = psydac_version
    libpath = psydac_path[0]
    python  = f'python {sys.version_info.major}.{sys.version_info.minor}'
    message = f'psydac {version} from {libpath} ({python})'

    parser.add_argument('-V', '--version', action='version',
                        help='Show version and exit.', version=message)

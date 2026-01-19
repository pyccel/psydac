#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import sys
import argparse

from termcolor import colored

from psydac import __version__ as psydac_version
from psydac import __path__ as psydac_path

__all__ = (
    'add_help_flag',
    'add_version_flag',
    'exit_with_error_message',
)

#------------------------------------------------------------------------------
def add_help_flag(parser: argparse.ArgumentParser) -> None:
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
def add_version_flag(parser: argparse.ArgumentParser) -> None:
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

#------------------------------------------------------------------------------
def exit_with_error_message(msg: str) -> None:
    """
    Print a colored error message and exit with status code 2.

    Print a colored error message and exit with status code 2.

    Parameters
    ----------
    msg : str
        The error message to be printed.
    """
    err = colored('ERROR', color='magenta', attrs=['bold'])
    sep = colored(': ', color='magenta')
    msg = colored(msg, color='magenta')
    print(f'{err}{sep}{msg}')
    sys.exit(2)

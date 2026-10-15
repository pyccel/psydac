#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import argparse
import sys

from psydac.cmd.argparse_helpers import add_help_flag, add_version_flag
from psydac.cmd.psydac_compile import setup_psydac_compile_parser, psydac_compile, PSYDAC_COMPILE_DESCR
from psydac.cmd.psydac_mesh import setup_psydac_mesh_parser, psydac_mesh, PSYDAC_MESH_DESCR
from psydac.cmd.psydac_test import setup_psydac_test_parser, psydac_test, PSYDAC_TEST_DESCR

#==============================================================================
def psydac_command() -> None:
    """
    Main entry point for the `psydac` command line interface.

    Main entry point for the `psydac` command line interface.
    Parses the command line arguments and calls the appropriate sub-command.
    """
    parser = argparse.ArgumentParser(
        description = 'PSYDAC: Python Spline librarY for Differential equations with Automatic Code generation',
        add_help = False,
    )

    group = parser.add_argument_group("Options")
    add_help_flag(group)
    add_version_flag(group)

    sub_commands = {
        'compile': (setup_psydac_compile_parser, psydac_compile, PSYDAC_COMPILE_DESCR),
        'mesh'   : (setup_psydac_mesh_parser   , psydac_mesh   , PSYDAC_MESH_DESCR   ),
        'test'   : (setup_psydac_test_parser   , psydac_test   , PSYDAC_TEST_DESCR   ),
    }

    subparsers = parser.add_subparsers(
        required=True, title='Subcommands', metavar='COMMAND')

    for key, (parser_setup, exe_func, descr) in sub_commands.items():
        sparser = subparsers.add_parser(
            key,
            help = descr.splitlines()[0],
            description = f"PSYDAC's CLI: {descr}",
            formatter_class = argparse.RawDescriptionHelpFormatter,
            add_help = False,
        )
        parser_setup(sparser)
        sparser.set_defaults(func = exe_func)

    argv = sys.argv[1:]
    if len(argv) == 0:
        parser.print_help()
        sys.exit(2)

    kwargs = vars(parser.parse_args())
    func = kwargs.pop('func')
    func(**kwargs)


import argparse
import os
from subprocess import run as sub_run
import shutil

import psydac

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Accelerate all computational kernels in Psydac using Pyccel"
    )

    # Add Argument --language at the pyccel command
    parser.add_argument('--language',
                        type=str,
                        default='fortran',
                        choices=['fortran', 'c'],
                        action='store',
                        dest='language',
                        help='Language used to pyccelize all the _kernels files'
                        )

    # Add flag --openmp at the pyccel command
    parser.add_argument('--openmp',
                        default=False,
                        action='store_true',
                        dest='openmp',
                        help="Use OpenMP multithreading in generated code."
                        )

    # Read input arguments
    args = parser.parse_args()

    # get the absolute path to the psydac directory
    psydac_path = os.path.abspath(os.path.dirname(psydac.__path__[0]))

    # Define all the parameters of the command in the parameters array
    parameters = ['--language', args.language]

    # check if the flag --openmp is passed and add it to the argument if it's the case
    if args.openmp:
        parameters.append('--openmp')

    # search in psydac/psydac folder all the files ending with the tag _kernels.py
    for path, subdirs, files in os.walk(psydac_path):
        for name in files:
            if name.endswith('_kernels.py'):
                command = [shutil.which('pyccel'), os.path.join(path, name), *parameters]
                print('  Pyccelize file: ' + os.path.join(path, name))
                print(' '.join(command))
                sub_run(command, shell=False)


import argparse
import os
from subprocess import run as sub_run
import shutil

# Add Argument --language at the command psydac-accelerate
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Get language for the pyccelisation."
)

parser.add_argument('--language',
                    type=str,
                    default='fortran',
                    choices=['fortran', 'c'],
                    action='store',
                    dest='language',
                    help='Language used to pyccelise all the _kernels files'
                    )

parser.add_argument('--openmp',
                    default=False,
                    action='store_true',
                    dest='openmp',
                    help='Do we read the OpenMP instrucions'
                    )

# Read input arguments
args = parser.parse_args()

# get the absolute path to the psydac directory
psydac_path = os.path.dirname(os.path.abspath(__file__))+'/psydac'

print("\nThis command should only be used if psydac was installed in editable mode.\n")

# Define all the parameters of the command in the parameters array
parameters = ['--language', args.language]

# check if the flag --openmp is passed and add it to the argument if it's the case
if args.openmp:
    parameters.append('--openmp')

# search in psydac/psydac folder all the files ending with the tag _kernels.py
for path, subdirs, files in os.walk(psydac_path):
    for name in files:
        if name.endswith('_kernels.py'):
            print('Pyccelise file :' + os.path.join(path, name))
            sub_run([shutil.which('pyccel'), os.path.join(path, name), *parameters], shell=False)
            print('\n')



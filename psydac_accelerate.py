
import argparse
import os
from subprocess import run as sub_run
import shutil
import sysconfig
libdir = sysconfig.get_config_var('LIBDIR')

#The purpose of this file is to be launched after an editable installation of Psydac, to pyccelise all the Psydac kernels.
#This file is useless during a classic installation because the kernels are already pyccelised in the construction folder.



parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Accelerate all computational kernels in Psydac using Pyccel (editable install only)"
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
psydac_path = os.path.dirname(os.path.abspath(__file__))+'/psydac'

print("\nNOTE: This script should only be used if Psydac was installed in editable mode.\n")

# Define all the parameters of the command in the parameters array
parameters = ['--language', args.language]

# check if the flag --openmp is passed and add it to the argument if it's the case
if args.openmp:
    parameters.append('--openmp')

# Append libdir
parameters.append('--libdir')
parameters.append(libdir)

# search in psydac/psydac folder all the files ending with the tag _kernels.py
for path, subdirs, files in os.walk(psydac_path):
    for name in files:
        if name.endswith('_kernels.py'):
            print('  Pyccelize file: ' + os.path.join(path, name))
            sub_run([shutil.which('pyccel'), os.path.join(path, name), *parameters], shell=False)
print()

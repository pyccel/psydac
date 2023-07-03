def main():
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
                        action='store',
                        dest='language',
                        help='Language used to pyccelise all the _kernels files'
                        )

    # Read input arguments
    args = parser.parse_args()


    # get the absolute path to the psydac directory
    psydac_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("\nThis command should only be used if psydac was installed in editable mode.\n")

    # check if the language have the good format
    language    = args.language

    if language not in ['fortran', 'c']:
        print("\nWarning: The language given is not used by pyccel. It must be 'fortran' or 'c'. For this run, it is taken as fortran.\n")
        language = 'fortran'

    # search in psydac/psydac folder all the files ending with the tag _kernels.py
    for path, subdirs, files in os.walk(psydac_path):
        for name in files:
            if name.endswith('_kernels.py'):
                print('Pyccelise file :' + os.path.join(path, name))
                sub_run([shutil.which('pyccel'), os.path.join(path, name), '--language', language, '--openmp'], shell=False)
                print('\n')

    return

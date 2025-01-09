import argparse
import os
import sys
from shutil import which, rmtree
from subprocess import run as sub_run, PIPE, STDOUT  # nosec B404

import psydac
# Get the absolute path to the psydac directory
psydac_path = os.path.abspath(os.path.dirname(psydac.__path__[0]))

def pyccelize_files(root_path: str, language: str = 'fortran', openmp: bool = False):
    """
    Recursively pyccelize all files ending with '_kernels.py' in the given root path.

    Parameters
    ----------
    root_path : str
        Path to the Psydac source directory.
    language : str, optional
        Programming language for pyccel generated code ('fortran' or 'c'), by default 'fortran'.
    openmp : bool, optional
        Whether to enable OpenMP multithreading, by default False.
    """
    if language not in ['c', 'fortran']:
        print(f"Unsupported language: {language}")
    pyccel_path = which('pyccel')
    if pyccel_path is None:
        print("`pyccel` not found in PATH. Please ensure it is installed and accessible.")
        return

    parameters = ['--language', language]
    if openmp:
        parameters.append('--openmp')

    # Cleanup if any files of the opposite language exist
    cleanup = False
    for root, _, files in os.walk(root_path):
        for name in files:
            if name.endswith('_kernels.py'):
                file_path = os.path.join(root, name)
                # Check if the corresponding pyccelized file already exists
                subdir = "__pyccel__"
                generated_file_fortran = os.path.join(root, subdir, name[:-3] + '.f90')
                generated_file_c = os.path.join(root, subdir, name[:-3] + '.c')
                if language == 'fortran' and os.path.isfile(generated_file_c):
                    cleanup = True
                elif language == 'c' and os.path.isfile(generated_file_fortran):
                    cleanup = True
    if cleanup:
        cleanup_files(psydac_path)

    for root, _, files in os.walk(root_path):
        for name in files:
            if name.endswith('_kernels.py'):
                file_path = os.path.join(root, name)
                # Check if the corresponding pyccelized file already exists
                subdir = "__pyccel__"

                if language == 'fortran':
                    generated_file = os.path.join(root, subdir, name[:-3] + '.f90')
                elif language == 'c':
                    generated_file = os.path.join(root, subdir, name[:-3] + '.c')

                if os.path.isfile(generated_file):
                    print(f"Skipping {file_path}: Already pyccelized to {generated_file}")
                    continue  # Skip already pyccelized files

                # print(f"Pyccelizing: {file_path}")
                command = [pyccel_path, file_path] + parameters
                print(f"Running command: {' '.join(command)}")

                try:
                    result = sub_run(command, stdout=PIPE, stderr=STDOUT, text=True, shell=False, check=True)  # nosec B603
                    if result.stdout.strip():
                        print(result.stdout.strip())
                except Exception as e:
                    print(f"Failed to pyccelize {file_path}: {e}")

def cleanup_files(root_path: str):
    """
    Remove unnecessary build artifacts, such as `__pyccel__` directories and `.lock_acquisition.lock` files.
    """
    # Remove __pyccel__ directories
    for root, dirs, _ in os.walk(root_path):
        for dirname in dirs:
            if dirname == '__pyccel__':
                dir_to_remove = os.path.join(root, dirname)
                print(f"Removing directory: {dir_to_remove}")
                rmtree(dir_to_remove)

    # Remove .lock_acquisition.lock files
    for root, _, files in os.walk(root_path):
        for filename in files:
            if filename == '.lock_acquisition.lock':
                file_to_remove = os.path.join(root, filename)
                print(f"Removing lock file: {file_to_remove}")
                os.remove(file_to_remove)


def main():
    parser = argparse.ArgumentParser(
        description="Pyccelize Psydac kernel files and optionally clean up build artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--language', type=str, default='c', choices=['fortran', 'c'],
                        help="Language used to pyccelize kernel files.")
    parser.add_argument('--openmp', action='store_true',
                        help="Use OpenMP multithreading in generated code.")
    parser.add_argument('--cleanup', action='store_true',
                        help="Remove unnecessary files and directories after pyccelizing.")

    args = parser.parse_args()

    # Cleanup if requested
    if args.cleanup:
        print('Cleanup')
        cleanup_files(psydac_path)
    else:
        # Pyccelize kernel files
        pyccelize_files(psydac_path, language=args.language, openmp=args.openmp)


if __name__ == "__main__":
    main()

# Based on https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/console/compile.py
import argparse
import os
import sys
from shutil import which, rmtree
from subprocess import run as sub_run, PIPE, STDOUT  # nosec B404
import subprocess
import sysconfig

import psydac

# Get the absolute path to the psydac directory
psydac_path = os.path.abspath(psydac.__path__[0])
libdir = sysconfig.get_config_var("LIBDIR")
psydac_makefile_dir = os.path.join(psydac_path, "accelerate")

def subp_run(cmd, cwd=None, check=True):
    """Call subprocess.run and print run command."""

    if cwd is None:
        cwd = psydac_path

    print(f"\nRunning the following command as a subprocess:\n{' '.join(cmd)}")
    print(f"Running in directory: {cwd}")
    subprocess.run(cmd, cwd=cwd, check=check)


def psydac_compile(language, compiler, omp, delete, status, verbose, dependencies, yes):
    """
    Compile Psydac kernels. All files that contain "kernels" are detected automatically and saved to state.yml.

    Parameters
    ----------
    language : str
        Either "c" (default) or "fortran".

    compiler : str
        Either "GNU" (default), "intel", "PGI", "nvidia" or the path to a JSON compiler file.
        Only "GNU" is regularly tested at the moment.

    omp_pic : bool
        Whether to compile PIC kernels with OpenMP (default=False).

    omp_feec : bool
        WHether to compile FEEC kernels with OpenMP (default=False).

    delete : bool
        If True, deletes generated Fortran/C files and .so files (default=False).

    status : bool
        If true, prints the current Psydac compilation status on screen.

    verbose : bool
        Call pyccel in verbose mode (default=False).

    dependencies : bool
        Whether to print Psydac kernels (to be compiled) and their dependencies on screen.

    yes : bool
        Whether to say yes to prompt when changing the language.
    """
    if delete:
        cleanup_files(psydac_path)
        return

    pyccel_path = which("pyccel")
    if pyccel_path is None:
        print(
            "`pyccel` not found in PATH. Please ensure it is installed and accessible."
        )
        return

    sources = []
    # Cleanup if any files of the opposite language exist
    cleanup = False
    for root, _, files in os.walk(psydac_path):
        for name in files:
            if name.endswith("_kernels.py"):
                file_path = os.path.join(root, name)
                sources.append(file_path)
                # Check if the corresponding pyccelized file already exists
                subdir = "__pyccel__"
                generated_file_fortran = os.path.join(root, subdir, name[:-3] + ".f90")
                generated_file_c = os.path.join(root, subdir, name[:-3] + ".c")
                if language == "fortran" and os.path.isfile(generated_file_c):
                    cleanup = True
                elif language == "c" and os.path.isfile(generated_file_fortran):
                    cleanup = True
    if cleanup:
        if yes:
            yesno = "Y"
        else:
            if language == "fortran":
                compiled_in = "C"
            else:
                compiled_in = "fortran"
            yesno = input(
                f"Kernels compiled in language {compiled_in} exist, will be deleted, continue (Y/n)?"
            )
        if yesno in ("", "Y", "y", "yes"):
            cleanup_files(psydac_path)
        else:
            return

    # pyccel flags
    # TODO: Compile psydac with OpenMP
    flag_omp = ""
    if omp:
        flag_omp = "--openmp"
    sources = " ".join(sources)
    flags = "--language=" + language
    flags += " --compiler=" + compiler

    cmd = [
        "make",
        "-f",
        "compile_psydac.mk",
        "sources=" + sources,
        "flags=" + flags,
        "flags_openmp=" + flag_omp,
    ]
    print(os.path.join(libdir, "accelerate"))
    subp_run(cmd, cwd=psydac_makefile_dir),


def cleanup_files(root_path: str):
    """
    Remove unnecessary build artifacts, such as `__pyccel__` directories and `.lock_acquisition.lock` files.
    """
    sources = []
    for root, _, files in os.walk(root_path):
        for filename in files:
            if filename.endswith("_kernels.py"):
                file_path = os.path.join(root, filename)
                sources.append(file_path)
    sources = " ".join(sources)
    # Delete using the makefile
    cmd = [
        "make",
        "clean",
        "-f",
        "compile_psydac.mk",
        "sources=" + sources,
    ]
    subp_run(cmd, cwd=psydac_makefile_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Pyccelize Psydac kernel files and optionally clean up build artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--language",
        type=str,
        default="c",
        choices=["fortran", "c"],
        help="Language used to pyccelize kernel files.",
    )
    parser.add_argument(
        "--openmp",
        action="store_true",
        help="Use OpenMP multithreading in generated code.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="If True, deletes generated Fortran/C files and .so files (default=False).",
    )
    parser.add_argument(
        "--compiler",
        type=str,
        default="GNU",
        help='either "GNU" (default), "intel", "PGI", "nvidia" or the path to a JSON compiler file.',
    )
    parser.add_argument(
        "--status", action="store_true", help="Show the status of pyccelization."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument(
        "--dependencies",
        action="store_true",
        help="Print Psydac kernels to be compiled (.py) and their dependencies (.so) on screen.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Automatically answer 'yes' to prompts."
    )

    args = parser.parse_args()

    # Assuming psydac_compile is a function defined elsewhere
    psydac_compile(
        language=args.language,
        compiler=args.compiler,
        omp=args.openmp,
        delete=args.cleanup,
        status=args.status,
        verbose=args.verbose,
        dependencies=args.dependencies,
        yes=args.yes,
    )


if __name__ == "__main__":
    main()

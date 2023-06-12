def main():
    import argparse
    import os

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


    psydac_path = os.path.dirname(os.path.abspath(__file__))
    language    = args.language

    if language not in ['fortran', 'c']:
        print("The language given is not used by pyccel. It must be 'fortran' or 'c'.")
        language = 'fortran'

    os.system('pyccel ' + psydac_path + '/../linalg/stencil2coo_kernels.py --language '    + language)
    os.system('pyccel ' + psydac_path + '/../api/ast/transpose_kernels.py --language '     + language)
    os.system('pyccel ' + psydac_path + '/../core/field_evaluation_kernels.py --language ' + language)
    os.system('pyccel ' + psydac_path + '/../core/bsplines_kernels.py --language '         + language)


    return


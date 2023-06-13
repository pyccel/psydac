def main():
    import argparse
    import os
    from glob import glob

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


    psydac_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    language    = args.language

    if language not in ['fortran', 'c']:
        print("The language given is not used by pyccel. It must be 'fortran' or 'c'.")
        language = 'fortran'


    # for file in glob(psydac_path+"/*/*_kernels.py"):
    #     print('Pyccelise file :' + file)
    #     os.system('pyccel '+ file +' --language ' + language)

    for path, subdirs, files in os.walk(psydac_path):
        for name in files:
            if name.endswith('_kernels.py'):
                print('Pyccelise file :' + os.path.join(path, name))
                os.system('pyccel '+ os.path.join(path, name) +' --language ' + language)

    # os.system('pyccel ' + psydac_path + '/linalg/stencil2coo_kernels.py --language '    + language)
    # os.system('pyccel ' + psydac_path + '/api/ast/transpose_kernels.py --language '     + language)
    # os.system('pyccel ' + psydac_path + '/core/field_evaluation_kernels.py --language ' + language)
    # os.system('pyccel ' + psydac_path + '/core/bsplines_kernels.py --language '         + language)


    return


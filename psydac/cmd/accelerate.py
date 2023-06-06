def main():

    import os
    psydac_path = os.path.dirname(os.path.abspath(__file__))
    language    = os.getenv('PYCCEL_LANGUAGE')

    if language not in ['fortran', 'C']:
        language = 'fortran'

    os.system('pyccel ' + psydac_path + '/../linalg/kernels.py --language '         + language)
    os.system('pyccel ' + psydac_path + '/../api/ast/linalg_kernels.py --language ' + language)
    os.system('pyccel ' + psydac_path + '/../core/kernels.py --language '           + language)
    os.system('pyccel ' + psydac_path + '/../core/bsplines_pyccel.py --language '   + language)
    return


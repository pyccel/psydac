# coding: utf-8

import os

def mkdir_p(dir):
    if os.path.isdir(dir):
        return
    os.makedirs(dir)

def write_code(name, code, ext='py', folder='tmp'):
    filename = '{name}.{ext}'.format(name=name, ext=ext)
    if folder:
        mkdir_p(folder)
        filename = os.path.join(folder, filename)

        # add __init__.py for imports
        cmd = 'touch {}/__init__.py'.format(folder)
        os.system(cmd)

    f = open(filename, 'w')
    for line in code:
        f.write(line)
    f.close()

    return filename

#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

def add_header_to_python_files(directory, string_to_add, *, dry_run=True):

    print(f'Adding header to all Python files:')
    print(f'{header}')

    import os
    for root, dirs, files in os.walk(directory):
        if os.path.basename(root) in ['__pycache__', '__pyccel__', '__psydac__']:
            print(f'- Skipping folder: {root}')
            continue
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f'+ Processing file: {file_path}')
                with open(file_path, 'r+') as f:
                    content = f.read()
                    f.seek(0, 0)
                    if not dry_run:
                        f.write(string_to_add.rstrip('\r\n') + '\n' + content)
    print()

#==============================================================================
if __name__ == '__main__':

    # Test the function
#    header = '# This is a test string'

    # Add license header
    header = """
#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
    header = header.strip()

    dry_run = True
    add_header_to_python_files('../psydac'     , header, dry_run=dry_run)
    add_header_to_python_files('../examples'   , header, dry_run=dry_run)
    add_header_to_python_files('../mesh'       , header, dry_run=dry_run)

import re

modules = ['api.ast', 'api', 'api.printing', 
           'cad', 
           'cmd', 
           'core', 
           'ddm', 
           'feec',
           'fem', 
           'linalg', 'linalg.kernels', 
           'mapping', 
           'polar', 
           'utilities']

html_files = []
old_links = []

for module in modules:
    html_files.append( f'docs/build/html/modules/{module}.html' )
    old_links.append( rf'#module-psydac.{module}.(.*?)"' )

for i, html_file in enumerate(html_files):
    with open(html_file, 'r+') as f:
        content = f.read()
        content = re.sub(old_links[i], '"', content)
        f.seek(0,0)
        f.write(content)
        f.truncate()

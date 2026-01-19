project(
  'psydac',
  'c', 'fortran',
  meson_version: '>=1.1.0',
)

run_command('pyccel', 'make', '-g', 'psydac/**/*_kernels.py', '--convert-only',
    check: true)

find = find_program('find', required: true)

pyccel_meson_math_file = run_command(find, '__pyccel__/math', '-name', 'meson.build',
    check: true).stdout().strip()
pyccel_meson_cwrapper_file = run_command(find, '__pyccel__/cwrapper', '-name', 'meson.build',
    check: true).stdout().strip()
pyccel_meson_files = run_command(find, '__pyccel__/psydac', '-name', 'meson.build',
    check: true).stdout().split()

sed = find_program('sed', required: true)

run_command(sed, '-i.swp', 's/install: true//g', pyccel_meson_math_file, pyccel_meson_cwrapper_file,
    check: true)

foreach f : pyccel_meson_files
  f_parts = f.split('/')
  rel_f = f_parts.get(1)
  foreach i : range(2, f_parts.length()-1)
      rel_f = rel_f / f_parts.get(i)
  endforeach
  run_command(sed, '-i.swp', 's/install_dir: .*/subdir: \'' + rel_f.replace('/','\/') + '\',/g', f,
      check: true)
endforeach


igakit_proj = subproject('igakit')

py = import('python').find_installation(modules: ['numpy'], pure: false)

subdir('__pyccel__/math')
subdir('__pyccel__/cwrapper')
subdir('__pyccel__/psydac')

# Install the pure-Python package itself
install_subdir('psydac', install_dir: py.get_install_dir())

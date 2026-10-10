# Change Log

All notable changes to this project will be documented in this file.

## Unreleased

### Added

-   #527 : Add constructor `Geometry.from_file` with option `domain` to reuse an existing SymPDE domain, after checking that it matches the file
-   #576 : Add module `psydac.feec.polar.conga_projections` with broken FEEC polar projections in 2D
-   #576 : Add 2D Poisson and TE Maxwell examples on polar mapped domains in `psydac/feec/polar/examples`
-   #576 : Add module `psydac.utilities.parallel_utils` for parallel execution and gathering variable-length arrays
-   #576 : Add module `psydac.utilities.operators` with class `Laplacian` used in some old examples
-   #577 : Add an installation configuration option to choose the backend language
-   #567 : Improve `psydac test` command (with several new features)
-   #565 : Expand editable install info in `README.md`
-   [DEVELOPER] Create action `install_petsc4py` to install PETSc & `petsc4py` w/ complex support
-   [DEVELOPER] Configure Pylint in `pyproject.toml`
-   [DEVELOPER] Add `CLAUDE.md` with developer rules

### Fixed

-   #527 : Require `sympde==0.20.0` which changes how multipatch interfaces are defined
-   #527 : Fix `psydac.cad.elevate` and `refine` for NURBS mappings, see #608
-   #527 : Keep the NURBS weights in `psydac.cad.translate`, see #609
-   #527 : Fix type checks of input mappings in module `psydac.cad.cad`, see #610
-   #527 : Fix `psydac.cad.elevate` and `refine` for distributed mappings, see #611
-   #527 : Check the weights field in `NurbsMapping.__init__`, see #616
-   #527 : Fix reading multipatch geometries with `pdim=1`, see #617
-   #527 : Fix `psydac.cad.elevate` and `refine` for mappings with `pdim=1`, see #618
-   #576 : Require `sympde==0.19.3` which fixes a bug in the linearity checks
-   #579 : Require `h5py>=3.16` which installs correctly with `setuptools>=81.0`
-   #576 : Fix bug in `TensorFemSpace.eval_field` caused by round-off at MPI subdomain boundaries
-   #579 : Don't run postprocessing unit tests with `pytest-xdist` because `h5py` is not thread-safe
-   #579 : Return error code on failure of the `psydac test` and `psydac compile` commands
-   #577 : Fix installation following release of Pyccel 2.2
-   #571 : Fix correct application of the sum factorization algorithm
-   #567 : Fix parallel creation of folder `__psydac__` in `psydac.api.fem_bilinear_form`
-   #566 : Fix command `psydac test --mpi` on Ubuntu machines
-   [DEVELOPER] Add missing 'description' properties (required!) to our GitHub actions
-   [DEVELOPER] Update CI installation of `petsc4py` after release of `setuptools` 81.0
-   [DEVELOPER] Check correct reporting of failure for `psydac test` command in CI testing
-   [DEVELOPER] Use correct configuration file in coverage CI tests

### Changed

-   #527 : Pass a domain decomposition `ddm` and physical dimension `pdim` to `Geometry.__init__`; remove arguments `ncells`, `periodic`, `filename`, `comm` and `mpi_dims_mask`
-   #527 : Use the domain decomposition of the mapping's space in `Geometry.from_discrete_mapping`; remove arguments `comm` and `mpi_dims_mask`
-   #527 : Key `Geometry.mappings` by the names of the domain interiors
-   #527 : Return Cartesian (not homogeneous) control points from the NURBS functions in `psydac.cad.gallery`, see #608
-   #576 : Use latest version of Igakit (commit dalcinl/igakit@92ee097 of 2026/07/24) which supports NumPy >= 2.4
-   #595 : Use PETSc 3.25.5 whose Python bindings `petsc4py` are built correctly with `cython>=3`
-   #580 : Use PETSc 3.25.0 whose Python bindings `petsc4py` install correctly with `setuptools>=81.0`
-   #579 : Require `pyccel>=2.2.3` which can compile all kernels with C
-   #579 : Require `numpy>=2.1` to support Python >= 3.10
-   #579 : Require `pytest>=9.0` and use `pytest.toml` instead of `pytest.ini` for Pytest configuration
-   #579 : Move coverage configuration from `pyproject.toml` to `psydac/pytest.toml`
-   #570 : Optimize PSYDAC logo
-   [DEVELOPER] Update GitHub Actions for repository checkout and Python setup
-   [DEVELOPER] Rename actions: `macos/ubuntu_install` -> `macos/ubuntu_installations`
-   [DEVELOPER] Do not check file changes to trigger testing workflow on PRs
-   [DEVELOPER] Run documentation workflow on pushes to `devel` whenever `README.md` is modified
-   [DEVELOPER] Run testing and documentation workflows on PRs only when set to "ready for review"

### Deprecated

### Removed

-   #527 : Remove method `Geometry.read`, use constructor `Geometry.from_file` instead
-   #527 : Remove property `Geometry.is_parallel`

## [1.0.0] - 2026-01-19

The first official release on PyPI. A complete overhaul since version 0.1.

## [0.1] - 2020-01-14

The first beta version, not on PyPI.

# Psydac's ouputs
## Structure
Psydac has a class meant to take care of outputing simulation results. This class, named `OuputManager` is located in `psydac/api/postprocessing.py`.
It writes `FemSpace` related information in the Yaml syntax. The file looks like this:
```yaml
ndim: 2
fields: 'file.h5' # Name of the associated HDF5 file
patches: 
- name: 'patch_0'
  scalar_spaces:
  - name: V0
    kind: H1
    dtype: float
    rational: false
    periodic: [true, false]
    degree: [3, 3]
    basis: [B, B]
    knots:
    - [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
    - [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
  - &id001
    name: V1[0]
    kind: None
    dtype: float
    rational: false
    periodic: [true, false]
    degree: [2, 3]
    basis: [M, B]
    knots:
    - [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
    - [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
  - &id002
    name: V1[1]
    kind: None
    dtype: float
    rational: false
    periodic: [true, false]
    degree: [3, 2]
    basis: [B, M]
    knots:
    - [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
    - [0.0, 0.0, 0.5, 1.0, 1.0]
  - name: V2
    kind: L2
    dtype: float
    rational: false
    periodic: [true, false]
    degree: [2, 2]
    basis: [M, M]
    knots:
    - [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
    - [0.0, 0.0, 0.5, 1.0, 1.0]
  vector_spaces:
  - name: V1
    kind: Hcurl
    components:
    - *id001
    - *id002
- name: patch_1
  scalar_spaces:
  - name: V0
    kind: UndefinedSpaceType()
    dtype: float
    rational: 'false'
    periodic: [false, false]
    degree: [2, 2]
    basis: [B, B]
    knots:
    - [0.5, 0.5, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0, 1.0]
    - [-1.0, -1.0, -1.0, -0.75, -0.5, -0.25, 0.0, 0.0, 0.0]
```
The field coefficients are saved to the `HDF5` format in the following manner :
```bash
file.h5
    attribute: spaces # name of the aforementioned Yaml file 
    static/
        scalar_space_1/
            field_s1_1
            field_s1_2
            ....
            field_s1_n
        vector_space_1_[0]/
            attribute: parent_space # 'vector_space_1'
            field_v1_1_[0]
                attribute: parent_field # 'field_v1_1'
        vector_space_1_[1]/
            attribute: parent_space # 'vector_space_1'
            field_v1_1_[1]
                attribute: parent_field # 'field_v1_1'
        ...
    snapshot_1/
        attribute: t
        attribute: ts 
        space_1/
        ...
        space_n/
    ...
    snapshot_n/
```
In addition to that, psydac also has a class to read those files, recreate all the `FemSpace` and `FemField` objects and export them to `VTK`. 

## Usage of class `OutputManager`

An instance of the `OutputManager` class is created at the beginning of the simulation, by specifying the following:

1.  The name of the YAML file (e.g. `spaces.yml`) where the information about all FEM spaces will be written, and
2.  The name of the HDF5 file (e.g. `fields.h5`) where the coefficients of all FEM fields will be written.

References to the available FEM spaces are given to the OutputManager object through the `add_spaces(**kwargs)` method, and the corresponding YAML file is created upon calling the method `export_spaces_info()`. In order to inform the OutputManager object that the next fields to be exported are time-independent, the user should call the `set_static()` method. In the case of time-dependent fields, the user should prepare a time snapshot (which is defined for a specific time step index `ts` and time value `t`) by calling the method `add_snapshot(t, ts)`. In both cases the fields are exported to the HDF5 file through a call to the method `export_fields(**kwargs)`. Here is a usage example:

```python
# SymPDE Layer
# Discretization 
# V0h and V1h are discretized SymPDE Space
# u0 and u1 are FemFields belonging to either of those spaces
output_m = OutputManager('spaces.yml', 'fields.h5')

output_m.add_spaces(V0=V0h, V1=V1h) 

output_m.set_static() # Tells the object to save in /static/
output_m.export_fields(u0_static=u0, u1_static=u1) # Actually does the saving

output_m.add_snapshot(t=0., ts=0) 
# The line above tells the object to:
# 1. create the group snapshot_x with attribute t and ts
# 2. save in this snapshot
output_m.export_fields(u0=u0, u1=u1)

output_m.export_spaces_info() # Writes the space information to Yaml
```

## Usage of class `PostProcessManager`

Typically the `PostProcessManager` class is used in a separate post-processing script, which is run after the simulation has finished. In essence it evaluates the FEM fields over a uniform grid (applying the appropriate push-forward operations) and exports the values to a VTK file (or a sequence of files in the case of a time series). An instance of the `PostProcessManager` class is created by specifying the following:

1.  The name of the geometry file (in HDF5 format) which defines the multi-patch geometry of interest
2.  The name of the YAML file that contains the information about the FEM spaces
3.  The name of the HDF5 file that contains the coefficients of all the FEM fields

In order to export the fields to a VTK file, the user needs to prepare the evaluation grid `grid` and then call the method `export_to_vtk(base_name, grid, npts_per_cell, snapshots, fields)`, where `base_name` is the base name for the VTK output files, `npts_per_cell` specifies the refinement in the case of a uniform grid, `snapshots` specifies which time snapshots should be extracted from the HDF5 file (`None` in the case of static fields) and `fields` is a dictionary of `(vtk_field_name, h5_field_name)` pairs. Here is a usage example:

```python
# geometry.h5 is where the domain comes from. See PostProcessManager's docstring for me information
post = PostProcessManager(geometry_file='geometry.h5', space_file='spaces.yml', fields_file='fields.h5')

# See PostProcessManager.export_to_vtk's and TensorFemSpace.eval_fields' doscstrings for more information
post.export_to_vtk('filename_vtk', grid, npts_per_cell=npts_per_cell, snapshots='all', fields = {'u0': 'field1', 'u1': 'field2'})
```

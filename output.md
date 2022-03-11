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
In addition to that, psydac also has a class to read those files, recreate and the `FemSpace` and `FemField` objects and export them to `VTK`. 
## Usage
Those two classes are meant to be used as follows. The `#` line is supposed to represent the end of a file.
```python
# SymPDE Layer
# Discretization 
# V0h and V1h are discretized SymPDE Space
# u0 and u1 are FemFields belonging to either of those spaces
output_m = OutputManager('spaces.yml', 'fields.h5')

output_m.add_spaces(V0=V0h, V1=V1h) 

output_m.set_static() # Tells the object to save in /static/
output_m.export_fields(u0=u0, u1=u1) # Actually does the saving

output_m.add_snapshot(t=0., ts=0) 
# The line above tells the object to:
# 1. create the group snapshot_x with attribute t and ts
# 2. save in this snapshot
output_m.export_fields(u0=u0, u1=u1)

output_m.export_spaces_info() # Writes the space information to Yaml

###############################################################################
# geometry.h5 is where the domain comes from
post = PostProcessManager('geometry.h5', 'spaces.yml', 'fields.h5')
post.recontruct_scope() 
# Fills post.spaces and post.fields two dictionnaries 
# See PostProcessManager's docstring for me information on those
post.export_to_vtk('filename_vtk',grid, npts_per_cell=npts_per_cell, dt=dt, u0='u0', u1='u1')
# See PostProcessManager.export_to_vtk's and TensorFemSpace.eval_fields' doscstrings for more information. 
```

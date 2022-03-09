# Structure of the Psydac's output
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
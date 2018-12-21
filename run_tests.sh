#!/bin/bash

#python3 -m pytest --pyargs spl/api/tests/ -m "not parallel"
#mpirun -np 2 python3 -m pytest --pyargs spl/api/tests/ -m "parallel"


python3 -m pytest spl/api/tests/test_api_2d_scalar.py
python3 -m pytest spl/api/tests/test_api_2d_scalar_mapping.py
python3 -m pytest spl/api/tests/test_api_2d_vector.py
python3 -m pytest spl/api/tests/test_api_2d_vector_mapping.py
python3 -m pytest spl/api/tests/test_api_3d_scalar.py
python3 -m pytest spl/api/tests/test_api_3d_scalar_mapping.py
python3 -m pytest spl/api/tests/test_api_3d_vector.py

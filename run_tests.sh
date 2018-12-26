#!/bin/bash

python3 -m pytest --pyargs spl/api/tests/ -m "not parallel"
mpirun -np 2 python3 -m pytest --pyargs spl/api/tests/ -m "parallel"

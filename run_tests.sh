#!/bin/bash

python3 spl/api/codegen/tests/test_kernel_1d.py
python3 spl/api/codegen/tests/test_kernel_2d.py
python3 spl/api/codegen/tests/test_kernel_3d.py

python3 spl/api/codegen/tests/test_assembly_1d.py
python3 spl/api/codegen/tests/test_assembly_2d.py
python3 spl/api/codegen/tests/test_assembly_3d.py

python3 spl/api/codegen/tests/test_interface_1d.py
python3 spl/api/codegen/tests/test_interface_2d.py
python3 spl/api/codegen/tests/test_interface_3d.py

python3 spl/api/tests/test_api_1d.py
python3 spl/api/tests/test_api_2d.py
python3 spl/api/tests/test_api_3d.py


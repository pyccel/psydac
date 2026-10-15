Contributing to the Docs
========================

We highly encourage anyone working with PSYDAC to contribute to the library
by adding new or by improving already existing docstrings.

PSYDAC's documentation is built using Sphinx and the `numpydoc` extension.
In particular, this means that this documentation is automatically generated from the library's docstrings, 
and hence is only as good as the docstrings themselves.

Contributing is as simple as adding docstrings and creating a pull request. 
Alternatively, you can add your docstring suggestions to an already existing "Improve docstrings CWxx" PR.

This documentation supports the math-dollar-sign in docstrings, which means that you can use LaTeX to write mathematical expressions.
We have started using this feature in the `linalg.basic` module, and we encourage you to use it as well.

Building the Docs
-----------------

In order to verify whether your docstrings are correctly formatted, you can build the documentation locally.
To do so, execute the following commands from the root directory of the repository:

.. code-block::

   # go to psydac/
   # if on Linux
   >>> sudo apt install graphviz
   # if on macOS
   >>> brew install graphviz

   >>> python -m pip install -r docs/requirements.txt

   # go to psydac/docs/
   >>> rm -rf source/modules/STUBDIR
   >>> make clean
   >>> make html

   # go to psydac/
   # this script cannot be executed properly from within psydac/docs/
   >>> python docs/update_links.py

   # Open the documentation on your browser with 
   >>> open docs/build/html/index.html

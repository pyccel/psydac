#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
__all__     = ['__version__', 'api', 'cad', 'core', 'ddm', 'feec', 'fem',
               'linalg', 'mapping', 'utilities']

from psydac.version import __version__

# TODO [YG 24.11.2025]: verify if we can remove the imports below
from psydac import api
from psydac import cad
from psydac import core
from psydac import ddm
from psydac import feec
from psydac import fem
from psydac import linalg
from psydac import mapping
from psydac import utilities

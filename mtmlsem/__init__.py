"""Multi-trait multi-locus SEM for GWAS, genomic prediction and selection."""

name = "mtmlSEM"
__version__ = "0.0.1"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"


from .mtml_model import mtmlModel
from .dataset import Data, CVset
from .manhattan import manhattan
from . import add_snps
from . import readers

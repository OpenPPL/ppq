from typing import List

from ppq.IR import Operation

from .default import DEFAULT_BACKEND_TABLE
from .base import *

import torch

ONNX_BACKEND_TABLE = DEFAULT_BACKEND_TABLE.copy()

# When you trying to implement a custimized function for ppl_gpu platform
# Be aware that you can just overwrite part of DEFAULT_DISPATCHING_TABLE
# rather than rewrite all dispatching table.
# here an example was given: Sample_Forward
def Sample_Forward():
    return None

ONNX_BACKEND_TABLE['Sample_Forward'] = Sample_Forward

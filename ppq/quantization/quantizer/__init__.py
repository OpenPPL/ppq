from .base import BaseQuantizer
from .DSPQuantizer import PPL_DSP_Quantizer
from .MyQuantizer import ExtQuantizer
from .NXPQuantizer import NXP_Quantizer
from .PPLQuantizer import (PPLCUDA_INT4_Quantizer,
                           PPLCUDAMixPrecisionQuantizer, PPLCUDAQuantizer)
from .TRTQuantizer import TensorRTQuantizer
from .ORTQuantizer import ORT_PerTensorQuantizer, ORT_PerChannelQuantizer
from .AcademicQuantizer import ACADEMICQuantizer, ACADEMIC_INT4_Quantizer, ACADEMIC_Mix_Quantizer
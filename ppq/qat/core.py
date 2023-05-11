from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from ppq import PPQuantFunction, SingletonMeta
from ppq.core import (OperationQuantizationConfig, TargetPlatform,
                      TensorQuantizationConfig, ppq_warning)
from ppq.IR import Operation, Variable
from ppq.lib import (FloatingQuantizationConfig, LinearQuantizationConfig,
                     ParameterQuant, TensorQuant)
from ppq.quantization.quantizer import BaseQuantizer


class QuantLayer():
    def __init__(self) -> None:
        self._quantize_controler = None


class QATController():
    def __init__(self) -> None:
        self.export_mode       = 'Plain Onnx'


class QConv1d(torch.nn.Conv1d, QuantLayer):
    def __init__(self, controller: QATController, **kwargs) -> None:
        super(torch.nn.Conv1d).__init__(**kwargs)
        self._quantize_controler = controller

        self.input_quant = TensorQuant(
            quant_config = LinearQuantizationConfig(
                symmetrical = True, power_of_2 = False, 
                channel_axis = None, # channel_axis = None -> Per tensor Quantization
                calibration = 'percentile'))

        self.weight_quant = ParameterQuant(
            quant_config = LinearQuantizationConfig(
                symmetrical = True, power_of_2 = False, 
                channel_axis = 0, # channel_axis = 1 -> Per channel Quantization
                calibration = 'minmax'))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(input)
        w = self.weight_quant(self.weight)
        return self._conv_forward(x, w, self.bias)


class QConv2d(torch.nn.Conv2d, QuantLayer):
    def __init__(self, controller: QATController, **kwargs) -> None:
        super(torch.nn.Conv2d).__init__(**kwargs)
        self._quantize_controler = controller

        self.input_quant = TensorQuant(
            quant_config = LinearQuantizationConfig(
                symmetrical = True, power_of_2 = False, 
                channel_axis = None, # channel_axis = None -> Per tensor Quantization
                calibration = 'percentile'))

        self.weight_quant = ParameterQuant(
            quant_config = LinearQuantizationConfig(
                symmetrical = True, power_of_2 = False, 
                channel_axis = 0, # channel_axis = 1 -> Per channel Quantization
                calibration = 'minmax'))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(input)
        w = self.weight_quant(self.weight)
        return self._conv_forward(x, w, self.bias)


class QConv3d(torch.nn.Conv3d, QuantLayer):
    def __init__(self, controller: QATController, **kwargs) -> None:
        super(torch.nn.Conv3d).__init__(**kwargs)
        self._quantize_controler = controller

        self.input_quant = TensorQuant(
            quant_config = LinearQuantizationConfig(
                symmetrical = True, power_of_2 = False, 
                channel_axis = None, # channel_axis = None -> Per tensor Quantization
                calibration = 'percentile'))

        self.weight_quant = ParameterQuant(
            quant_config = LinearQuantizationConfig(
                symmetrical = True, power_of_2 = False, 
                channel_axis = 1, # channel_axis = 1 -> Per channel Quantization
                calibration = 'minmax'))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(input)
        w = self.weight_quant(self.weight)
        return self._conv_forward(x, w, self.bias)


class ENABLE_CALIBRATION:
    """ """
    def __init__(self, controler: QATController) -> None:
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

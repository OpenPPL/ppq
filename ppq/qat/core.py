import torch
import torch.nn as nn
from ppq import PPQuantFunction
from ppq.core import (OperationQuantizationConfig, TargetPlatform,
                      TensorQuantizationConfig, ppq_warning)
from ppq.IR import Operation, Variable
from ppq.quantization.quantizer import BaseQuantizer


class ENABLE_CALIBRATION:
    """ """
    def __init__(self, model: nn.Module, quantizer: BaseQuantizer) -> None:
        self.model = model

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class PPQuantComputingLayer(nn.Module):
    def __init__(self, convert_from: torch.nn.Module, config: OperationQuantizationConfig = None) -> None:
        super().__init__()
        
        self.is_scale_trainable = False
        self.config = config
        
        if not isinstance(self.config, OperationQuantizationConfig):
            raise TypeError('Unexpected Initialization Error, Quantization Config is Invalid.')
        
        assert len(self.config.input_quantization_config) == 3, (
            'Quantization Config of PPQuantComputingLayer should has exactly 3 input tensor quantization config(s).', 
            f'While {len(self.config.input_quantization_config)} was given.')
        self._input_config  = self.config.input_quantization_config[0]
        self._weight_config = self.config.input_quantization_config[1]
        self._bias_config   = self.config.input_quantization_config[2]
        self._layer         = convert_from

        try:
            self.bias   = convert_from.bias
            self.weight = convert_from.weight
        except KeyError as e:
            ppq_warning('Failed to Create PPQuantComputingLayer on your Pytorch Layer, this layer will keep as unchanged.')
            self.bias   = None
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            self._layer.weight = PPQuantFunction(self.weight, self._weight_config)
        if self.bias is not None:
            self._layer.bias = PPQuantFunction(self.bias, self._input_config)
        return self._layer.forward(x)  


class PPQuantConv(PPQuantComputingLayer):
    def __init__(self, convert_from: torch.nn.Module, config: OperationQuantizationConfig = None) -> None:
        super().__init__(convert_from, config)
        if type(convert_from) not in {nn.Conv1d, nn.Conv2d, nn.Conv3d}:
            raise TypeError(f'Can not Convert layer {type(convert_from)} to PPQuantConv, Layer type is invalid.')


class PPQuantConvTranspose(PPQuantComputingLayer):
    def __init__(self, convert_from: torch.nn.Module, config: OperationQuantizationConfig = None) -> None:
        super().__init__(convert_from, config)
        if type(convert_from) not in {nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d}:
            raise TypeError(f'Can not Convert layer {type(convert_from)} to PPQuantConvTranspose, Layer type is invalid.')


class PPQuantLinear(PPQuantComputingLayer):
    def __init__(self, convert_from: torch.nn.Module, config: OperationQuantizationConfig = None) -> None:
        super().__init__(convert_from, config)
        if type(convert_from) not in {nn.Linear}:
            raise TypeError(f'Can not Convert layer {type(convert_from)} to PPQuantLinear, Layer type is invalid.')


class QATHelper:
    @ staticmethod
    def generate_config_by_platform(layer: torch.nn.Module, platform: TargetPlatform):
        quantizer = create_quantizer(platform) # creating a quantizer is not expansive.
        created = quantizer.init_quantize_config(
            Operation(name='Unamed', op_type='', inputs=[Variable(name='Unamed') for _ in range(3)]))
        return created

    @ staticmethod
    def report_quantization_state(model: nn.Module):
        pass

    @ staticmethod
    def quantize_pytorch_layer(layer: nn.Module, platform: TargetPlatform) -> PPQuantComputingLayer:
        config = QATHelper.generate_config_by_platform(layer=layer, platform=platform)

        if type(layer) in {nn.Conv1d, nn.Conv2d, nn.Conv3d}:
            return PPQuantConv(convert_from=layer, config=config)
        
        if type(layer) in {nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d}:
            return PPQuantConvTranspose(convert_from=layer, config=config)
        
        if type(layer) in {nn.Linear}:
            return PPQuantLinear(convert_from=layer, config=config)

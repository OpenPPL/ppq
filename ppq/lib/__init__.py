"""
## PPQ Foundation Library - PFL

PPQ 基础类库

PFL is a collection of basic classes and functions that provides fundamental functionalities.

    - Parser: Get a network parser.
    
    - Exporter: According to given platform, get a network exporter.
    
    - OperationForwardFunction: According to given platform and optype, get a forward function.
    
    - Dispatcher: Get a network dispatcher.
    
    - FloatingQuantizationConfig: Get a TensorQuantizationConfig for FP8 Quantization.
    
    - LinearQuantizationConfig: Get a TensorQuantizationConfig for INT8 Quantization.
    
    - QuantStub: Get a QuantStub class instance.
    
    - Quantizer: Get a Quantizer corresponding to given platform.
    
    - Observer: Get a Tensor Observer, which is bound to given TensorQuantizationConfig.
    
    - Pipeline: Build Optimization Pipeline.
    
    - QuantFunction: Get PPQ Default Quantize Function.
    
PFL also provides a set of functions to register Quantizer, Parser, Exporter to PPQ.

    - register_network_quantizer
    
    - register_network_parser
    
    - register_network_exporter
    
    - register_operation_handler
    
    - register_calibration_observer
    
"""

from .extension import (register_calibration_observer,
                        register_network_exporter, register_network_parser,
                        register_network_quantizer, register_operation_handler)
from .quant import (Dispatcher, Exporter, FloatingQuantizationConfig,
                    LinearQuantizationConfig, Observer,
                    OperationForwardFunction, Parser, Pipeline, QuantFunction,
                    Quantizer, TensorQuant, ParameterQuant)

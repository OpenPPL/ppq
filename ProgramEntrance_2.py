"""
This is a highly flexible PPQ quantization entry script, 
    and you will witness the power of PPQ as an offline neural network quantization tool.

PPQ abstracts neural network quantization into several parts, such as 
    Quantizer,                  (ppq.quantization.quantizers) 
    Optimization Pass,          (ppq.quantization.optim)
    Optimization Pipeline,      (ppq.quantization.optim)
    Exporter,                   (ppq.parser)
    Tensor Quantization Config, (ppq.core)
    etc.

In this example, we will create them one by one to customize the entire quantization logic. 
    We will customize the quantization rules and use a custom optimization process to fine-tune the quantization rules in detail. 
    We will create a custom exporter to print all quantization information to the screen.

"""

import os

import numpy as np
import torch
import torchvision

import ppq.lib as PFL
from ppq import (BaseGraph, BaseQuantizer, GraphExporter, Operation,
                 OperationQuantizationConfig, QuantableOperation,
                 QuantableVariable, QuantizationOptimizationPass, SearchableGraph,
                 QuantizationPolicy, QuantizationProperty, QuantizationStates,
                 TargetPlatform, TorchExecutor, graphwise_error_analyse)
from ppq.api import ENABLE_CUDA_KERNEL, load_torch_model
from ppq.quantization.optim import *


class MyExporter(GraphExporter):
    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, **kwargs):
        print('This exporter does not export quantitative information to file, '
              'it prints quantitative information to the console instead.')
        for opname, op in graph.operations.items():
            # Skip those operators that are not involved in quantization.
            # They do not have a quantization configuration.
            if not isinstance(op, QuantableOperation): continue

            print(f'### Quantization Configuration of {opname}: ')
            for idx, config in enumerate(op.config.input_quantization_config):
                print(f'\t #### Input {idx}: ')
                print(f'\t Scale: {config.scale.tolist()}')
                print(f'\t Offset: {config.offset.tolist()}')
                print(f'\t State: {config.state}')
                print(f'\t Bitwidth: {config.num_of_bits}')
                print(f'\t Quant_min: {config.quant_min}')
                print(f'\t Quant_max: {config.quant_max}')
            
            for idx, config in enumerate(op.config.output_quantization_config):
                print(f'\t #### Output {idx}: ')
                print(f'\t Scale: {config.scale.tolist()}')
                print(f'\t Offset: {config.offset.tolist()}')
                print(f'\t State: {config.state}')
                print(f'\t Bitwidth: {config.num_of_bits}')
                print(f'\t Quant_min: {config.quant_min}')
                print(f'\t Quant_max: {config.quant_max}')


class MyInt8Quantizer(BaseQuantizer):
    def __init__(self, graph: BaseGraph, per_channel: bool = True, 
                 sym: bool = True, power_of_2: bool = True, 
                 num_of_bits: int = 8) -> None:
        """ A Generalized int8 Quantizer. """
        assert 16 >= num_of_bits >= 2, 'Unacceptable bit-width.'
        
        self.num_of_bits = num_of_bits
        self.power_of_2  = power_of_2
        self.per_channel = per_channel
        self.symmetric   = sym

        if sym:
            self.quant_min = -pow(2, num_of_bits - 1)
            self.quant_max = pow(2, num_of_bits - 1) - 1
            self.policy    = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR + 
                QuantizationProperty.LINEAR + 
                QuantizationProperty.SYMMETRICAL)
        else:
            self.quant_min = 0
            self.quant_max = pow(2, num_of_bits) - 1
            self.policy    = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR + 
                QuantizationProperty.LINEAR + 
                QuantizationProperty.ASYMMETRICAL)

        if power_of_2:
            self.policy = QuantizationPolicy(
                self.policy._policy + 
                QuantizationProperty.POWER_OF_2)

        super().__init__(graph, True)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        """
        When implementing a custom quantizer, you need to initialize the quantization 
        information structure(TQC) for each type of operators.
        
        Check Predefined Quantizers within ppq.quantization.quantizer folder, see how to implements a
        customized quantizer.
        
        TQC is made up of input_quantization_config and output_quantization_config.
        The quantization information includes 
            quantization policy, 
            quantization bit width, 
            quantization maximum and minimum values,
            and scale & offset.

        Scale and offset are generated and maintained by the calibration pass.
        """
        OQC = self.create_default_quant_config(
            op=operation, num_of_bits=8,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            observer_algorithm='percentile',
            policy=self.policy
        )

        if operation.type in {'Conv', 'ConvTranspose', 'MatMul', 'Gemm'}:

            if operation.num_of_input == 3: # has bias
                # disable quantization of bias
                OQC.input_quantization_config[-1].state = QuantizationStates.PASSIVE_INIT
                OQC.input_quantization_config[-1].quant_min   = -1 << 30
                OQC.input_quantization_config[-1].quant_max   = 1 << 30
                OQC.input_quantization_config[-1].num_of_bits = 32

            # modify calibration method of parameter(for higher accuracy)
            OQC.input_quantization_config[1].observer_algorithm = 'minmax'

            # for both SYMMETRICAL and ASYMMETRICAL quantization,
            # weight should always be quantized symmetrically.
            OQC.input_quantization_config[1].quant_min = - pow(2, self.num_of_bits - 1)
            OQC.input_quantization_config[1].quant_max = pow(2, self.num_of_bits - 1) - 1
            OQC.input_quantization_config[1].policy = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR +
                QuantizationProperty.LINEAR + 
                QuantizationProperty.SYMMETRICAL +
                (QuantizationProperty.POWER_OF_2 if self.power_of_2 else 0))

            if operation.num_of_parameter > 1:
                # Per-channel Variation
                if self.per_channel:
                    OQC.input_quantization_config[1].policy = QuantizationPolicy(
                        QuantizationProperty.PER_CHANNEL + 
                        QuantizationProperty.LINEAR + 
                        QuantizationProperty.SYMMETRICAL +
                        (QuantizationProperty.POWER_OF_2 if self.power_of_2 else 0))
                    OQC.input_quantization_config[1].channel_axis = 0

                    if operation.type == 'ConvTranspose':
                        OQC.input_quantization_config[1].channel_axis = 1

        elif operation.type in {'LayerNormalization'}:
            # LayerNormalization only take input & output quantization, parameter shall not been quantized.
            for input_config in OQC.input_quantization_config[1:]:
                input_config.state = QuantizationStates.FP32

        return OQC

    @ property
    def quant_operation_types(self) -> set:
        return {'Conv', 'ConvTranspose', 'MatMul', 'Gemm', 
                'Relu', 'Clip', 'Sub', 'Abs', 'Mul',
                'LayerNormalization'}



class MyOptimPass(QuantizationOptimizationPass):
    """
    This custom Optimization Pass will perform a series of customized quantization.
    This is an example code, and you need to carefully read the code definition of the 
        Optimization Pass and understand how to control the quantization logic through the code.

    This Optimization Pass will:
        1. fuse relu - clip structure.
        2. set clip output scale in the network to 1/127.
        3. set the input and output quantization information of the abs operators to be the same.
        4. modify calibration method for some operators.
    """
    def __init__(self, name: str = 'My Optim Pass') -> None:
        super().__init__(name)

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        # fuse relu - clip, set output scale of relu to be 1/127
        processor = SearchableGraph(graph)
        patterns = processor.pattern_matching(
            patterns=['Relu', 'Clip'],
            edges=[[0, 1]], exclusive=True)

        for relu, clip in patterns:
            print(f'Fuse {relu.name} and {clip.name}')
            if not isinstance(clip, QuantableOperation): continue
            if not isinstance(relu, QuantableOperation): continue
            relu.config.output_quantization_config[0].dominated_by = (
                clip.config.output_quantization_config[0])
            clip.config.input_quantization_config[0].dominated_by = (
                clip.config.output_quantization_config[0])
            clip.config.output_quantization_config[0].scale = torch.tensor(1 / 127).cuda()
            clip.config.output_quantization_config[0].offset = torch.tensor(0.0).cuda()
            clip.config.output_quantization_config[0].state = QuantizationStates.ACTIVATED

        # keep input and output scale of abs as the same.
        for op in graph.operations.values():
            print(f'Op {op.name} has processed.')
            if op.type != 'Abs': continue
            if (isinstance(op, QuantableOperation)):
                ITQC = op.config.input_quantization_config[0]
                OTQC = op.config.output_quantization_config[0]
                ITQC.dominated_by = OTQC

        # modify calibration methods.
        for op in graph.operations.values():
            if op.name != 'My Op': continue
            if (isinstance(op, QuantableOperation)):
                ITQC = op.config.input_quantization_config[0]
                OTQC = op.config.output_quantization_config[0]
                ITQC.observer_algorithm = 'kl'
                OTQC.observer_algorithm = 'mse'


# load data.
calibration_dataloader = []
for file in os.listdir('imagenet'):
    path = os.path.join('imagenet', file)
    arr  = np.fromfile(path, dtype=np.dtype('float32')).reshape([1, 3, 224, 224])
    calibration_dataloader.append(torch.tensor(arr))

with ENABLE_CUDA_KERNEL():
    model = torchvision.models.resnet18(pretrained=True).cuda()
    graph = load_torch_model(model=model, sample=torch.zeros(size=[1, 3, 224, 224]).cuda())

    quantizer   = MyInt8Quantizer(graph=graph, per_channel=True, sym=True, power_of_2=False)
    dispatching = PFL.Dispatcher(graph=graph).dispatch(
        quant_types=quantizer.quant_operation_types)

    # initialize quantization information
    for op in graph.operations.values():
        quantizer.quantize_operation(
            op_name = op.name, platform = dispatching[op.name])

    collate_fn = lambda x: x.to('cuda')
    executor = TorchExecutor(graph=graph, device='cuda')
    executor.tracing_operation_meta(inputs=torch.zeros(size=[1, 3, 224, 224]).cuda())
    executor.load_graph(graph=graph)

    # Manually create a quantization optimization pipeline.
    pipeline = PFL.Pipeline([
        QuantizeSimplifyPass(),
        QuantizeFusionPass(
            activation_type=quantizer.activation_fusion_types),
        ParameterQuantizePass(),
        MyOptimPass(),                                         # <----- Insert Our Optimization Pass 
        RuntimeCalibrationPass(),
        PassiveParameterQuantizePass(),
        QuantAlignmentPass(force_overlap=True),

        # 微调你的网路
        # LearnedStepSizePass(steps=1500)

        # 如果需要训练微调网络，训练过程必须发生在 ParameterBakingPass 之前
        # ParameterBakingPass()
    ])

    # Calling quantization optimization pipeline.
    pipeline.optimize(
        graph=graph, dataloader=calibration_dataloader, verbose=True, 
        calib_steps=32, collate_fn=collate_fn, executor=executor)

    graphwise_error_analyse(
        graph=graph, running_device='cuda', dataloader=calibration_dataloader, 
        collate_fn=lambda x: x.cuda())

    MyExporter().export(file_path=None, graph=graph, config_path=None)

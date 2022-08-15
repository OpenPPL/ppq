from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Union

from ppq.core import OperationMeta, TargetPlatform, TensorQuantizationConfig
from ppq.executor.op import (DEFAULT_BACKEND_TABLE, EXTENSION_BACKEND_TABLE,
                             NXP_BACKEND_TABLE, PPL_DSP_BACKEND_TABLE,
                             PPL_GPU_BACKEND_TABLE, SOI_BACKEND_TABLE,
                             ONNX_BACKEND_TABLE, ACADEMIC_BACKEND_TABLE)
from ppq.IR import BaseGraph, Operation, QuantableOperation

import torch

GLOBAL_DISPATCHING_TABLE = {platform:{} for platform in TargetPlatform}
GLOBAL_DISPATCHING_TABLE[TargetPlatform.FP32] = DEFAULT_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.TRT_INT8] = PPL_GPU_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.NCNN_INT8] = DEFAULT_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.TENGINE_INT8] = DEFAULT_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.FPGA_INT8] = DEFAULT_BACKEND_TABLE

GLOBAL_DISPATCHING_TABLE[TargetPlatform.PPL_CUDA_FP16] = DEFAULT_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.PPL_CUDA_INT8] = PPL_GPU_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.PPL_CUDA_INT4] = PPL_GPU_BACKEND_TABLE

GLOBAL_DISPATCHING_TABLE[TargetPlatform.UNSPECIFIED] = DEFAULT_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.PPL_DSP_INT8] = PPL_DSP_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.PPL_DSP_TI_INT8] = PPL_DSP_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.NXP_INT8] = NXP_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.SHAPE_OR_INDEX] = SOI_BACKEND_TABLE

GLOBAL_DISPATCHING_TABLE[TargetPlatform.ORT_OOS_INT8] = ONNX_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.METAX_INT8_C] = DEFAULT_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.METAX_INT8_T] = DEFAULT_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.ACADEMIC_INT4] = ACADEMIC_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.ACADEMIC_INT8] = ACADEMIC_BACKEND_TABLE

GLOBAL_DISPATCHING_TABLE[TargetPlatform.EXTENSION] = EXTENSION_BACKEND_TABLE
GLOBAL_DISPATCHING_TABLE[TargetPlatform.OPENVINO_INT8] = DEFAULT_BACKEND_TABLE

def register_operation_handler(handler: Callable, operation_type: str, platform: TargetPlatform):
    """Regitser a custimized function as operation handler.
    
    Function should accept at least 3 input parameters, return one or more tensor as result:
    func(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    
    If there is already another operation handler for given operation_type,
        new handler will replace the old one without warrning.

    Args:
        handler (Callable): _description_
        operation_type (str): _description_
        platform (TargetPlatform): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    if platform not in GLOBAL_DISPATCHING_TABLE:
        raise ValueError('Unknown Platform detected, Please check your platform setting.')
    GLOBAL_DISPATCHING_TABLE[platform][operation_type] = handler


class RuntimeHook(metaclass=ABCMeta):
    """RuntimeHook is an abstract class designed for executor customizing.

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """
    def __init__(self, operation: Operation, operation_meta: OperationMeta = None) -> None:
        self._hook_to = operation
        self._op_meta = operation_meta

    def pre_forward_hook(self, inputs: list, **kwargs) -> list:
        """user-customized pre-processing procedure of input data.

        Args:
            inputs (list): a list includes all input data.

        Returns:
            list: a list includes all input data(processed).
        """
        return inputs

    def post_forward_hook(self, outputs: list, **kwargs) -> list:
        """user-customized post-processing procedure of output data.

        Args:
            inputs (list): a list includes all output data.

        Returns:
            list: a list includes all output data(processed).
        """
        return outputs


class QuantOPRuntimeHook(RuntimeHook, metaclass=ABCMeta):
    """QuantOPRuntimeHook is an abstract class designed for executor
    customizing.

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """
    def __init__(self, operation: QuantableOperation, operation_meta: OperationMeta = None) -> None:
        if not isinstance(operation, QuantableOperation):
            raise TypeError(f'You are trying to bind a QuantRuntimeHook to a non-quantized operation {operation}.')
        super().__init__(operation, operation_meta)

    def pre_forward_hook(
        self,
        inputs: list,
        quant_inputs: list,
        quant_configs: List[TensorQuantizationConfig]
    ) -> list:
        return quant_inputs

    def post_forward_hook(
        self,
        outputs: list,
        quant_outputs: list,
        quant_configs: List[TensorQuantizationConfig]
    ) -> list:
        return quant_outputs


class BaseGraphExecutor(Callable, metaclass=ABCMeta):
    """PPQ Base Graph Executor.

    Args:
        Callable ([type]): [description]
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """
    def __init__(self, graph: BaseGraph) -> dict:
        self._graph = None
        self._graph_input_dictionary = None
        self._graph_output_dictionary = None
        self._executing_order = None
        self.load_graph(graph=graph)

    def load_graph(self, graph: BaseGraph) -> dict:
        self._graph = graph
        self._graph_input_dictionary = self._graph.inputs
        self._graph_output_dictionary = self._graph.outputs
        self._executing_order = self._graph.topological_sort()

    def prepare_input(self, inputs: Union[dict, list, torch.Tensor]):
        assert type(inputs) in (dict, list, torch.Tensor), \
            f'Input format misunderstood. Except either dict, list or tensor; while {type(inputs)} was given.'

        inputs_dictionary = self._graph.inputs
        if len(inputs_dictionary) == 0:
            assert inputs is None, 'Graph do not need any inputs. please set your inputs to be None.'
            return None

        if isinstance(inputs, torch.Tensor):
            assert len(inputs_dictionary) == 1, \
                'Graph needs more than one input, while only one tensor was given.'
            return {list(inputs_dictionary.keys())[0]: inputs}

        elif isinstance(inputs, list):
            assert len(inputs_dictionary) == len(inputs), \
                f'Inputs format misunderstood. Given inputs has '\
                f'{len(inputs)} elements, while graph needs {len(inputs_dictionary)}'
            return {key: inputs[idx] for idx, key in enumerate(inputs_dictionary)}

        elif isinstance(inputs, dict):
            assert len(inputs_dictionary) == len(inputs), \
                f'Inputs format misunderstood. Given inputs has '\
                f'{len(inputs)} elements, while graph needs {len(inputs_dictionary)}'
            return inputs

        else:
            raise Exception('Oops, you can never reach here.')

    @ abstractmethod
    def forward(
        self,
        inputs: Union[dict, list, torch.Tensor],
        output_names:List[str] = None,
        hooks: Dict[str, RuntimeHook] = None,
    ) -> List[torch.Tensor]:
        raise NotImplementedError('Please implement this function first.')

    @ abstractmethod
    def tracing_operation_meta(
        self,
        inputs: Union[dict, list, torch.Tensor],
        output_names:List[str] = None
    ) -> None:
        raise NotImplementedError('Please implement this function first.')

    def __call__(
        self,
        inputs: Union[dict, torch.Tensor],
        output_names:List[str] = None
    ) -> List[torch.Tensor]:
        return self.forward(inputs=inputs, output_names=output_names)

    def __str__(self) -> str:
        return (f'PPQ GraphExecuter Object: {self.__hash__()}')

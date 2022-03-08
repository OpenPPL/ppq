from typing import Any, List, Tuple

from ppq.core import (OperationQuantizationConfig, QuantizationStates,
                      TargetPlatform, TensorQuantizationConfig,
                      convert_any_to_torch_tensor)
from ppq.quantization.qfunction import BaseQuantFunction

from .base.command import (GraphCommand, GraphCommandType,
                           QuantizeOperationCommand, ReplaceOperationCommand,
                           ReplaceVariableCommand)
from .base.graph import Operation, Variable
from .processer import GraphCommandProcesser


class QuantableOperation(Operation):
    def __init__(
        self, 
        convert_from: Operation,
        quantize_config: OperationQuantizationConfig,
        platform: TargetPlatform
    ):

        # Simply copy all attributes from fp32 operation
        # inputs, outputs will be created by QuantableGraph
        super().__init__(
            op_type      = convert_from.type,
            inputs       = convert_from.inputs.copy(),
            outputs      = convert_from.outputs.copy(),
            attributes   = convert_from.attributes,
            name         = convert_from.name,
            platform     = platform
        )

        self._config            = quantize_config
        # meta data is a crucial attribute for quantization
        self._meta              = convert_from.meta_data
        self._dequantized       = False

    @ property
    def config(self) -> OperationQuantizationConfig:
        return self._config

    @ config.setter
    def set_config(self, config):
        """ we will update variable's during this function

        Args:
            config ([type]): [description]

        Raises:
            TypeError: [description]
            ExecError: [description]

        Returns:
            [type]: [description]
        """
        if not isinstance(config, OperationQuantizationConfig):
            raise TypeError(f'object {str(config)}({type(config)}) is not a acceptable config for operation {self.name}')
        self._config = config

    def baking_parameters(self, quant_func: BaseQuantFunction):
        for config, var in self.config_with_variable:
            if var.is_parameter and config.state in {QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE}:
                assert isinstance(var, QuantableVariable)
                assert len(var.dest_ops) == 1, f', Parameter {var.name} has {len(var.dest_ops)} destinations, '\
                    'Baking parameter that has more than 1 destinations will incur unexpected problems, '\
                    'PPQ does not support parameters with more than 1 related operation, reform your graph first.'
                var.value = quant_func(var.value, config)

                if config.state == QuantizationStates.ACTIVATED:
                    config.state = QuantizationStates.BAKED
                if config.state == QuantizationStates.PASSIVE:
                    config.state = QuantizationStates.PASSIVE_BAKED
        return self

    def store_parameter_value(self):
        for var, _ in zip(self.inputs + self.outputs, 
            self.config.input_quantization_config + self.config.output_quantization_config):
            if var.is_parameter:
                assert isinstance(var, QuantableVariable), 'Unexpected error.'
                # convert var.value to torch.Tensor
                # notice here we set device = None, this conversion will not change var.value.device anyway.
                # so that we can use var.value.device as a depoly device for stored_value
                var.stored_value = convert_any_to_torch_tensor(var.value, device='cpu')
        return self

    def dequantize(self, parameter_only: bool = False, expire_device: str = 'cpu'):
        if self._dequantized: return self
        for var, quant_config in zip(self.inputs + self.outputs, 
            self.config.input_quantization_config + self.config.output_quantization_config):
            if parameter_only and not var.is_parameter: continue
            quant_config.detail['Stored State'] = quant_config.state
            assert isinstance(var, QuantableVariable), f'Unexpected error with variable {var.name}.'
            if var.is_parameter:
                # convert var.value to torch.Tensor
                # notice here we set device = None, this conversion will not change var.value.device anyway.
                # so that we can use var.value.device as a depoly device for stored_value
                stored_value = convert_any_to_torch_tensor(var.value, device=expire_device)
                var.value = convert_any_to_torch_tensor(var.value, device=None)
                var.value = convert_any_to_torch_tensor(var.stored_value, device=var.value.device)
                var.stored_value = stored_value
            quant_config.state = QuantizationStates.DEQUANTIZED
        self._dequantized = True
        return self

    def restore_quantize_state(self, expire_device: str = 'cpu'):
        if not self._dequantized: return self
        for var, quant_config in zip(self.inputs + self.outputs, 
            self.config.input_quantization_config + self.config.output_quantization_config):
            if 'Stored State' in quant_config.detail:
                quant_config.state = quant_config.detail['Stored State']
                quant_config.detail.pop('Stored State')
                if var.is_parameter:
                    assert isinstance(var, QuantableVariable), 'Unexpected error.'
                    # convert var.value to torch.Tensor
                    # notice here we set device = None, this conversion will not change var.value.device anyway.
                    # so that we can use var.value.device as a depoly device for stored_value
                    stored_value = convert_any_to_torch_tensor(var.value, device=expire_device)
                    var.value = convert_any_to_torch_tensor(var.value, device=None)
                    var.value = convert_any_to_torch_tensor(var.stored_value, device=var.value.device)
                    var.stored_value = stored_value
        self._dequantized = False
        return self

    @ property
    def config_with_variable(self) -> List[Tuple[TensorQuantizationConfig, Variable]]:
        """
        Just a helper function.

        Returns:
            List[Tuple[TensorQuantizationConfig, Variable]]: [description]
        """
        ret = []
        for cfg, var in zip(self.config.input_quantization_config, self.inputs):
            ret.append((cfg, var))
        for cfg, var in zip(self.config.output_quantization_config, self.outputs):
            ret.append((cfg, var))
        return ret


class QuantableVariable(Variable):
    def __init__(self, convert_from: Variable) -> None:
        super().__init__(
            name      = convert_from.name,
            dest_ops  = convert_from.dest_ops.copy(),
            source_op = convert_from.source_op,
            value     = convert_from.value,
            is_parameter = convert_from.is_parameter)
        self._fp32_value = None
        if convert_from.value is not None:
            self._fp32_value = convert_any_to_torch_tensor(convert_from.value, device='cpu')

    @ property
    def stored_value(self) -> Any:
        return self._fp32_value
    
    @ stored_value.setter
    def stored_value(self, value: Any):
        self._fp32_value = value

    @ property
    def dest_op_configs(self) -> List[TensorQuantizationConfig]:
        _dest_op_configs, _dest_idx = [], self.dest_idx
        for idx, op in enumerate(self.dest_ops):
            if isinstance(op, QuantableOperation):
                _dest_op_configs.append(op.config.input_quantization_config[_dest_idx[idx]])
            else: _dest_op_configs.append(None)
        return _dest_op_configs

    @ property
    def dest_op_platfroms(self) -> List[TargetPlatform]:
        _dest_op_platforms = []
        for op in self.dest_ops:
            if op is not None:
                _dest_op_platforms.append(op.platform)
            else: _dest_op_platforms.append(TargetPlatform.FP32)
        return _dest_op_platforms

    @ property
    def source_op_config(self) -> TensorQuantizationConfig:
        if self.source_op is not None:
            if isinstance(self.source_op, QuantableOperation):
                return self.source_op.config.output_quantization_config[self.src_idx]
            else: return None
        return None

    @ property
    def source_op_platform(self) -> TargetPlatform:
        if self.source_op is None:
            return TargetPlatform.FP32
        else: return self.source_op.platform


class DeviceSwitchOP(Operation):
    """
    DeviceSwitch is a PPQ internal operation.
        This operation is inserted at platfrom's boundary
        for transfering data between devices.

    Args:
        Operation ([type]): [description]
    """
    def __init__(self, name: str,
                 inputs: List[Variable] = None, 
                 outputs: List[Variable] = None) -> None:
        super().__init__(
            attributes={},
            name=name, op_type='PPQDeviceSwitch',
            platform=TargetPlatform.UNSPECIFIED,
            inputs=inputs, outputs=outputs)


class QuantableGraph(GraphCommandProcesser):
    def process(self, command: GraphCommand) -> Any:
        if command.command_type == GraphCommandType.QUANTIZE_OPERATION:
            assert isinstance(command, QuantizeOperationCommand)
            return self.quantize_operation(
                command.op_name, command.target_platform, command.config)

    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            GraphCommandType.QUANTIZE_OPERATION,
        ]
    
    def quantize_operation(
        self, 
        operation_name: str,
        target_platform: TargetPlatform,
        quantization_config: OperationQuantizationConfig
    ) -> QuantableOperation:
        if operation_name not in self.graph.operations:
            raise KeyError(f'Operation {operation_name} is not in your graph, Please check your input.')

        if not TargetPlatform.is_quantized_platform(target_platform):
            raise ValueError(
                f'You are trying to quantize a operation({operation_name})'\
                f' to target platform {target_platform}, however it is an non-quantized platform.')

        operation = self._graph.operations[operation_name]

        quantized_operation = QuantableOperation(
            convert_from=operation,
            quantize_config=quantization_config,
            platform=target_platform,
        )

        # calling other chain responser to replace operation with quantized one.
        if self._next_command_processer is None:
            raise RuntimeError(
                'To replace a opeartion, your processer chain must have a GraphMorpher Processer.')
        self._next_command_processer(ReplaceOperationCommand(operation_name, quantized_operation))

        # replace all related variable with quantable one.
        for var in quantized_operation.inputs + quantized_operation.outputs:
            if isinstance(var, QuantableVariable): continue
            self._next_command_processer(
                ReplaceVariableCommand(
                    var_name=var.name, 
                    replace_to=QuantableVariable(convert_from=var)
                )
            )
        quantized_operation.store_parameter_value()

    def dequantize_opeartion(
        self,
        operation_name: str
    ) -> Operation:
        if operation_name not in self.graph.operations:
            raise KeyError(f'Operation {operation_name} is not in your graph, Please check your input.')
        operation = self._graph.operations[operation_name]
        if not isinstance(operation, QuantableOperation): return operation
        else: return operation.dequantize()

    def dequantize_graph(self):
        """
            一个方便懒人的函数
        """
        for operation in self.graph.operations.values():
            if isinstance(operation, QuantableOperation):
                operation.dequantize()
    
    def restore_quantize_state(self):
        """
            一个方便懒人的函数
        """
        for operation in self.graph.operations.values():
            if isinstance(operation, QuantableOperation):
                operation.restore_quantize_state()
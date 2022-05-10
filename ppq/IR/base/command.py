from enum import Enum

from ppq.core import OperationQuantizationConfig, TargetPlatform

from .graph import Operation, Variable


class GraphCommandType(Enum):
    # 图上权重部署到 GPU(tensor)，由 RunnableGraph 进行处理
    # deploy graph weights to GPU
    DEPLOY_TO_CUDA = 1

    # 图上权重部署到 CPU(tensor)，由 RunnableGraph 进行处理
    # deploy graph weights to CPU, in tensor format
    DEPLOY_TO_CPU = 2

    # 图上权重部署到 CPU(tensor)，由 RunnableGraph 进行处理
    # deploy graph weights to CPU, in array format
    DEPLOY_TO_NUMPY = 3

    # 量化一个指定OP，同时将所有关联的 variable 转换为量化 variable
    # quantize a specified operation, and converts all connected
    # variables to quantable variables
    QUANTIZE_OPERATION = 5

    # 将一个OP的量化暂时解除，同时将所有关联的 variable 解除量化
    # deactivate quantization state of an op temporarily,
    # and deactivated all related variables
    DISABLE_OPERATION_QUANTIZATION = 6
    # 将一个OP的量化状态恢复
    # restore quantization state of a dequantized op
    RESTORE_OPERATION_QUANTIZATION = 7

    # 格式化 CLIP 算子，将不同行为的 CLIP 行为统一
    # regularize Clip operator
    FORMAT_CLIP = 9
    # 格式化 PAD 算子，将不同行为的 PAD 行为统一
    # regularize Pad operator
    FORMAT_PAD = 10
    # 格式化 GATHER 算子，将 index 参数（由input输入）植入算子属性
    # regularize gather operator
    FORMAT_GATHER = 11
    # 格式化 CAST 算子，统一 CAST 参数到 PPQ.core.DataType
    # regularize Cast operator
    FORMAT_CAST = 12
    # 格式化所有常量输入，尝试将他们转换为int32的
    # regularize all constant inputs
    FORMAT_INT64_CONSTANT = 13
    # 移除所有孤立节点
    # remove all isolated operators
    DELETE_ISOLATED = 14
    # 尝试用 add 替换 sub
    # try use Add op to replace Sub op
    REPLACE_SUB = 15
    # 将所有参数变量进行分裂（只允许一个 dest_op ）
    # split variables and each variable is allowed to be used by only one operator
    FORMAT_PARAMETERS = 16

    # 用一个新的算子替换一个原来的
    # replace an old op with a new one
    REPLACE_OP = 17
    # 用一个新的 var 替换一个原来的
    # replace an old variable with a new one
    REPLACE_VAR = 18
    # 移除一个算子的输入参数，只能移除 Parameter
    # remove input parameters of am operator
    REMOVE_INPUT = 19

    # 图遍历模式匹配（要求返回路径）
    # graph traversal pattern matching(return paths)
    TRAVERSAL_PATTERN_MATCHING = 20
    # 图遍历模式匹配（仅要求返回点集）
    # graph traversal pattern matching(return op set)
    TRAVERSAL_OPSET_MATCHING = 21
    # 激活函数匹配
    # activation op matching
    ACTIVATION_MATCHING = 22
    # Concat 匹配
    # Concat op matching
    CONCAT_MATCHING = 23

    # 插入 Device Switcher
    # insert switcher
    INSERT_SWITCHER = 25
    # 移除 Device Switcher
    # remove switcher
    REMOVE_SWITCHER = 26

    # 融合图中的 计算层 与 BN
    # fuse Computing layer and BN
    FUSE_BN = 27
    # 删除图中的 Constant Input
    # remove constant input
    FORMAT_CONSTANT_INPUT = 28
    # 将 opset1 的 slice 弄成 opset 11 的
    FORMAT_SLICE = 29
    # 从一个指定位置将图截断
    TRUNCATE_ON_VAR = 30

class GraphCommand():
    def __init__(self, command_type: GraphCommandType, **kwargs) -> None:
        assert isinstance(command_type, GraphCommandType), \
            f'Command Type must be a GraphCommandType object, but {type(command_type)} received.'
        self.command_type = command_type
        self.kwargs = kwargs

    def __str__(self) -> str:
        return f'GraphCommand object {self.__hash__()},\t Command type: {self.command_type},\t Args:{self.kwargs}'


class GraphDeployCommand(GraphCommand):
    def __init__(self, device: str) -> None:
        if device.startswith('cuda'):
            super().__init__(GraphCommandType.DEPLOY_TO_CUDA)
        elif device.startswith('cpu'):
            super().__init__(GraphCommandType.DEPLOY_TO_CPU)
        else:
            raise ValueError(f'Device type {device} not understand.')
        self._device = device

    def __str__(self) -> str:
        return super().__str__()


class QuantizeOperationCommand(GraphCommand):
    def __init__(self, op_name: str, target_platform: TargetPlatform, config: OperationQuantizationConfig) -> None:
        super().__init__(command_type=GraphCommandType.QUANTIZE_OPERATION)
        self.op_name = op_name
        self.target_platform = target_platform
        self.config = config


class ReplaceOperationCommand(GraphCommand):
    def __init__(self, op_name: str, replace_to: Operation) -> None:
        super().__init__(command_type=GraphCommandType.REPLACE_OP)
        self.op_name = op_name
        self.replace_to = replace_to


class ReplaceVariableCommand(GraphCommand):
    def __init__(self, var_name: str, replace_to: Variable) -> None:
        super().__init__(command_type=GraphCommandType.REPLACE_VAR)
        self.op_name = var_name
        self.replace_to = replace_to


class TruncateGraphCommand(GraphCommand):
    def __init__(self, var: Variable, mark_as_output: bool) -> None:
        super().__init__(command_type=GraphCommandType.TRUNCATE_ON_VAR)
        self.var = var
        self.mark_as_output = mark_as_output

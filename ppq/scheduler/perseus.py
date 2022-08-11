from typing import Dict, List, Set

from ppq.core import TargetPlatform, ppq_warning
from ppq.IR import BaseGraph, Operation, SearchableGraph

from .base import GraphDispatcher
from .core import (DEFAULT_SOCKET_CREATOR, DEFAULT_SOCKET_TABLE, OpSocket,
                   VProperty, VType)


class Perseus(GraphDispatcher):
    """ 英仙座 是 PPQ 0.6.5 版本引入的全新调度器

    基于新的 OpSocket, VProperty, VLink 数据类型抽象，
    英仙座调度器能够以一种可扩展的、可控制的方式切分你的计算图。
    
    英仙座调度器基于静态图分析与数值追踪, 
    借由 OpSocket 定义的 "接线器" 算子模型, 英仙座调度器可以对
    Onnx中数据流的来源进行追踪，并进一步求解关于节点的数值传递闭包。
    这一过程完全是静态完成的，不需要送入数据执行。
    
    对于一个简单的 Reshape 算子而言，接线器模型定义了该算子内部的数据流动情况：
    (对于Reshape算子而言, 其输出结果与第一个输入之间存在数值关联, 而与第二个输入直接不存在直接关联)
    
      Input(VALUE)   Shape(SOI)
        |              |
    * ---------------------- *
    |   |                    |
    |   * ----- *   Reshape  |
    |           |            |
    * ---------------------- *
                |
               Out(VALUE)

    接线器模型在算子内部定义了这样的输入 - 输出关系网，从而调度器可以使用这样的关系网来划分子图。
    
    Args:
        GraphDispatcher (_type_): _description_
    """

    def __init__(self, graph: BaseGraph, verbose: bool = False) -> None:
        """ 初始化英仙座调度器
        在这一过程中，英仙座调度器将为每一个算子生成它的 OpSocket 接线器
        一旦算子接线器初始化完毕，你将不能对图结构进行进一步修改，否则调度将会失败
        
        在这一过程中，英仙座将为 onnx 算子生成预定义的 onnx 算子接线器，
        所有被列举在 ppq/scheduler/core/default.py 中的算子均有预定义的算子接线器实现
        支持 opset 1 ~ 18
        
        如果你的网络具有自定义算子，或存在未在 default.py 列举的 onnx 算子
        英仙座会为他们生成默认接线器，请知悉：这可能导致错误
        
        你应当为你的新算子注册新的接线器，从而英仙座调度器能够使用新定义的接线器模型完成调度
        当英仙座调度器为你未知的算子生成默认接线器时，它将给出警报
        """ 
        self.sockets   = {}
        self.verbose   = verbose
        self.graph     = graph
        self._search_engine = SearchableGraph(graph)
        self._precomputed_op_fanout = {}
        self._precomputed_op_fanin  = {}

        for op in self.graph.operations.values():
            if op.type in DEFAULT_SOCKET_TABLE:
                socket = DEFAULT_SOCKET_TABLE[op.type](op)

                fanin, fanout = set(), set()
                for link in socket.links:
                    source_var  = op.inputs[link.source_idx]
                    dest_var    = op.outputs[link.dest_idx]
                    source_op   = source_var.source_op
                    dest_ops    = dest_var.dest_ops
                    if source_op is not None:
                        fanin.add(source_op)
                    for dest_op in dest_ops:
                        fanout.add(dest_op)
                self._precomputed_op_fanin[op.name], self._precomputed_op_fanout[op.name] = fanin, fanout
                self.sockets[op.name] = socket

            else:
                ppq_warning(f'Perseus Dispatcher do not konw how to dispatch op {op.name}({op.type}), '
                            f'cause optype {op.type} is unsupported now. '
                            f'Perseus will initialize a default socket for it, '
                            'which might cause some error in execution. '
                            'You are supposed to register a socket handler for this type instead.')
                socket = DEFAULT_SOCKET_CREATOR(op)
                self._precomputed_op_fanin[op.name]   = set(self.graph.get_upstream_operations(op))
                self._precomputed_op_fanout[op.name]  = set(self.graph.get_downstream_operations(op))
                self.sockets[op.name] = socket
        super().__init__()


    def solve_transitive_closure(
        self, sources: List[Operation], 
        recursive: bool = True) -> Set[Operation]:
        """ 从指定节点出发解传递闭包，该方法是量化子图切分的核心函数
        传递闭包是一个节点的集合，以 C(x) = x_1, x_2, x_3 ... 进行表示
        以 C(x) 表示从节点 x 出发得到的传递闭包，则有以下结论
            1. C(x) 必须包含 x
            2. C(x) 中任意节点的传递闭包等于 C(x)

        在 Onnx 定义的神经网络中存在两类数据：
            其一是从 Gemm, Conv 等计算节点出发的数据，他们往往是需要量化的；
            以及从 Shape, TopK 出发的数据，他们往往是不能被量化的；

        为此 PPQ 将从网络中所有计算节点出发，求解关于计算节点的传递闭包 A
        而后从 Shape, TopK 等节点出发，求解非计算节点的传递闭包 B
        
        集合 A - B 中的节点将被量化，也被称为量化区节点
        集合 A * B 中的节点将被称为冲突区节点，默认不量化
        集合 B 中的节点将被称为 SOI 节点，不量化且调度到 Cpu 执行
        集合 A, B 之外的节点为未知区域节点，默认不量化
        
        PPQ 使用朴素递归求解传递闭包，对于神经网络这种稀疏图而言，
        
        其时间复杂度大约为O(n)
        
        注意一些细节问题，TopK 节点有两个输出：前K大的值和前K大的index，
        如果从 TopK 节点 出发求传递闭包，只会寻找到其第一个输出分支上的节点。
        
        节点上的链接关系由算子接线器进行定义；算子接线器是一种 PPQ 内置的抽象数据结构，
        参考：ppq/scheduler/core/default.py

        Args:
            sources (List[Operation]): _description_
            recursive (bool, optional): _description_. Defaults to True.

        Returns:
            Set[Operation]: _description_
        """
        if isinstance(sources, Operation): sources = [sources]

        closure = set(sources)
        closure_size = len(sources)
        closure_is_changing = True

        # loop until convergence
        while closure_is_changing:
            closure_is_changing = False
            b_extension = self.parse_transitive_fanin(closure)
            f_extension = self.parse_transitive_fanout(closure)
            closure.update(b_extension)
            closure.update(f_extension)
            if len(closure) != closure_size and recursive:
                closure_is_changing = True
            closure_size = len(closure)

        return set(closure)

    def mark_quantable_op(self) -> Set[Operation]:
        """追踪图中所有可量化节点，即求解所有计算节点的传递闭包

        Returns:
            Set[Operation]: _description_
        """
        sources = [op for op in self.graph.operations.values() if op.is_computing_op]
        return self.solve_transitive_closure(sources)

    def mark_non_quantable_op(self) -> Set[Operation]:
        """追踪图中所有不可量化节点，即求解所有不可量化节点的传递闭包
        不可量化节点种类繁多，常见的节点包括 Shape, Topk, NMS
        有一些节点的输入不能量化，如 Reshape, Slice 等
        所有不可量化节点由 opsocket 具体定义给出

        Raises:
            KeyError: _description_

        Returns:
            Set[Operation]: _description_
        """
        # initialize non quantable group.
        sources = set()
        for op in self.graph.operations.values():
            if op.name not in self.sockets: 
                raise KeyError(f'Can not find Opsocket for {op.name}, graph has been modified.')
            socket = self.sockets[op.name]
            assert isinstance(socket, OpSocket)

            for ocls, ovar in zip(socket.cls_output, op.outputs):
                if VType(ocls).non_quantable():
                    for dop, dix in zip(ovar.dest_ops, ovar.dest_idx):
                        if self.opsocket(dop).cls_input[dix] == VProperty.VALUE:
                            sources.add(dop)

            for icls, ivar in zip(socket.cls_input, op.inputs):
                if VType(icls).non_quantable() and ivar.source_op != None:
                    sources.add(ivar.source_op)
        return self.solve_transitive_closure(sources)

    def parse_transitive_fanout(self, parsing_from: List[Operation]) -> Set[Operation]:
        """从指定节点出发，寻找所有扇出节点
        注意此函数与 graph.get_downstream_operations 的区别
        graph.get_downstream_operations 函数将直接返回目标节点的所有下游节点
        而此函数将只返回下游节点中与当前节点存在数值传递关系的节点
        
        例如对于子图 conv - shape
            graph.get_downstream_operations(conv) 将返回 shape 节点
            parse_transitive_fanout(conv) 将返回空
        """
        fanout = self._search_engine.opset_matching(
            sp_expr=lambda op: op in parsing_from,
            rp_expr=lambda f, t: f in self._precomputed_op_fanin[t.name],
            ep_expr=None, direction='down')
        return fanout

    def parse_transitive_fanin(self, parsing_from: List[Operation]) -> Set[Operation]:
        """从指定节点出发，寻找所有扇出节点
        注意此函数与 graph.get_upstream_operations 的区别
        graph.get_upstream_operations 函数将直接返回目标节点的所有上游节点
        而此函数将只返回上游节点中与当前节点存在数值传递关系的节点
        
        例如对于子图 conv - shape
            graph.get_upstream_operations(shape) 将返回 conv 节点
            parse_transitive_fanin(shape) 将返回空
        """
        fanin = self._search_engine.opset_matching(
            sp_expr=lambda op: op in parsing_from,
            rp_expr=lambda f, t: t in self._precomputed_op_fanin[f.name],
            ep_expr=None, direction='up')
        return fanin

    def dispatch(self, graph_input_cls: List[VProperty] = None, 
                 graph_output_cls: List[VProperty] = None) -> Dict[str, TargetPlatform]:
        """对当前图执行默认算子调度逻辑

        在 Onnx 定义的神经网络中存在两类数据：
            其一是从 Gemm, Conv 等计算节点出发的数据，他们往往是需要量化的；
            以及从 Shape, TopK 出发的数据，他们往往是不能被量化的；

        为此 PPQ 将从网络中所有计算节点出发，求解关于计算节点的传递闭包 A
        而后从 Shape, TopK 等节点出发，求解非计算节点的传递闭包 B
        
        集合 A - B 中的节点将被量化，也被称为量化区节点
        集合 A * B 中的节点将被称为冲突去节点，默认不量化
        集合 B 中的节点将被称为 SOI 节点，不量化且调度到 Cpu 执行
        集合 A, B 之外的节点为未知区域节点，默认不量化
        
        PPQ 使用朴素递归求解传递闭包，对于神经网络这种稀疏图而言，
        
        其时间复杂度大约为O(n)

        你可以额外指定图输入和输出的属性，如果有些输出值不想量化，你可以将其指定为
        OType.NONQUANTABLE，则该数据流上所有的算子都不会被量化（未实现）

        Returns:
            Dict[str, OType]: _description_
        """
        computing_ops     = self.mark_quantable_op()
        non_quantable_ops = self.mark_non_quantable_op()

        dispatching_table = {}
        for op in self.graph.operations.values():
            dispatching_table[op.name] = TargetPlatform.FP32
        
        for op in self.graph.operations.values():
            if op in computing_ops:
                dispatching_table[op.name] = TargetPlatform.UNSPECIFIED

        for op in self.graph.operations.values():
            if op in non_quantable_ops:
                if op.name in dispatching_table and dispatching_table[op.name] == TargetPlatform.UNSPECIFIED:
                    dispatching_table[op.name] = TargetPlatform.FP32
                else: dispatching_table[op.name] = TargetPlatform.SHAPE_OR_INDEX

        # if op has an non-quantable output, force it to be non-quantable op
        for op in self.graph.operations.values():
            socket = self.opsocket(op)
            for ocls in socket.cls_output:
                if ocls in {VProperty.ATTRIB, VProperty.SOI}:
                    dispatching_table[op.name] = TargetPlatform.SHAPE_OR_INDEX
                    break
                if ocls in {VProperty.LOGICAL}:
                    dispatching_table[op.name] = TargetPlatform.FP32
                    break

        return dispatching_table

    def opsocket(self, op: Operation) -> OpSocket:
        return self.sockets[op.name]

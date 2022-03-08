from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Iterable, Iterator, List, Set, Union

from .base.command import GraphCommand, GraphCommandType
from .base.graph import BaseGraph, Operation
from .processer import GraphCommandProcesser


class PointPattern(Callable, metaclass=ABCMeta):
    @ abstractmethod
    def match(self, operation: Operation) -> bool: pass

    def __call__(self, operation: Operation) -> bool:
        return self.match(operation)


class RelayPattern(Callable, metaclass=ABCMeta):
    @ abstractmethod
    def match(self, from_where: Operation, to_where: Operation) -> bool: pass

    def __call__(self, from_where: Operation, to_where: Operation) -> bool:
        return self.match(from_where, to_where)


class Path(Iterable):
    def __init__(self, operation: Operation = None) -> None:
        self._container = deque()
        if operation is not None: self._container.append(operation)
    
    def append(self, op: Operation):
        self._container.append(op)
        return self

    def append_left(self, op: Operation):
        self._container.appendleft(op)
        return self

    def __iter__(self) -> Iterator[Operation]:
        return self._container.__iter__()
    
    def __getitem__(self, index: int) -> Operation:
        return self._container[index]
    
    def __str__(self) -> str:
        return ''.join([str(_) for _ in self._container.__str__()])
    
    def tolist(self) -> List[Operation]:
        return list(self._container)


class OperationSet(set):
    def __init__(self) -> None:
        super().__init__()
    
    def add(self, element: Operation):
        if not isinstance(element, Operation):
            raise TypeError('Operation Set can only contains opeartion instance.')
        super().add(element)
        return self

    def __iter__(self) -> Iterator[Operation]:
        return super().__iter__()
    
    def filter(self, condition: PointPattern):
        removing = []
        for item in self:
            if condition(item):
                removing.append(item)
        for item in removing:
            self.remove(item)

class TraversalCommand(GraphCommand):
    def __init__(self,
        sp_expr: Union[PointPattern, Callable],
        rp_expr: Union[RelayPattern, Callable],
        ep_expr: Union[PointPattern, Callable],
        direction: str = 'down',
        matching_type: str = 'path'
    ) -> None:
        """
            TraversalCommand 是一个用于表示图检索指令的结构体
            一个最简单的例子：
                sp_expr = lamdba x: x.type == 'Conv'
                rp_expr = lamdba x: x.type == 'Relu'
                ep_expr = lamdba x: x.type == 'Conv'
                limitation = None
                direction = 'down'
            该指令检索出从任意 Conv 出发，到任意 Conv 的所有可行路径
            其中路径上含有任意多个 Relu 节点，并且只能包含 Relu

            TraversalCommand is a graph search class
            Example:
                sp_expr = lamdba x: x.type == 'Conv'
                rp_expr = lamdba x: x.type == 'Relu'
                ep_expr = lamdba x: x.type == 'Conv'
                limitation = None
                direction = 'down'
            this Command searchs paths which starts with any 
            Conv op, visits any number of Relu ops, and finally
            ends with any Conv op
        Args:
            sp_expr (Union[PointPattern, Callable]): 
                start point expression, 用于匹配检索起点的表达式
            rp_expr (Union[RelayPattern, Callable]):
                relay point expression, 用于匹配检索中继点的表达式
            ep_expr (Union[PointPattern, Callable]):
                end point expression, 用于匹配检索终点的表达式
            direction (str, optional):
                图检索方向，up, down. Defaults to 'down'.
            matching_type (str, optional)
                指定匹配模式，可以指定为 path，则系统匹配完整路径
                指定为 opset，则系统只匹配涉及到的节点
        """
        self._direction = direction
        self._sp_expr     = sp_expr
        self._rp_expr     = rp_expr
        self._ep_expr     = ep_expr
        if matching_type == 'path':
            super().__init__(GraphCommandType.TRAVERSAL_PATTERN_MATCHING)
        elif matching_type == 'opset':
            super().__init__(GraphCommandType.TRAVERSAL_OPSET_MATCHING)
        else:
            raise ValueError('PPQ only support "opset" matching and "path" matching for now.')
            

    @ staticmethod
    def complie(query: str):
        """
        compile 函数把一个查询字符串编译成一个 TraversalCommand。
        我们还没有具体实现这个函数，但我们已经定义好了语法：

        查询字符串应该是下面的格式：
        SELECT  ["START"|"END"|"PATH"]
        FROM    [START PATTERN CLAUSE]
        TO      [END PATTERN CLAUSE]
        THROUGH [RELAY PATTERN CLAUSE]
        WHERE   [PATH PATTERN CLAUSE]

        例如:
        SELECT  "START"
        FROM    Conv, Gemm
        TO      Relu
        THROUGH Any
        WHERE   Path.length < 3
        语句将返回从 Conv, Gemm 出发，到 Relu 的所有可能路径
        从中筛选出路径长度小于 3 的，并且将其中的所有起点作为集合并返回

        SELECT  "PATH"
        FROM    Conv
        TO      Relu, Clip
        THROUGH Any
        WHERE   Path.length < 2
        语句将返回从 Conv 出发，到 Relu, Clip 的所有可能路径
        从中筛选出路径长度小于 2 的，并且将路径本身作为结果返回

        Args:
            query (str): [description]

        Raises:
            TypeError: [description]
            KeyError: [description]

        Returns:
            [type]: [description]
        """
        pass


class SearchableGraph(GraphCommandProcesser):
    """
        PPQ Graph Searching Engine.

    Args:
        GraphCommandProcesser ([type]): [description]
    """
    def __init__(self, graph: Union[BaseGraph, Callable]) -> None:
        super().__init__(graph)
        self._cache = {}

    def process(self, command: GraphCommand) -> Any:
        if isinstance(command, TraversalCommand):
            if command.command_type == GraphCommandType.TRAVERSAL_PATTERN_MATCHING:
                return self.path_matching(
                    sp_expr=command._sp_expr,
                    rp_expr=command._rp_expr,
                    ep_expr=command._ep_expr,
                    direction=command._direction)
            else:
                return self.opset_matching(
                    sp_expr=command._sp_expr,
                    rp_expr=command._rp_expr,
                    ep_expr=command._ep_expr,
                    direction=command._direction)
        else:
            raise TypeError(
                'To execute a traversal-based pattern matching, a TraversalCommand is required here.\n' 
                'Initialize your own TraversalCommand instance for invoking '
                'this function rather than using plain GraphCommand Please.')

    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            GraphCommandType.TRAVERSAL_PATTERN_MATCHING,
            GraphCommandType.TRAVERSAL_OPSET_MATCHING,
            GraphCommandType.CONCAT_MATCHING,
            GraphCommandType.ACTIVATION_MATCHING,
        ]

    def _path_matching(
        self,
        start_point: Operation,
        rp_expr: RelayPattern,
        ep_expr: PointPattern,
        direction: str = 'up'
    ) -> List[Path]:
        # memoization based searching.
        if start_point in self._cache: return self._cache[start_point]

        # find next operations with given direction
        if direction == 'up': following_ops = self.graph.get_upstream_operations(start_point)
        else: following_ops = self.graph.get_downstream_operations(start_point)

        ret_collection = []
        for op in following_ops:
            # if operation is a valid end point, add it to path and stop further searching.
            if ep_expr(op):
                ret_collection.append(Path(start_point).append(op))

            else:
                # if operation is not a valid relay point, end searching here.
                if not rp_expr(start_point, op): continue

                # searching following operations.
                for path in self._path_matching(start_point=op, rp_expr=rp_expr, 
                    ep_expr=ep_expr, direction=direction):
                    ret_collection.append(path.append_left(start_point))

        self._cache[start_point] = ret_collection
        return ret_collection

    def _opset_matching(
        self,
        start_point: Operation,
        rp_expr: RelayPattern,
        ep_expr: PointPattern,
        direction: str = 'up'
    ) -> OperationSet:
        
        # memoization based searching.
        if start_point in self._cache: return self._cache[start_point]
        
        # begin searching.
        ret_collection = OperationSet()

        # find next operations with given direction
        if direction == 'up': following_ops = self.graph.get_upstream_operations(start_point)
        else: following_ops = self.graph.get_downstream_operations(start_point)

        for op in following_ops:
            # if operation is a valid end point, add it to path and stop further searching.
            if ep_expr(op):
                ret_collection.update([start_point, op])

            else:
                # if operation is not a valid relay point, end searching here.
                if not rp_expr(start_point, op): continue

                # searching following operations.
                further_result = self._opset_matching(
                    start_point=op, rp_expr=rp_expr, 
                    ep_expr=ep_expr, direction=direction)

                if len(further_result) > 0:
                    ret_collection.update(further_result.add(start_point))

        self._cache[start_point] = ret_collection
        return ret_collection

    def path_matching(
        self,
        sp_expr: Callable, 
        rp_expr: Callable,
        ep_expr: Callable,
        direction: str = 'up'
        ) -> List[Path]:
        """
        path_matching is used for path searching on the
        graph, and complete paths will be returned. Note
        that it's posssible for results to overflow when
        there are numerous matchings

        path_matching 用于图检索，匹配完整路径
            一个最简单的例子：
                sp_expr = lamdba x: x.type == 'Conv'
                rp_expr = lamdba x, y: y.type == 'Relu'
                ep_expr = lamdba x: x.type == 'Conv'
                limitation = None
                direction = 'down'
            该指令检索出从任意 Conv 出发，到任意 Conv 的所有可行路径
            其中路径上含有任意多个 Relu 节点，并且只能包含 Relu
            
            你需要注意到 rp_expr 是一个二元表达式，其中 x 代表起点，y 代表终点
        Args:
            sp_expr (Union[Pattern, Callable]): 
                start point expression, 用于匹配检索起点的表达式
            rp_expr (Union[Pattern, Callable]):
                relay point expression, 用于匹配检索中继点的表达式
            ep_expr (Union[Pattern, Callable]):
                end point expression, 用于匹配检索终点的表达式
            limitation (Union[Limitation, Callable]):
                用于过滤路径的表达式
            direction (str, optional):
                图检索方向，search direction: up, down. Defaults to 'down'.
            greedy (bool, optional):
                是否执行贪心匹配，设置为True则尝试匹配直到最后一个end point.
                whether to search greedily
        """
        if direction not in {'up', 'down'}:
            raise KeyError("traversal_direction must be one of {'up', 'down'}")

        _ret_collection = []
        for operation in self.graph.operations.values():

            # filter out operation which can not be a start point.
            if not sp_expr(operation): continue

            # match patterns, add patterns towards result collection.
            matchings = self._path_matching(
                start_point=operation, rp_expr=rp_expr, 
                ep_expr=ep_expr, direction=direction)
            _ret_collection.extend(matchings)

        # clear cache
        self._cache.clear()

        # use limitation to filter out invalid path.
        return _ret_collection
    
    def opset_matching(
        self,
        sp_expr: Callable, 
        rp_expr: Callable,
        ep_expr: Callable,
        direction: str = 'up'
        ) -> OperationSet:
        """
        opset_matching is used for operator set searching,
        it returns relevant op set instead of specific paths,
        should be used when results of path_matching overflow

        opset_matching 用于图检索，匹配算子集合（无序）
            一个最简单的例子：
                sp_expr = lamdba x: x.type == 'Conv'
                rp_expr = lamdba x, y: y.type == 'Relu'
                ep_expr = lamdba x: x.type == 'Conv'
                limitation = None
                direction = 'down'
            该指令检索出从任意 Conv 出发，经过任意多个Relu，到任意 Conv 的所有相关算子
            注意返回结果是无序的，相比于 path matching，opset matching 的性能更高。
            
            你需要注意到 rp_expr 是一个二元表达式，其中 x 代表起点，y 代表终点
            
            在极端情况下，path matching 的结果是无法返回的（由于其结果数可能过多）
            此时应当使用 opset matching 
        Args:
            sp_expr (Union[Pattern, Callable]): 
                start point expression, 用于匹配检索起点的表达式
            rp_expr (Union[Pattern, Callable]):
                relay point expression, 用于匹配检索中继点的表达式
            ep_expr (Union[Pattern, Callable]):
                end point expression, 用于匹配检索终点的表达式
            limitation (Union[Limitation, Callable]):
                用于过滤路径的表达式
            direction (str, optional):
                图检索方向，up, down. Defaults to 'down'.
            greedy (bool, optional):
                是否执行贪心匹配，设置为True则尝试匹配直到最后一个end point.
        """
        ret_collection, candidates = OperationSet(), OperationSet()
        for operation in self.graph.operations.values():
            # filter out operation which can not be a start point.
            if sp_expr(operation): candidates.add(operation)

        for operation in candidates:
            # match patterns, add patterns towards result collection.
            partial_matchings = self._opset_matching(
                start_point=operation, rp_expr=rp_expr, 
                ep_expr=ep_expr, direction=direction)

            if len(partial_matchings) > 0:
                ret_collection.update(partial_matchings)

        # clear cache
        self._cache.clear()
        return ret_collection

    def activation_matching(
        self, start_op_types: Set[str],
        end_types: Set[str],
    ) -> Dict[str, List[str]]:

        matchings = self.path_matching(
            sp_expr   = lambda op: op.type in start_op_types,
            rp_expr   = lambda x, y: False, # must have no relay operation.
            ep_expr   = lambda op: op.type in end_types,
            direction = 'down'
        )

        activation_matchings = defaultdict(list)
        for path in matchings:
            op, act = path[0], path[-1]
            activation_matchings[op.name].append(act.name)
        return dict(activation_matchings)

    def concat_matching(
        self, relay_pattern: Callable,
        end_pattern: Callable) -> Dict[str, List[str]]:
        matchings = self.path_matching(
            sp_expr = lambda op: op.type in {'Concat'},
            rp_expr = relay_pattern,
            ep_expr = end_pattern,
            direction = 'up'
        )

        concat_matchings = defaultdict(list)
        for path in matchings:
            concat, op = path[0], path[-1]
            concat_matchings[concat.name].append(op.name)
        return dict(concat_matchings)

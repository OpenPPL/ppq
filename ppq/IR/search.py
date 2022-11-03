from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Iterable, Iterator, List, Set, Union

from ppq.core import ppq_warning

from .base.command import GraphCommand, GraphCommandType
from .base.graph import BaseGraph, Operation
from .processer import GraphCommandProcessor


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

    def copy(self):
        c = Path()
        c._container = self._container.copy()
        return c


class OperationSet(set):
    def __init__(self) -> None:
        super().__init__()

    def add(self, element: Operation):
        if not isinstance(element, Operation):
            raise TypeError('Operation Set can only contains operation instance.')
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
            this Command searches paths which starts with any
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
        """compile 函数把一个查询字符串编译成一个 TraversalCommand。 我们还没有具体实现这个函数，但我们已经定义好了语法：

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
        """
        pass


class GraphPattern():

    def __init__(self, node_patterns: List[Callable], edges: List[List[int]]) -> None:
        """Pattern Tree 是一个用来表示图模式的结构体 这将在图中检索任意一个子图.

        你将使用 Graph Pattern 定义你的子图结构
        使用 patterns 确定每一个节点需要满足的条件
        使用 edges 将节点们彼此相连从而构成图结构
        
        构成的图必须可以进行拓扑排序，不可以检索有环结构，不可以检索不连通结构
        例子：
            pattern = ['Conv', 'Conv', 'Conv'],
            edges = [[0, 1], [1, 2], [0, 2]]

        描述了一个类似这样的树形结构:
        
            Conv -+- Conv -+- Conv
                  |        |
                  ----------

        第二个例子:
        pt = PatternTree(
                patterns = [lambda x: x.is_computing_op, 'Softplus', 'Tanh', 'Mul']
                edges = [[0, 1], [1, 2], [2, 3], [0, 3]])

            pt create an abstract tree pattern of:
                                            --- 'Softplus'   ---   'Tanh' --
            lambda x: x.is_computing_op --- +                              + --- 'Mul'
                                            ---     ---     ---    ---    --

        错误的例子:
            pattern = ['Conv', 'Conv', 'Conv'],
            edges = [[0, 1], [1, 2], [2, 0]]
        因为图中存在循环结构而无法检索
        """
        for idx, node_pattern in enumerate(node_patterns):
            if isinstance(node_pattern, str):
                node_patterns[idx] = TypeExpr(node_pattern)
            elif not isinstance(node_pattern, Callable):
                raise TypeError(f'Can not create Pattern with node pattern {str(node_pattern)} it is not callable.')

        for edge in edges:
            if not isinstance(edge, tuple) and not isinstance(edge, list):
                raise TypeError(f'Can not create Pattern with edge {str(edge)} it is not tuple or list.')
            if len(edge) != 2:
                raise ValueError(f'Can not create Pattern with edge {str(edge)} '
                                 f'it should contains exact 2 elements, however {len(edge)} was given.')
            sp, ep = edge
            if not isinstance(sp, int) or not isinstance(ep, int):
                raise TypeError(f'Can not create Pattern was given edge {[str(sp), str(ep)]}, '
                                'expect int value here.')
        
        self.order, self.output_table, self.input_table = self.compile(node_patterns=node_patterns, edges=edges)
        self.argsort_order = sorted([(_, idx) for idx, _ in enumerate(self.order)])
        self.argsort_order = [idx for _, idx in self.argsort_order]
        self.node_patterns = node_patterns


    def compile(self, node_patterns: List[Callable], edges: List[List[int]]):
        """ Pattern Compile.
        1. Do topological Sort on given pattern.
        2. Reorganize pattern by topological order.
        """

        # prepare for topological sort
        visited = [False for _ in node_patterns]
        num_of_inputs = [0 for _ in node_patterns]
        forward_table = [set() for _ in node_patterns]
        backward_table = [set() for _ in node_patterns]
        roots = []
        
        for edge in edges:
            sp, ep = edge
            if sp >= len(node_patterns) or sp < 0:
                raise IndexError(f'Can not Compile Pattern, Edge {edge} Out of Node Range, '
                                 f'Except Value between 0 and {len(node_patterns) - 1}, however {sp} was given.')
            if ep >= len(node_patterns) or ep < 0:
                raise IndexError(f'Can not Compile Pattern, Edge {edge} Out of Node Range, '
                                 f'Except Value between 0 and {len(node_patterns) - 1}, however {ep} was given.')
            forward_table[sp].add(ep)
            backward_table[ep].add(sp)
            num_of_inputs[ep] += 1

        # initialization
        pop_list, sort_ret = deque(), []
        for idx, n_input in enumerate(num_of_inputs):
            if n_input == 0: 
                pop_list.append(idx)
                roots.append(idx)

        # topological sort
        for _ in range(len(visited)):
            if len(pop_list) == 0: break
            current = pop_list.popleft()

            for next in forward_table[current]:
                num_of_inputs[next] -= 1
                if num_of_inputs[next] == 0:
                    pop_list.append(next)

            visited[current] = True
            sort_ret.append(current)

        if all(visited):
            if len(roots) > 1:
                ppq_warning('More than 1 pattern root was found, '
                            'Complext Pattern might cause memory overflow ...')
            return sort_ret, forward_table, backward_table
        else:
            raise RuntimeError('Topological Sort failed. '
                               'Some node can not be sorted (might due to circular reference)')


class TypeExpr(Callable):
    def __init__(self, type: str) -> None:
        self.type = type
        super().__init__()

    def __call__(self, op: Operation) -> bool:
        return op.type == self.type


class PatternMatchHelper:
    @ staticmethod
    def match_burte_force(
        graph: BaseGraph, pattern: GraphPattern, 
        exclusive: bool, max_candidates: int = 1000000) -> List[List[Operation]]:
        """暴力子图模式匹配 这是 PPQ 0.6.6 更新的内容
        在 0.6.6 之前，我们使用具有不确定性的贪心匹配算法，但是考虑到实际应用中的问题
        在 0.6.6 版本之后，我们将其修改为枚举匹配。
        
        子图匹配问题是一个 NP-Hard 的问题，不存在多项式时间复杂度的解法。
        你需要给出一个模式子图，match_burte_force 方法将在 graph 中对模式子图进行匹配。
        
        PPQ 使用了非递归的算法完成上述匹配，其最坏时间和空间复杂度大概都是 O(NM^k)
        其中 N 是母图节点个数，M 是子图节点个数，k 是母图的最大出度
        
        对于存在二义性子图模式，匹配复杂度将指数级增长；为了限制算法执行时间，当匹配到多于
        max_candidates 个模式子图时，算法强制停机，并报错返回。
        
        实际使用中的时间复杂度更加接近于 O(NM)
        
        参数 exclusive 指定了是否需要进行精确匹配。在精确匹配模式下：
            1. 不允许模式子图中除根节点外的其他节点有来自模式子图以外节点的输入
            2. 不允许模式子图中除叶节点外的其他节点有朝向模式子图以外节点的输出

        Example:
            pt = PatternTree(
                patterns = [lambda x: x.is_computing_op, 'Softplus', 'Tanh', 'Mul']
                edges = [[0, 1], [1, 2], [2, 3], [0, 3]])

            pt create an abstract tree pattern of:
                                            --- 'Softplus'   ---   'Tanh' --
            lambda x: x.is_computing_op --- +                              + --- 'Mul'
                                            ---     ---     ---    ---    --

        """

        def is_linked(upstream_op: Operation, downstream_op: Operation) -> bool:
            if upstream_op is None or downstream_op is None: return True
            return downstream_op in graph.get_downstream_operations(upstream_op)

        node_order = pattern.order
        matched_patterns = []

        # match root from graph, further pattern matching will start from root.
        for operation in graph.operations.values():
            root_idx = node_order[0]
            if pattern.node_patterns[root_idx](operation):
                matched_patterns.append([operation] + [None for _ in range(len(node_order) - 1)])

        for idx in node_order[1: ]:
            node_candidates, next_generation = [], []
            for operation in graph.operations.values():
                if pattern.node_patterns[idx](operation):
                    node_candidates.append(operation)

            for matched_pattern in matched_patterns:
                for operation in node_candidates:
                    is_pattern_root = len(pattern.input_table[idx]) == 0
                    link_check, exclusive_check, duplicated_check = True, True, True

                    for upstream_idx in pattern.input_table[idx]:
                        upstream_op = matched_pattern[upstream_idx]

                        if not is_linked(upstream_op=upstream_op, downstream_op=operation):
                            link_check = False

                    if (operation in matched_pattern) and link_check:
                        duplicated_check = False

                    if (not is_pattern_root and exclusive) and link_check:
                        matched_set = set(matched_pattern)
                        for op in graph.get_upstream_operations(operation):
                            if op not in matched_set: exclusive_check = False
                        if len(pattern.input_table[idx]) != len(set(graph.get_upstream_operations(operation))):
                            exclusive_check = False

                    if link_check and duplicated_check and exclusive_check:
                        generated = matched_pattern.copy()
                        generated[idx] = operation
                        next_generation.append(generated)
 
            matched_patterns = next_generation
            if len(matched_patterns) > max_candidates:
                raise OverflowError('Too many candidate patterns. Simplify your pattern first.')

            if len(matched_patterns) == 0: break

        # final exclusive check
        if exclusive:
            filtered = []
            for matched_pattern in matched_patterns:
                check_flag = True
                for idx, operation in enumerate(matched_pattern):
                    if len(pattern.output_table[idx]) != 0:
                        if not all([op in matched_pattern for op in graph.get_downstream_operations(operation)]):
                            check_flag = False
                if check_flag:
                    filtered.append(matched_pattern)
            matched_patterns = filtered
        return matched_patterns


class SearchableGraph(GraphCommandProcessor):
    """PPQ Graph Searching Engine.

    Args:
        GraphCommandProcessor ([type]): [description]
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
                    ret_collection.append(path.copy().append_left(start_point))

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
        
        # new feature with ppq 0.6.5, if ep_expr is None, means search until mismatch.
        if len(following_ops) == 0 and ep_expr is None:
            return ret_collection.add(start_point)

        for op in following_ops:
            # if operation is a valid end point, add it to path and stop further searching.
            if ep_expr is not None and ep_expr(op):
                ret_collection.update([start_point, op])

            else:
                # if operation is not a valid relay point, end searching here.
                if not rp_expr(start_point, op): 
                    # new feature with ppq 0.6.5, if ep_expr is None, means search until mismatch.
                    if ep_expr is None: ret_collection.add(start_point)
                    continue

                # searching following operations.
                further_result = self._opset_matching(
                    start_point=op, rp_expr=rp_expr,
                    ep_expr=ep_expr, direction=direction)

                if len(further_result) > 0:
                    ret_collection.update(further_result)
                    ret_collection.add(start_point)

        self._cache[start_point] = ret_collection
        return ret_collection

    def path_matching(
        self,
        sp_expr: Callable,
        rp_expr: Callable,
        ep_expr: Callable,
        direction: str
        ) -> List[Path]:
        """path_matching is used for path searching on the graph, and complete
        paths will be returned. Note that it's possible for results to overflow
        when there are numerous matchings.

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
        """opset_matching is used for operator set searching, it returns
        relevant op set instead of specific paths, should be used when results
        of path_matching overflow.

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

    def pattern_matching(self, patterns: List[Callable], 
                         edges: List[List[int]], 
                         exclusive: bool = True) -> List[List[Operation]]:
        """暴力子图模式匹配 这是 PPQ 0.6.6 更新的内容
        在 0.6.6 之前，我们使用具有不确定性的贪心匹配算法，但是考虑到实际应用中的问题
        在 0.6.6 版本之后，我们将其修改为枚举匹配。

        子图匹配问题是一个 NP-Hard 的问题，不存在多项式时间复杂度的解法。
        你需要给出一个模式子图，match_burte_force 方法将在 graph 中对模式子图进行匹配。

        PPQ 使用了非递归的算法完成上述匹配，其最坏时间和空间复杂度大概都是 O(NM^k)
        其中 N 是母图节点个数，M 是子图节点个数，k 是母图的最大出度
        
        对于存在二义性子图模式，匹配复杂度将指数级增长；为了限制算法执行时间，当匹配到多于
        max_candidates 个模式子图时，算法强制停机，并报错返回。

        实际使用中的时间复杂度更加接近于 O(NM)
        
        参数 exclusive 指定了是否需要进行精确匹配。在精确匹配模式下：
            1. 不允许模式子图中除根节点外的其他节点有来自模式子图以外节点的输入
            2. 不允许模式子图中除叶节点外的其他节点有朝向模式子图以外节点的输出

        Example:
            pt = PatternTree(
                patterns = [lambda x: x.is_computing_op, 'Softplus', 'Tanh', 'Mul']
                edges = [[0, 1], [1, 2], [2, 3], [0, 3]])

            pt create an abstract tree pattern of:
                                            --- 'Softplus'   ---   'Tanh' --
            lambda x: x.is_computing_op --- +                              + --- 'Mul'
                                            ---     ---     ---    ---    --

        """
        return PatternMatchHelper().match_burte_force(
            graph=self.graph, pattern=GraphPattern(node_patterns=patterns, edges=edges), 
            exclusive=exclusive)

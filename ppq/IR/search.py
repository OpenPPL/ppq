from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Iterable, Iterator, List, Set, Union

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

        Args:
            query (str): [description]

        Raises:
            TypeError: [description]
            KeyError: [description]

        Returns:
            [type]: [description]
        """
        pass


class TreeNode:
    def __init__(self, idx: int, pattern: Callable, edges: List[int]) -> None:
        self.idx     = idx
        self.pattern = pattern
        self.edges   = edges


class PatternTree(Iterable):
    def __init__(self, patterns: List[Callable], edges: List[List[int]]) -> None:
        """Pattern Tree 是一个用来表示图模式的结构体 这将在图中检索任意一个抽象树形结构.

        你将使用 Pattern Tree 定义你的树结构
        使用 patterns 确定每一个节点需要满足的条件
        使用 edges 将节点们彼此相连从而构成树形结构

        patterns[0] 将被设定为树的根节点

        Args:
            patterns (List[Callable]): _description_
            edges (List[List[int]]): _description_

        Raises:
            TypeError: _description_
            TypeError: _description_
            ValueError: _description_
        """
        for pattern in patterns:
            if not isinstance(pattern, Callable):
                raise TypeError(f'Can not create Pattern Tree with pattern {str(pattern)} it is not callable.')
        for edge in edges:
            if not isinstance(edge, tuple) and not isinstance(edge, list):
                raise TypeError(f'Can not create Pattern Tree with edge {str(edge)} it is not tuple or list.')
            if len(edge) != 2:
                raise ValueError(f'Can not create Pattern Tree with edge {str(edge)} '
                                 f'it should contains exact 2 elements, however {len(edge)} was given.')
            sp, ep = edge
            if not isinstance(sp, int) or not isinstance(ep, int):
                raise TypeError(f'Can not create Pattern Tree was given edge {[str(sp), str(ep)]}, '
                                'expect int value here.')
        '''
        if len(edges) != len(patterns) - 1:
            raise ValueError('Can not create Pattern with you input, input node and edges is not a tree. '
                             '[num of edges != num of nodes - 1]')
        '''

        forward_stars = {node_id: [] for node_id in range(len(patterns))}
        for sp, ep in edges: forward_stars[sp].append(ep)

        self._nodes = [TreeNode(idx, pattern, forward_stars[idx]) for idx, pattern in enumerate(patterns)]
        self.root = self._nodes[0]

    def following_patterns(self, node_idx: int) -> List[Callable]:
        node = self._nodes[node_idx]
        return [self._nodes[following_node].pattern for following_node in node.edges]

    def __getitem__(self, idx: int) -> TreeNode:
        return self._nodes[idx]

    def __iter__(self) -> Iterator[TreeNode]:
        return super().__iter__()


class HungarianSolver:
    def __init__(self, num_of_ops: int, num_of_patterns: int, matches: List[List[int]]) -> None:
        """
        Hungarian Solver - Part of Tree Pattern Matching Algorithm
        For complex pattern, there might be more than 1 corresponding op was founded,
        Example:
        pt = PatternTree(
                patterns = [lambda x: x.is_computing_op, lambda x: x.is_computing_op, 'Conv', 'Mul']
                edges = [[0, 1], [1, 2], [2, 3], [0, 3]])

            pt create an abstract tree pattern of:
                                            --- lambda x: x.is_computing_op  --
            lambda x: x.is_computing_op --- +                                 + --- 'Mul'
                                            --- 'Conv'                       --

        There is an pattern overlap between x.is_computing_op and 'Conv',
            so pattern x.is_computing_op might have more than 1 matchings.

        in this case, we use hungarian algorithm to find an optimal matching(maximum).
            if there is more than 1 optimal matching, this solver will return ARBITRARY ONE.

        Args:
            num_of_ops (int):

            num_of_patterns (int):

            matches (List[List[int]]): matches is a collection of op - pattern matchings,
                if matches = [[0, 0], [1, 1], [2, 2]],
                    it means op0 is matched with pattern 0
                    it means op1 is matched with pattern 1
                    it means op2 is matched with pattern 2

            Hungarian Algorithm will find an optimial matching between ops and patterns.
        """
        self.neighbor_table  = {node_idx: set() for node_idx in range(num_of_patterns)}
        self.num_of_ops      = num_of_ops
        self.num_of_patterns = num_of_patterns
        self.matched         = [-1 for _ in range(num_of_ops)]
        for sp, ep in matches:
            self.neighbor_table[sp].add(ep)

    def solve(self) -> bool:
        # 有一个匹配失败就立即返回，无需继续计算
        visited = [False for _ in range(self.num_of_ops)]
        for i in range(self.num_of_ops):
            if not self._solve(i, visited): return False
        return True

    def _solve(self, pattern: int, visited: List[bool]) -> bool:
        for ep in self.neighbor_table[pattern]:
            if visited[ep]: continue
            visited[ep] = True
            if self.matched[ep] == -1 or self._solve(self.matched[ep], visited):
                self.matched[ep] = pattern
                return True
        return False


class TypeExpr(Callable):
    def __init__(self, type: str) -> None:
        self.type = type
        super().__init__()

    def __call__(self, op: Operation) -> bool:
        return op.type == self.type


class TreePatternMatcher:
    def __init__(self, pattern_tree: PatternTree) -> None:
        """TreePatternMatcher offers you a really powerful pattern matching
        tool that helps you finding specific structure from your graph. Define
        your network structure with pattern tree --- an internal data structure
        of PPQ, then extract corresponding graph structures from ppq graph via
        TreePatternMatcher.

        This feature will benefits you a lot when dealing with graph fusion and graph editing.
        PPQ use Depth First Search and Hungarian Algorithm to match pattern from your graph,
            be aware that for complex pattern it might cost a lot of time in matching.

        Time complexity: O(nkd^3),
            where n is the num of operation in your graph,
            k is the num of pattern in your pattern tree,
            d is the maximum drgree in your graph.

        Notice in most cases it won't reach this complexity,
        it always have a complexity like O(nk)

        ATTENTION: Do not use poorly designed pattern which might cause multiple matching results.
                   For this case, ppq will randomly pick one matched result and return.

        Example:
            pt = PatternTree(
                patterns = [lambda x: x.is_computing_op, 'Softplus', 'Tanh', 'Mul']
                edges = [[0, 1], [1, 2], [2, 3], [0, 3]])

            pt create an abstract tree pattern of:
                                            --- 'Softplus'   ---   'Tanh' --
            lambda x: x.is_computing_op --- +                              + --- 'Mul'
                                            ---     ---     ---    ---    --

        Args:
            pattern_tree (PatternTree): _description_
        """
        self.pattern_tree   = pattern_tree
        self._interal_store = []
        self.results        = []

    def match(self, graph: BaseGraph, exclusive: bool) -> List[List[Operation]]:
        root, candidates = self.pattern_tree.root, []

        # match root from graph, further pattern matching will start from root.
        for operation in graph.operations.values():
            if root.pattern(operation):
                candidates.append(operation)

        for op in candidates:
            self._interal_store = [None for _ in range(len(self.pattern_tree._nodes))]
            self._interal_store[0] = op
            if self._match(graph=graph, op=op, node_idx=0, exclusive=exclusive):
                self.results.append(self._interal_store)
        return self.results

    def _match(self, graph: BaseGraph, op: Operation, node_idx: int, exclusive: bool) -> bool:
        pnode, following_ops = self.pattern_tree[node_idx], graph.get_downstream_operations(op)
        num_of_ops, num_of_patterns = len(following_ops), len(self.pattern_tree.following_patterns(node_idx))

        # 递归终止于 pattern 结尾
        if len(pnode.edges) == 0:
            if self._interal_store[node_idx] is not None:
                if self._interal_store[node_idx] != op:
                    return False # 这种情况出现于复杂模式，直接返回失败即可
            self._interal_store[node_idx] = op
            return True

        matches = []
        for op_idx, downstream_op in enumerate(following_ops):
            for pattern_idx, pattern in enumerate(self.pattern_tree.following_patterns(node_idx)):
                if pattern(downstream_op):
                    matches.append((pattern_idx, op_idx))

        # exclusive 模式，不允许节点有额外输出边
        if exclusive and num_of_ops != num_of_patterns:
            return False

        # pattern 可以匹配到多个不同的节点，最坏情况下，找出所有可行的匹配方案时间复杂度为 n!
        # 其中 n 为 op 的下游节点个数，对于这种情况，我们不可能枚举所有匹配方案并完成算法 (P-completed)。
        # 如果 pattern 匹配到了多个节点，我们使用匈牙利算法找出任意一个最优匹配，并警告用户这样做的不完备性与随机性。
        # 作为用户，你不应该输入有歧义的 pattern.
        # 例如下面的 pattern:
        #   nodes = [lambda x: x.is_computing_op, 'Mul', 'Mul', 'Mul', ...]
        #   edges = [[0, 1], [0, 2], [0, 3], ...]
        # 树形匹配将有 3! 种可行方案，我们将随机选取其中一种，进一步枚举可能产生不可预期的结果
        matching_solver = HungarianSolver(
            num_of_ops = num_of_ops,
            num_of_patterns = num_of_patterns,
            matches = matches)

        # 二部图匹配失败，直接返回 False
        if not matching_solver.solve():
            return False

        # 二部图匹配成功，则 matching_solver.matched 中按顺序存储了匹配到的 op index
        # 以此继续向下递归
        for pattern_idx, op_idx in enumerate(matching_solver.matched):
            if self._interal_store[node_idx] is not None:
                if self._interal_store[node_idx] != op:
                    return False # 这种情况出现于复杂模式，直接返回失败即可
            self._interal_store[node_idx] = op
            flag = self._match(graph=graph, op=following_ops[op_idx],
                               node_idx=pnode.edges[pattern_idx], exclusive=exclusive)
            if not flag: return False
        return True


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
                         edges: List[List[int]], exclusive: bool = True) -> List[List[Operation]]:
        # compile string to type expr.
        for pidx in range (len(patterns)):
            pfunc = patterns[pidx]
            if isinstance(pfunc, str):
                patterns[pidx] = TypeExpr(pfunc)

        pm = TreePatternMatcher(PatternTree(patterns=patterns, edges=edges))
        return pm.match(self.graph, exclusive=exclusive)

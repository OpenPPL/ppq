
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Callable, List, Union

from .base.command import GraphCommand, GraphCommandType
from .base.graph import BaseGraph


class GraphCommandProcesser(Callable, metaclass=ABCMeta):
    def __init__(self, graph_or_processor: Union[BaseGraph, Callable]) -> None:
        """
            GraphCommandProcesser 是用于处理图上相关操作的抽象基类

            我们使用指令-责任链模式处理 PPQ 计算图的相关操作，具体来说：

                所有图上相关操作都由一个 GraphCommand 对象进行封装，
                这一对象封装了操作的类型和必要参数

                同时我们设计了 GraphCommandProcesser 类用于接收并处理对应的 GraphCommand

                GraphCommandProcesser 被设计为责任链模式，当接收到无法处理的 GraphCommand 时
                将把无法识别 GraphCommand 传递给责任链上的下一任 GraphCommandProcesser
                直到 GraphCommand 被处理，或抛出无法处理的异常

                当你实现一个新的 GraphCommandProcesser 时，需要实现其中的方法 _acceptable_command_types，
                该方法返回了所有可以被识别的 GraphCommand 类型，同时在 _process 的逻辑中对 GraphCommand 的请求进行处理

                这两个方法被设计为私有的，这意味着你不能单独访问责任链中的独立 GraphCommandProcesser，
                只能够通过责任链的方式发起请求

                如果在责任链中有多个可以处理同一类请求的 GraphCommandProcesser，
                只有最先接触到 GraphCommand 的 GraphCommandProcesser将会被调用

                GraphCommandProcesser 将按照自定义逻辑解析 GraphCommand，
                在 BaseGraph 做出相应处理并给出结果，实现方法请参考 RunnableGraph

            GraphCommandProcesser is an absract class for manipulating the graph

            We use Command-Responsibility Chain to process operations on the computational graph:
            
                all graph-related operations are encapsulated by GraphCommand objects, which contain
                the operation type and necessary parameters

                we design GraphCommandProcesser to process corresponding GraphCommand

                GraphCommandProcesser follows responsibility chain convention, a processer will pass
                the unrecogonized GraphCommand to the next processer in the chain, the GraphCommand will
                finally be executed when meeting the corresponding processer. An exception will be thrown
                if no processer in the current chain is able to deal with the given command

                When you implement a new GraphCommandProcesser, you should implement its inner method 
                _acceptable_command_types, it returns all available GraphCommand types for execution
                and you should implement corresponding execution details in _process

                You are not supposed to visit single GraphCommandProcesser in the chain, and if there are
                multiple processers which could deal with same Command type in the chain, the first processer
                receiving the command will execute

                GraphCommandProcesser follows its predefined logic to parse GraphCommand, execute on the graph 
                correspondingly and return the final result

        Args:
            graph_or_processor (BaseGraph, Callable): 被处理的图对象，可以是 BaseGraph 或者 GraphCommandProcesser
                如果是 GraphCommandProcesser 对象，则自动将 self 与 graph 链接成链
                the graph being executed or previous GraphCommandProcesser in the chain
        """
        if isinstance(graph_or_processor, GraphCommandProcesser):
            self._next_command_processer   = graph_or_processor
            self._graph                    = graph_or_processor._graph
        else:
            self._next_command_processer   = None
            self._graph                    = graph_or_processor

    @property
    @abstractproperty
    def _acceptable_command_types(self) -> List[GraphCommandType]:
        """

            Subclass of GraphCommandProcesser must give an implementation of this function

            Return all acceptable GraphCommandTypes in a list as result.
            something like:
                return [
                    GraphCommandType.DEPLOY_TO_CPU,
                    GraphCommandType.DEPLOY_TO_CUDA,
                    GraphCommandType.DEPLOY_TO_NUMPY
                ]

        Returns:
            List[GraphCommandType]: all acceptable GraphCommandTypes
        """
        raise NotImplementedError(
            'Oh, seems you forgot to implement GraphCommandProcesser._acceptable_command_types function')

    def __call__(self, command: GraphCommand) -> Any:
        """

            Invoking interface of GraphCommandProcesser responsibility chain.
            All processers within the chain shall be invoked by this function one be one,
            unitl there is a processer claim to accept input command object,
            the entire processing of responsibility chain ends then.

            invoke a GraphCommandProcesser chain like that:

                _ = GraphCommandProcesser(graph, next_command_processer=None)
                _ = GraphCommandProcesser(graph, next_command_processer=_)
                command_processer = GraphCommandProcesser(graph, next_command_processer=_)

                command = GraphCommand(GraphCommandType.DEPLOY_TO_CUDA)
                command_processer(command)

            All three GraphCommandProcesser will then be called one by one

            Never attempt to use function like _(command) in above case,
            all responsibility chains should only be called by its head.

        Args:
            command (GraphCommand): An acceptable GraphCommand object
            if an inproper GraphCommand is given, it will incurs ValueError at end.

        Raises:
            ValueError: raise when there is no suitable processer for your command.

        Returns:
            Any: processer will decide what is it result.
        """

        if not isinstance(command, GraphCommand):
            raise ValueError(
                f'command should be an instance of GraphCommand, {type(command)} recieved yet.'
            )

        if command.command_type in self._acceptable_command_types():
            return self.process(command)

        elif self._next_command_processer is not None:
            self._next_command_processer(command)

        else:
            raise ValueError(
                f'Command Type {command.command_type} is not acceptable in this graph, ' \
                'please make sure you have added proper command processer into processing chain.\n'\
                'For more information, you may refer to ppq.IR.graph.GraphCommandType'
            )

    def acceptable_command_types(self) -> List[GraphCommandType]:
        """
            Return all acceptable command types of current chain.
            Notice there might be duplicated types.

        Returns:
            List[GraphCommandType]: all acceptable command types
        """
        my_types = self._acceptable_command_types()
        assert isinstance(my_types, list), \
            'GraphCommandProcesser.__acceptable_command_types must return a list of GraphCommandType'

        if self._next_command_processer is not None:
            other_types = self._next_command_processer.acceptable_command_types()
        return my_types.extend(other_types)

    @ abstractmethod
    def process(self, command: GraphCommand) -> Any:
        """

            Subclass of GraphCommandProcesser must give an implementation of this function

            Process recieved GraphCommand instance and give result(if there is any)

        Args:
            command (GraphCommand): input command object.

        Returns:
            Any: any result is fine.
        """
        raise NotImplementedError(
            'Oh, seems you forgot to implement GraphCommandProcesser.process function(or may be forgot to return?)')

    @ property
    def graph(self) -> BaseGraph:
        return self._graph

    def __str__(self) -> str:
        return f'GraphCommandProcesser {self.__hash__}'


class DefaultGraphProcesser(GraphCommandProcesser):
    def __init__(self, graph: BaseGraph, next_command_processer: Callable) -> None:
        super().__init__(graph, next_command_processer=next_command_processer)

    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return []

    def _process(self, command: GraphCommand) -> Any:
        return super().process(command)

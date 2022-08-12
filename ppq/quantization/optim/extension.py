from typing import Iterable
from ppq.IR.base.graph import BaseGraph

from ppq.executor import BaseGraphExecutor
from ppq.IR import BaseGraph

from .base import QuantizationOptimizationPass


class ExtensionPass(QuantizationOptimizationPass):
    """ExtensionPass 并没有什么用，它就是告诉你你可以像这样写一个 pass。 你可以直接改写 ExtensionPass
    的逻辑来实现你的功能，并将修改后的代码提交到 github.

    不过比较我们已经为 ExtensionPass 创建了一个 TemplateSetting 用来给它传递参数
        你可以去 ppq.api.setting.py 里面找到它

    There is nothing in ExtensionPass, it is literally an empty pass,
        -- just show you how to create your own pass.

    A TemplateSetting class has been created for passing parameter to this pass.
        You can find it in ppq.api.setting.py

    You can overwrite logic inside this pass.
    """
    def __init__(self, parameter: str) -> None:
        self.parameter = parameter
        super().__init__(name='PPQ Extension Pass')

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        assert isinstance(graph, BaseGraph)

        print('You are invoking Extension Pass now.')

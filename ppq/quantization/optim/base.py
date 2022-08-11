import time
from abc import ABCMeta, abstractmethod
from typing import Container, Iterable, Iterator, List

from ppq.executor import BaseGraphExecutor
from ppq.IR import BaseGraph, BaseGraph


class QuantizationOptimizationPass(metaclass = ABCMeta):
    """QuantizationOptimizationPass is a basic building block of PPQ
    quantization logic.

    PPQ is designed as a Multi pass Compiler of quantization network.
        where pass here refers to a traversal through the entire network.

    This class is an abstract base class of all customized passes.
    Quantizer will build an optimization pipeline later to quantize and optimize your network.
    """
    def __init__(self, name: str = 'Default Quanzation Optim') -> None:
        self.name = name

    def apply(
        self, graph: BaseGraph,
        dataloader: Iterable, executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        if not isinstance(graph, BaseGraph):
            raise TypeError(
                f'Incorrect graph object input, expect PPQ BaseGraph here, '
                f'while {type(graph)} was given.')
        self.optimize(graph, dataloader=dataloader, executor=executor, **kwargs)

    @ abstractmethod
    def optimize(
        self, graph: BaseGraph,
        dataloader: Iterable, executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        raise NotImplementedError('Implement this function first.')

    def __str__(self) -> str:
        return f'QuantizationOptimizationPass[{self.name}]'


class QuantizationOptimizationPipeline(Container, Iterable):
    """QuantizationOptimizationPipeline is a sorted set PPQ Optimization
    passes.

    PPQ is designed as a Multi pass Compiler of quantization network.
        where pass here refers to a traversal through the entire network.

    Quantizer is going to calling optimization pass from pipeline one by one to
        eventually finish network quantization procedure
    """
    def __init__(self, passes: List[QuantizationOptimizationPass]) -> None:
        super().__init__()
        self._pipeline = []
        if passes is not None:
            for optim in passes:
                self.append_optimization_to_pipeline(optim_pass=optim)

    def __len__(self) -> int:
        return len(self._pipeline)

    def __contains__(self, __x: QuantizationOptimizationPass) -> bool:
        assert isinstance(__x, QuantizationOptimizationPass), \
            f'Quantization Optimization Pipeline object only suppose to contain optimization passes, '\
            f'while you require to check a/an {type(__x)} whether in the optimization list'
        return __x in self._pipeline

    def __iter__(self) -> Iterator[QuantizationOptimizationPass]:
        return self._pipeline.__iter__()

    def optimize(
        self, graph: BaseGraph,
        dataloader: Iterable, executor: BaseGraphExecutor, verbose: bool = False,
        **kwargs
    ) -> None:
        max_name_length = 0
        if len(self._pipeline) > 0:
            names = [p.name for p in self._pipeline]
            max_name_length = max([len(name) for name in names])

        for optim_pass in self._pipeline:
            if not isinstance(optim_pass, QuantizationOptimizationPass):
                raise TypeError(f'Quantization Optimization Pipeline object only suppose to contain optimization passes only, '
                     f'while {str(optim_pass)}({type(optim_pass)}) was found.')

            if verbose:
                padding_length = abs(max_name_length - len(optim_pass.name))
                print(f'[{time.strftime("%H:%M:%S", time.localtime())}] {optim_pass.name} Running ... '
                      + ' ' * padding_length, end='')
            
            if not isinstance(graph, BaseGraph): 
                raise TypeError(f'parameter 1 should be an instance of PPQ BaseGraph when calling optim pass, '
                                f'however {type(graph)} was given.')
            optim_pass.apply(graph=graph, dataloader=dataloader, executor=executor, **kwargs)
            if verbose: print(f'Finished.')

    def append_optimization_to_pipeline(self, optim_pass: QuantizationOptimizationPass, at_front:bool = False):
        assert isinstance(optim_pass, QuantizationOptimizationPass), \
            f'Quantization Optimization Pipeline object only suppose to contain optimization passes, '\
            f'while we got a/an {type(optim_pass)} in the optimization list'
        if not at_front:
            self._pipeline.append(optim_pass)
        else:
            self._pipeline = [optim_pass] + self._pipeline
        return self

    def report(self) -> str:
        report = ''
        for optimization_pass in self._pipeline:
            report += str(optimization_pass) + '\n'
        return report

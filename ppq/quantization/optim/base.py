import time
from abc import ABCMeta, abstractmethod
from typing import Container, Iterable, Iterator, List, Union

from ppq.executor import BaseGraphExecutor
from ppq.IR import BaseGraph, DefaultGraphProcesser, GraphCommandProcesser


class QuantizationOptimizationPass(metaclass = ABCMeta):
    """
    QuantizationOptimizationPass is a basic building block of PPQ quantization logic.
    
    PPQ is designed as a Multi pass Compiler of quantization network.
        where pass here refers to a traversal through the entire network.

    This class is an abstract base class of all custimized passes.
    Quantizer will build an optimization pipeline later to quantize and optimize your network.
    """
    def __init__(self, name: str = 'Default Quanzation Optim') -> None:
        self.name = name

    def apply(
        self, processer: GraphCommandProcesser,
        dataloader: Iterable, executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        if not isinstance(processer, GraphCommandProcesser):
            raise TypeError(f'Incorrect graph object input, expect one GraphCommandProcesser object, ' \
            f'while {type(processer)} was given.')
        self.optimize(processer, dataloader=dataloader, executor=executor, **kwargs)

    @ abstractmethod
    def optimize(
        self, processer: GraphCommandProcesser, 
        dataloader: Iterable, executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        raise NotImplementedError('Implement this function first.')

    def __str__(self) -> str:
        return f'QuantizationOptimizationPass[{self.name}]'


class QuantizationOptimizationPipeline(Container, Iterable):
    """
    QuantizationOptimizationPipeline is a sorted set PPQ Optimization passes.
    
    PPQ is designed as a Multi pass Compiler of quantization network.
        where pass here refers to a traversal through the entire network.
    
    Quantizer is going to calling optimization pass from pipeline one by one to 
        eventully finish network quantization procedure
    """
    def __init__(self, passes: List[QuantizationOptimizationPass]) -> None:
        super().__init__()
        self._optimization_passes = []
        if passes is not None:
            for optim in passes:
                self.append_optimization_to_pipeline(optimization_pass=optim)

    def __len__(self) -> int:
        return len(self._optimization_passes)

    def __contains__(self, __x: QuantizationOptimizationPass) -> bool:
        assert isinstance(__x, QuantizationOptimizationPass), \
            f'Quantization Optimization Pipeline object only suppose to contain optimization passes, '\
            f'while you require to check a/an {type(__x)} whether in the optimization list'
        return __x in self._optimization_passes

    def __iter__(self) -> Iterator[QuantizationOptimizationPass]:
        return self._optimization_passes.__iter__()

    def optimize(
        self, graph: Union[GraphCommandProcesser, BaseGraph],
        dataloader: Iterable, executor: BaseGraphExecutor, verbose: bool = False,
        **kwargs
    ) -> None:
        if isinstance(graph, BaseGraph): processer = DefaultGraphProcesser(graph=graph)
        else: processer = graph
        
        max_name_length = 0
        if len(self._optimization_passes) > 0:
            names = [p.name for p in self._optimization_passes]
            max_name_length = max([len(name) for name in names])

        for optimization_pass in self._optimization_passes:
            if not isinstance(optimization_pass, QuantizationOptimizationPass):
                raise TypeError(f'Quantization Optimization Pipeline object only suppose to contain optimization passes only, '
                     f'while {str(optimization_pass)}({type(optimization_pass)}) was found.')
            
            if verbose: 
                padding_length = abs(max_name_length - len(optimization_pass.name))
                print(f'[{time.strftime("%H:%M:%S", time.localtime())}] {optimization_pass.name} Running ... ' 
                      + ' ' * padding_length, end='')
            optimization_pass.apply(processer=processer, dataloader=dataloader, executor=executor, **kwargs)
            if verbose: print(f'Finished.')

    def append_optimization_to_pipeline(self, optimization_pass: QuantizationOptimizationPass, at_front:bool = False):
        assert isinstance(optimization_pass, QuantizationOptimizationPass), \
            f'Quantization Optimization Pipeline object only suppose to contain optimization passes, '\
            f'while we got a/an {type(optimization_pass)} in the optimization list'
        if not at_front:
            self._optimization_passes.append(optimization_pass)
        else:
            self._optimization_passes = [optimization_pass] + self._optimization_passes
        return self

    def report(self) -> str:
        report = ''
        for optimization_pass in self._optimization_passes:
            report += str(optimization_pass) + '\n'
        return report

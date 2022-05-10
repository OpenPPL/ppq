from collections import deque
from math import ceil, floor
from typing import Callable, Dict, Iterable, List, Set, Tuple

import torch
import torch.nn.functional as F
from ppq.core import (LINEAR_ACTIVATIONS, PASSIVE_OPERATIONS,
                      PPLCUDA_ACTIVATIONS, TargetPlatform)
from ppq.executor.base import BaseGraphExecutor
from ppq.IR import BaseGraph, Operation
from ppq.IR.search import SearchableGraph
from ppq.log import NaiveLogger
from tqdm import tqdm

from .dispatchers import PPLNNDispatcher

logger = NaiveLogger.get_logger('PPQ')


class HessianDispatcher:

    def __init__(self,
                percentage: float,
                memory_aware: bool,
                max_iter: int=50,
                tol: float=1e-3,
                device: str='cuda'
    ) -> None:
        self.percentage = percentage
        self.memory_aware = memory_aware
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def find_all_computing_ops(self, graph: BaseGraph) -> Tuple[Set[Operation], Set[Operation]]:
        search_engine = SearchableGraph(graph)
        last_op_names = [var.source_op.name for var in graph.outputs.values() if var.source_op is not None]
        results = search_engine.opset_matching(
            sp_expr = lambda x: x.name in last_op_names,
            rp_expr = lambda x,y: True,
            ep_expr = lambda x: x.is_computing_op,
            direction = 'up'
        )
        all_computing_ops, last_computing_ops = set(), set()
        for op in results:
            if op.is_computing_op:
                last_computing_ops.add(op)
        for op in graph.operations.values():
            if op.is_computing_op:
                all_computing_ops.add(op)
        return all_computing_ops, last_computing_ops


    def enable_gradient(self, graph: BaseGraph) -> None:
        for op in graph.operations.values():
            for param in op.parameters:
                if torch.is_tensor(param.value) and param.value.dtype == torch.float:
                    param.value.requires_grad = True

    def disable_gradient(self, graph: BaseGraph) -> None:
        for op in graph.operations.values():
            for param in op.parameters:
                if torch.is_tensor(param.value) and param.value.dtype == torch.float:
                    param.value.requires_grad = False

    def get_params_grad(self,
                        all_computing_ops: List[Operation],
                        allow_grad_none: bool=True
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        params, grads = [], []
        for op in all_computing_ops:
            params.append(op.parameters[0].value)
            if not allow_grad_none:
                assert(op.parameters[0].value.grad is not None, f'grad of {op.name} weight is None')

            grads.append(0. if op.parameters[0].value.grad is None else op.parameters[0].value.grad + 0.)
        return params, grads

    def zero_grad(self, graph: BaseGraph) -> None:
        for op in graph.operations.values():
            for param in op.parameters:
                if param.value.requires_grad and param.value.grad is not None:
                    param.value.grad.detach_()
                    param.value.grad.zero_()


    def hv_product(self,
                v: List[torch.Tensor],
                params: List[torch.Tensor],
                output_names: List[str],
                all_computing_ops: Set[Operation],
                executor: BaseGraphExecutor,
                dataloader: Iterable,
                calib_steps: int,
                collate_fn: Callable
    ) -> Tuple[List[float], List[torch.Tensor]]:
        THv = [torch.zeros(p.size()).to(self.device) for p in params]
        calib_step, num_data = 0, 0
        for calib_epoch in range(ceil(calib_steps / len(dataloader))):
            for _,data in enumerate(dataloader):
                if collate_fn is not None:
                    data = collate_fn(data)
                tmp_num_data = data.size(0)
                outputs = executor.forward_with_gradient(data, output_names)
                for i in range(len(outputs)):
                    loss = F.mse_loss(outputs[i], outputs[i].detach())
                    loss.backward(create_graph=True, retain_graph=True)
                params_, grads_ = self.get_params_grad(all_computing_ops, allow_grad_none=False)
                self.zero_grad(executor._graph)

                Hessian_v = torch.autograd.grad(grads_,
                                            params_,
                                            grad_outputs=v,
                                            only_inputs=True,
                                            retain_graph=False)
                THv = [THv_ + H_ * float(tmp_num_data) + 0. for
                       THv_, H_ in zip(THv, Hessian_v)]
                # THv = [THv_ + H_ + 0. for
                #        THv_, H_ in zip(THv, Hessian_v)]

                num_data += float(tmp_num_data)
                calib_step += 1
                if calib_step >= calib_steps:
                    break

        THv = [THv_ / float(num_data) for THv_ in THv]
        eigenvalues = [torch.sum(x * y).cpu().item() for (x, y) in zip(THv, v)]
        return eigenvalues, THv


    def normalize(self, v: List[torch.Tensor]) -> List[torch.Tensor]:
        s = [torch.sum(x * x).sqrt() for x in v]
        v = [v_ / (s_ + 1e-6) for (v_, s_) in zip(v, s)]
        return v


    def converge(self, pre_eigenvalues: List[float], eigenvalues: List[float]) -> bool:
        for i in range(len(eigenvalues)):
            if pre_eigenvalues[i] is None:
                return False
            if abs(eigenvalues[i] - pre_eigenvalues[i]) / (abs(eigenvalues[i]) + 1e-6) > self.tol:
                return False
        return True

    def broadcast(self, graph: BaseGraph, computing_op_platforms: Dict[str, TargetPlatform]) -> None:

        def available(op: Operation) -> bool:
            return is_activation(op) or is_passive(op)

        def is_activation(op: Operation) -> bool:
            return op.type in PPLCUDA_ACTIVATIONS or op.type in LINEAR_ACTIVATIONS

        def is_passive(op: Operation) -> bool:
            return op.type in PASSIVE_OPERATIONS

        def is_root(op: Operation) -> bool:
            for var in op.inputs:
                if var.source_op is not None and var.source_op.name not in computing_op_platforms:
                    return False
            return True

        def recursive_mark(pre_operation: Operation, cur_operation: Operation) -> None:
            if pre_operation is not None:
                if not available(cur_operation):
                    return

                if is_activation(cur_operation) and not pre_operation.is_computing_op\
                    and not is_passive(cur_operation):
                    computing_op_platforms[cur_operation.name] = TargetPlatform.PPL_CUDA_INT8
                else:
                    computing_op_platforms[cur_operation.name] = computing_op_platforms[pre_operation.name]
            for op in graph.get_downstream_operations(cur_operation):
                recursive_mark(cur_operation, op)

        def trace_binary(pre_operation: Operation, cur_operation: Operation) -> None:
            if pre_operation is None:
                for var in cur_operation.inputs:
                    if var.source_op is not None and\
                        computing_op_platforms[var.source_op.name] == TargetPlatform.PPL_CUDA_INT8:
                        computing_op_platforms[cur_operation.name] = TargetPlatform.PPL_CUDA_INT8
                        break
                if  cur_operation.name not in computing_op_platforms:
                    computing_op_platforms[cur_operation.name] = TargetPlatform.PPL_CUDA_INT4
            else:
                if not available(cur_operation):
                    if cur_operation.type in ['Add', 'Concat', 'Sub', 'Mul'] and\
                        is_root(cur_operation) and cur_operation not in queue:
                        queue.append(cur_operation)
                    return
                if is_activation(cur_operation) and not pre_operation.type in ['Add', 'Concat', 'Sub', 'Mul']\
                        and not is_passive(cur_operation):
                    computing_op_platforms[cur_operation.name] = TargetPlatform.PPL_CUDA_INT8
                else:
                    computing_op_platforms[cur_operation.name] = computing_op_platforms[pre_operation.name]
            for op in graph.get_downstream_operations(cur_operation):
                trace_binary(cur_operation, op)

        for op in graph.operations.values():
            if op.name in computing_op_platforms:
                recursive_mark(None, op)

        queue = deque()
        for op in graph.operations.values():
            if op.type in ['Add', 'Concat', 'Sub', 'Mul'] and is_root(op):
                queue.append(op)

        while len(queue):
            op = queue.popleft()
            trace_binary(None, op)


    def dispatch(self,
                graph: BaseGraph,
                executor: BaseGraphExecutor,
                dataloader: Iterable,
                calib_steps: int,
                collate_fn: Callable,
                **kwargs
    ) -> Dict[str, TargetPlatform]:
        self.enable_gradient(graph)
        all_computing_ops, last_computing_ops = self.find_all_computing_ops(graph)
        output_names = [op.outputs[0].name for op in last_computing_ops]
        params, _ = self.get_params_grad(all_computing_ops)
        eigenvalues = [None for i in range(len(params))]
        dispatching_table = {}

        v = [torch.randn(p.size()).to(self.device) for p in params]
        v = self.normalize(v)
        logger.info('Now computing eigenvalues by power iteration')
        for i in tqdm(range(self.max_iter), total=self.max_iter):
            self.zero_grad(graph)

            tmp_eigenvalues, Hv = self.hv_product(v,
                                                params,
                                                output_names,
                                                all_computing_ops,
                                                executor,
                                                dataloader,
                                                calib_steps,
                                                collate_fn)
            v = self.normalize(Hv)
            if not self.converge(eigenvalues, tmp_eigenvalues):
                eigenvalues = tmp_eigenvalues
            else:
                logger.info('Converged already, stopping the iteration')
                eigenvalues = tmp_eigenvalues
                break

        all_op_names = [op.name for op in all_computing_ops]
        all_op_eigenvalues = [(x, y) for x,y in zip(all_op_names, eigenvalues)]
        if self.memory_aware:
            all_param_nums = [float(p.numel()) for p in params]
            for i in range(len(all_op_eigenvalues)):
                all_op_eigenvalues[i][1] /= all_param_nums[i]

        all_op_eigenvalues = sorted(all_op_eigenvalues, key=lambda x: x[1])

        for i in range(len(all_op_eigenvalues)):
            if i < floor(len(all_op_eigenvalues) * self.percentage):
                dispatching_table[all_op_eigenvalues[i][0]] = TargetPlatform.PPL_CUDA_INT4
            else:
                dispatching_table[all_op_eigenvalues[i][0]] = TargetPlatform.PPL_CUDA_INT8
        self.disable_gradient(graph)
        self.broadcast(graph, dispatching_table)

        # for op in graph.operations.values():
        #     if op.name not in dispatching_table:
        #         print(op.name)
        return dispatching_table


class PPLNNMixPrecisionDispatcher:
    @staticmethod
    def dispatch(
        graph: BaseGraph,
        quant_platform: TargetPlatform,
        fp32_platform: TargetPlatform,
        SOI_platform: TargetPlatform,
        INT4_COMPUTING_OPS: List[str],
        **kwargs
    ) -> Dict[str, TargetPlatform]:
        """dispatch certain ops to INT4 platforms, note that INT4_COMPUTING_OPS
        should be determined by users in advance.

        Args:
            graph (BaseGraph): ppq graph ir.
            quant_platform (TargetPlatform): platform object where quantable parts will goes to.
            SOI_platform (TargetPlatform): platform object where SOI parts will goes to.
            fp32_platform (TargetPlatform): platform object where remaining parts will goes to.
            INT4_COMPUTING_OPS (List[str]): a list of ops which need being dispatched to PPL_CUDA_INT4

        Returns:
            Dict[str, TargetPlatform]: final dispatching table
        """

        # whether dispatched to quant platform by pplnn dispatcher
        def pplnn_permit(op: Operation) -> bool:
            return pplnn_dispatching_table[op.name] == quant_platform

        # whether the search will continue at current op
        def available(op: Operation) -> bool:
            return (is_activation(op) or is_passive(op)) and pplnn_permit(op)

        # whether it's activation op
        def is_activation(op: Operation) -> bool:
            return op.type in PPLCUDA_ACTIVATIONS or op.type in LINEAR_ACTIVATIONS

        # whether it's passive op
        def is_passive(op: Operation) -> bool:
            return op.type in PASSIVE_OPERATIONS

        # whether it's root op, i.e., platforms of input ops all identified
        def is_root(op: Operation) -> bool:
            for var in op.inputs:
                if var.source_op is not None and var.source_op.name not in computing_op_platforms:
                    return False
            return True

        # start from computing ops, recursively mark subsequent ops
        def recursive_mark(pre_operation: Operation, cur_operation: Operation) -> None:
            if pre_operation is not None:
                if not available(cur_operation):
                    return

                if is_activation(cur_operation) and not pre_operation.is_computing_op\
                    and not is_passive(cur_operation):
                    computing_op_platforms[cur_operation.name] = TargetPlatform.PPL_CUDA_INT8
                else:
                    computing_op_platforms[cur_operation.name] = computing_op_platforms[pre_operation.name]
            for op in graph.get_downstream_operations(cur_operation):
                recursive_mark(cur_operation, op)

        # start from binary ops, recursively mark and add newly discovered root op
        def trace_binary(pre_operation: Operation, cur_operation: Operation) -> None:
            if pre_operation is None:
                for var in cur_operation.inputs:
                    if var.source_op is not None and\
                        computing_op_platforms[var.source_op.name] == TargetPlatform.PPL_CUDA_INT8:
                        computing_op_platforms[cur_operation.name] = TargetPlatform.PPL_CUDA_INT8
                        break
                if  cur_operation.name not in computing_op_platforms:
                    computing_op_platforms[cur_operation.name] = TargetPlatform.PPL_CUDA_INT4
            else:
                if not available(cur_operation):
                    if cur_operation.type in ['Add', 'Concat', 'Sub', 'Mul'] and\
                        is_root(cur_operation) and pplnn_permit(cur_operation) and\
                        cur_operation not in queue:
                        queue.append(cur_operation)
                    return
                if is_activation(cur_operation) and not pre_operation.type in ['Add', 'Concat', 'Sub', 'Mul']\
                        and not is_passive(cur_operation):
                    computing_op_platforms[cur_operation.name] = TargetPlatform.PPL_CUDA_INT8
                else:
                    computing_op_platforms[cur_operation.name] = computing_op_platforms[pre_operation.name]
            for op in graph.get_downstream_operations(cur_operation):
                trace_binary(cur_operation, op)

        computing_op_platforms = {}
        pplnn_dispatching_table = PPLNNDispatcher.dispatch(
            graph = graph,
            quant_platform = quant_platform,
            fp32_platform = fp32_platform,
            SOI_platform = SOI_platform
        )
        for name in pplnn_dispatching_table:
            if graph.operations[name].is_computing_op and \
                pplnn_dispatching_table[name] == quant_platform:
                computing_op_platforms[name] = TargetPlatform.PPL_CUDA_INT8

        for name in INT4_COMPUTING_OPS:
            if pplnn_dispatching_table[name] == quant_platform:
                computing_op_platforms[name] = TargetPlatform.PPL_CUDA_INT4
            else:
                logger.warning(f"""Op {name} is not dispatched to quantable platforms by
                pplnn dispatcher, ignore this op here""")

        for op in graph.operations.values():
            if op.name in computing_op_platforms:
                recursive_mark(None, op)

        queue = deque()
        for op in graph.operations.values():
            if op.type in ['Add', 'Concat', 'Sub', 'Mul'] and is_root(op) and pplnn_permit(op):
                queue.append(op)

        while len(queue):
            op = queue.popleft()
            trace_binary(None, op)

        for name in computing_op_platforms:
            if computing_op_platforms[name] == TargetPlatform.PPL_CUDA_INT4:
                pplnn_dispatching_table[name] = TargetPlatform.PPL_CUDA_INT4

        return pplnn_dispatching_table

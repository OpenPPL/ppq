from typing import Dict, List

import numpy as np
import torch
from ppq.core import (CAFFE_DOMAIN, IS_DISPATCHED_GRAPH, NetworkFramework,
                      empty_ppq_cache)
from ppq.executor import TorchExecutor
from ppq.IR import (BaseGraph, GraphCommand, GraphCommandType, GraphFormatter,
                    GraphMerger, Operation, Opset, Variable)

from . import ppl_caffe_pb2

caffe_import_map = {}

from ppq.log import NaiveLogger

logger = NaiveLogger.get_logger('PPQ')


def build_temp_graph(initializer: Dict[str, dict], nodes: List[dict], inputs: List[str]=[], outputs: List[str]=[], graph: BaseGraph=None) -> BaseGraph:
    """could either build graph from scratch or append new nodes to a existing
    graph, if build form scratch, graph should be None, otherwise we use the
    given graph and append nodes and variables."""
    from_scratch = False
    if graph is None:
        graph = BaseGraph(name='Infer', built_from=NetworkFramework.CAFFE)
        from_scratch = True

    op_inputs_dict, op_outputs_dict = {}, {}
    for node in nodes:
        if node['name'] in graph.operations:
            raise KeyError(f"Duplicated operation {node['name']} was found.")
        graph.operations[node['name']] = Operation(name=node['name'], op_type=node['op_type'], 
                                    attributes=node['attribute'].copy(), opset=Opset(domain=CAFFE_DOMAIN, version=1))
        op_inputs_dict[node['name']] = [_ for _ in node['inputs']]
        op_outputs_dict[node['name']] = [_ for _ in node['outputs']]

    var_list = []
    for op_name, input_vars in op_inputs_dict.items():
        var_list.extend(input_vars)
    for op_name, output_vars in op_outputs_dict.items():
        var_list.extend(output_vars)

    # create all variable at once.
    for var_name in set(var_list):
        if var_name in graph.variables:
            continue
        graph.variables[var_name] = Variable(name=var_name)

    # build graph's input, output variables.
    # we only set graph inputs and outputs when build from scratch, i.e., when we are sure about graph inputs and outputs,
    # otherwise the graph inputs and outputs should manually set after the whole appending process from outside
    if from_scratch:
        try:
            for var_name in inputs:
                if var_name not in graph.variables: continue
                graph.inputs[var_name] = graph.variables[var_name]
            for var_name in outputs:
                graph.outputs[var_name] = graph.variables[var_name]
        except KeyError as e:
            raise KeyError(
                'seems you got an input/output variable that is not linked to any operation.')

    # build operation inputs, outputs variables.
    for op_name in op_inputs_dict:
        for var_name in op_inputs_dict[op_name]:
            var = graph.variables[var_name]
            var.dest_ops.append(graph.operations[op_name])
            graph.operations[op_name].inputs.append(graph.variables[var_name])
        for var_name in op_outputs_dict[op_name]:
            var = graph.variables[var_name]
            var.source_op = graph.operations[op_name]
            graph.operations[op_name].outputs.append(graph.variables[var_name])

    # initialize variable
    for var_name in initializer:
        if var_name in graph.variables:
            for dest_op in graph.variables[var_name].dest_ops:
                dest_op.parameters.append(graph.variables[var_name])
            graph.variables[var_name].value = initializer[var_name]['value']
            graph.variables[var_name].is_parameter = True

    return graph

def get_input_shape(net_def: ppl_caffe_pb2.NetParameter) -> Dict[str, list]:
    # Only support one format input shape, not support mixed format
    def layer_exist(layer_type):
        return layer_type in [item.type for item in net_def.layer]

    input_shape = {k: None for k in net_def.input}
    # Given input shape use input_shape field
    if len(net_def.input_shape) != 0:
        for i, name in enumerate(net_def.input):
            input_shape[name] = list(net_def.input_shape[i].dim)
    # Given input shape use input_dim
    # TODO: Here only support  4-D input
    elif len(net_def.input_dim) != 0:
        for i, name in enumerate(net_def.input):
            input_shape[name] = list(net_def.input_dim[i * 4:(i + 1) * 4])
    # Given input shape use input layer
    elif layer_exist('Input'):
        input_layer = [item for item in net_def.layer]
        for layer in input_layer:
            input_shape[layer.top[0]] = list(layer.input_param.shape.dim)
    else:
        raise TypeError('Unsupported network input format.')

    for k, v in input_shape.items():
        if v is None:
            raise TypeError("shape of input '%s' is not specified." % k)

    return input_shape

def register_class(cls):
    caffe_import_map[cls.__name__] = cls
    return cls


class CaffeOpBuilder(object):
    caffe_to_onnx = {
        'Convolution': 'Conv',
        'BatchNorm': 'BatchNormalization',
        'BN': 'BatchNormalization',
        'Deconvolution': 'ConvTranspose',
        'ReLU6': 'Clip',
        'PReLU': 'PRelu',
        'InnerProduct': 'Gemm',
        'ReflectionPad': 'Pad',
        'NNUpsample': 'Resize',
        'TanH': 'Tanh',
        'Power': 'Pow',
        'InstanceNorm': 'InstanceNormalization',
        'SubpixelUp': 'DepthToSpace',
        'SubpixelDown': 'SpaceToDepth',
        'ArgMax': 'CaffeArgMax',
        'HSwish': 'HardSwish',
        'HSigmoid': 'HardSigmoid'
    }

    def __init__(self,
                graph: ppl_caffe_pb2.NetParameter,
                layer: ppl_caffe_pb2.LayerParameter,
                input_shape: List[list]
    ):
        self.graph = graph
        self.layer = layer
        self.settings = {}
        self.initializer = {}
        self.nodes = []
        self.op_type = self.get_type()

        assert len(self.layer.bottom) == len(input_shape), 'Given input_shape and layer.bottom have different length'
        self.input_shape = input_shape
        self.out_shape = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_type(self):
        op_type = self.layer.type
        if self.layer.type in CaffeOpBuilder.caffe_to_onnx:
            op_type = CaffeOpBuilder.caffe_to_onnx[self.layer.type]

        return op_type

    def get_attr(self):
        return {}

    def trans(self):
        sub_int, param_name = self.param_process()
        self.initializer.update(sub_int)
        self.nodes.append({'name': self.layer.name, 'inputs': list(self.layer.bottom) + param_name,
                           'outputs': list(self.layer.top), 'op_type': self.op_type, 'attribute': self.get_attr()})
        self.infer_shape()
        return build_temp_graph(self.initializer, self.nodes, graph=self.graph)

    @empty_ppq_cache
    def infer_shape(self):
        input_names = self.layer.bottom
        # if len(set(self.layer.bottom)) != len(self.layer.bottom):
        #     # There are same variable in bottom, need to de_inplace
        #     visited_var = {}
        #     for var in self.layer.bottom:
        #         if var in visited_var.keys():
        #             input_names.append(var + '_' + str(visited_var[var]))
        #             visited_var[var] += 1
        #         else:
        #             input_names.append(var)
        #             visited_var[var] = 1
        # else:
        #     input_names = self.layer.bottom
        input_value = [np.random.rand(*i).astype('f') for i in self.input_shape]
        input_value = [torch.from_numpy(tensor).to(self.device) for tensor in input_value]
        input_value = dict([(x, y) for x, y in zip(input_names, input_value)])
        temp_graph = build_temp_graph(self.initializer, self.nodes, input_names, list(self.layer.top))

        # BUG FIX 12162021, Format graph when loading caffe model.
        formatter = GraphFormatter(GraphMerger(temp_graph))
        formatter(GraphCommand(GraphCommandType.FORMAT_CONSTANT_INPUT))
        formatter(GraphCommand(GraphCommandType.FUSE_BN))
        formatter(GraphCommand(GraphCommandType.FORMAT_PARAMETERS))
        formatter(GraphCommand(GraphCommandType.FORMAT_CAST))
        formatter(GraphCommand(GraphCommandType.DELETE_ISOLATED))

        # PATCH 20220805, add dispatching tag for executing.
        temp_graph.set_extension_attrib(IS_DISPATCHED_GRAPH, True)
        executor = TorchExecutor(graph=temp_graph, device=self.device)
        outputs = executor.forward(input_value, list(self.layer.top))
        self.out_shape = [list(i.shape) for i in outputs]

    def param_process(self):
        initializer = {}
        param_name = []
        for i, tensor in enumerate(self.layer.blobs):
            init_name = self.layer.name + '_param_' + str(i)
            dims = tensor.shape.dim
            if self.layer.type == 'BN' and len(tensor.shape.dim) == 4:
                # Some BN layers give the dim in [1, channel, 1, 1], while some give the dim in [channel]
                dims = tensor.shape.dim[1:2]
            np_type = 'float32' if not isinstance(tensor.data[0], int) else 'int'
            value = np.array(tensor.data, dtype=np_type).reshape(dims)
            initializer[init_name] = {'dims': dims, 'value': value}
            param_name.append(init_name)
        return initializer, param_name


@register_class
class Convolution(CaffeOpBuilder):
    def get_attr(self):
        conv = self.layer.convolution_param
        if conv.HasField('kernel_h') and conv.HasField('kernel_w'):
            kernel_shape = [conv.kernel_h, conv.kernel_w]
        else:
            kernel_shape = [conv.kernel_size] * 2
        if conv.HasField('stride_h') and conv.HasField('stride_w'):
            strides = [conv.stride_h, conv.stride_w]
        else:
            strides = [conv.stride] * 2
        if conv.HasField('pad_h') and conv.HasField('pad_w'):
            pads = [conv.pad_h, conv.pad_w] * 2
        else:
            pads = [conv.pad] * 4
        self.settings['kernel_shape'] = kernel_shape
        self.settings['strides'] = strides
        self.settings['pads'] = pads
        group = conv.group
        # convolution layer should not use hole parameter
        if conv.hole_h != 1 or conv.hole_w != 1:
            logger.warning(f'{self.layer.name} is dilated convolution, check if the real backend supports')

        self.settings['dilations'] = [conv.hole_h, conv.hole_w]
        self.settings['group'] = group
        return self.settings


@register_class
class BatchNorm(CaffeOpBuilder):
    # caffe batchnorm layer, only compute x_norm = (x-u)/std
    # Thus set weights = 1 and bias = 0
    def get_attr(self):
        batchnorm = self.layer.batch_norm_param
        # If use_global_stats is not given, the CaffeOpBuilder value of protobuf is false,
        # but PPL change it to be True in the code space
        if not batchnorm.use_global_stats:
            logger.warning('Now cannot convert BatchNorm layer with use_global_stats==False. Change it to True')
        # Because only support use_global_stats == True, we do not parse 'moving_average_fraction' here
        # self.settings['momentum'] = batchnorm.moving_average_fraction
        self.settings['epsilon'] = batchnorm.eps
        return self.settings

    def trans(self):
        # Set weights = 1
        dim = self.layer.blobs[0].shape.dim
        weights = ppl_caffe_pb2.BlobProto()
        weights.data.extend(np.ones(dim, dtype='f'))
        weights.shape.dim.extend(dim)
        self.layer.blobs.insert(0, weights)

        # Set bias = 0
        bias = ppl_caffe_pb2.BlobProto()
        bias.data.extend(np.zeros(dim, dtype='f'))
        bias.shape.dim.extend(dim)
        self.layer.blobs.insert(1, bias)

        if len(self.layer.blobs) == 5:
            # Remove scale_factor
            scale_factor = np.array(self.layer.blobs[4].data, dtype='f')
            self.layer.blobs.remove(self.layer.blobs[4])

            # Update running_mean and running_var according to scale_factor
            running_mean = self.layer.blobs[2]
            new_mean = np.array(running_mean.data, dtype='f') / scale_factor
            running_mean.ClearField('data')
            running_mean.data.extend(new_mean)

            running_var = self.layer.blobs[3]
            new_var = np.array(running_var.data, dtype='f') / scale_factor
            running_var.ClearField('data')
            running_var.data.extend(new_var)

        return super(BatchNorm, self).trans()


@register_class
class BN(CaffeOpBuilder):
    # BatchNorm + Scale, the same with BatchNormalization in pytorch
    # y = alpha * ((x-u)/std) + beta
    def get_attr(self):
        batchnorm = self.layer.bn_param
        if batchnorm.moving_average:
            logger.warning('Now cannot convert BatchNorm layer with moving_average==True. Change it to False')
        self.settings['epsilon'] = batchnorm.var_eps
        # Because only support moving_average == False, we do not parse 'decay' in BN layer
        return self.settings


@register_class
class Deconvolution(Convolution):
    pass


@register_class
class ReLU(CaffeOpBuilder):
    def get_type(self):
        if self.layer.relu_param.HasField('negative_slope'):
            return 'LeakyRelu'
        else:
            return 'Relu'

    def get_attr(self):
        relu = self.layer.relu_param
        if relu.HasField('negative_slope'):
            self.op_type = 'LeakyRelu'
            self.settings['alpha'] = relu.negative_slope
        return self.settings


@register_class
class PReLU(CaffeOpBuilder):
    def trans(self):
        assert len(self.layer.blobs) == 1, 'PReLU lacks of slope'
        dim_info = self.layer.blobs[0].shape.dim
        # squeeze the PReLU slope [1, 128, 1, 1] to [1, 128] if input is 2-D
        # prelu only has one input data
        if len(dim_info) == 4 and len(self.input_shape[0]) == 2:
            dim_info.pop()
            dim_info.pop()
        return super(PReLU, self).trans()


@register_class
class Concat(CaffeOpBuilder):
    def get_attr(self):
        concat = self.layer.concat_param
        self.settings['axis'] = concat.axis
        return self.settings


@register_class
class Softmax(CaffeOpBuilder):
    def get_attr(self):
        softmax = self.layer.softmax_param
        self.settings['axis'] = softmax.axis
        return self.settings


@register_class
class Transpose(CaffeOpBuilder):
    def get_attr(self):
        transpose = self.layer.transpose_param
        self.settings['perm'] = transpose.dim[:]
        return self.settings


@register_class
class ReduceL2(CaffeOpBuilder):
    def get_attr(self):
        reducel2 = self.layer.reducel2_param
        self.settings['axes'] = reducel2.axes
        self.settings['keepdims'] = reducel2.keepdims
        return self.settings


@register_class
class Reduce(CaffeOpBuilder):
    def get_type(self):
        return 'ReduceMean'

    def get_attr(self):
        reduce = self.layer.reduce_param
        self.settings['axis'] = reduce.axis
        return self.settings


@register_class
class Div(CaffeOpBuilder):
    pass


@register_class
class Pooling(CaffeOpBuilder):
    def get_attr(self):
        pool = self.layer.pooling_param
        if pool.global_pooling:
            return self.settings

        if pool.HasField('kernel_h') and pool.HasField('kernel_w'):
            kernel_shape = [pool.kernel_h, pool.kernel_w]
        else:
            kernel_shape = [pool.kernel_size] * 2
        if pool.HasField('stride_h') and pool.HasField('stride_w'):
            strides = [pool.stride_h, pool.stride_w]
        else:
            strides = [pool.stride] * 2
        if pool.HasField('pad_h') and pool.HasField('pad_w'):
            pads = [pool.pad_h, pool.pad_w] * 2
        else:
            pads = [pool.pad] * 4
        self.settings['kernel_shape'] = kernel_shape
        self.settings['strides'] = strides
        self.settings['pads'] = pads
        self.settings['ceil_mode'] = int(pool.ceil_mode)
        if not pool.global_pooling and pool.pool == 1:
            # set count_include_pad for average pooling
            self.settings['count_include_pad'] = 1
        return self.settings

    def get_type(self):
        # MAX=0, AVE=1
        pool = self.layer.pooling_param
        if not pool.global_pooling:
            if pool.pool == 0:
                return 'MaxPool'
            elif pool.pool == 1:
                return 'AveragePool'
            else:
                raise TypeError('unknown pooling operation type')

        else:
            if pool.pool == 0:
                return 'GlobalMaxPool'
            elif pool.pool == 1:
                return 'GlobalAveragePool'
            else:
                raise TypeError('unknown pooling operation type')


@register_class
class Eltwise(CaffeOpBuilder):
    def get_type(self):
        # PROD/MUL=0, ADD=1, MAX=2
        param = self.layer.eltwise_param
        if param.operation == 0:
            return 'Mul'
        elif param.operation == 1:
            return 'Add'
        elif param.operation == 2:
            return 'Max'
        else:
            raise TypeError('unknown eltwise operation type')

    def trans(self):
        def _add_mul(_node_name, input_name, factor):
            new_node_name = _node_name + '_mul'
            out_name = new_node_name + '_out'
            factor_name = new_node_name + '_coeff'
            init = {factor_name: {'dims': [1], 'value': np.array(factor, dtype='f')}}
            node = {'name': new_node_name, 'inputs': [input_name, factor_name], 'outputs': [out_name],
                    'op_type': 'Mul', 'attribute': self.get_attr()}
            self.initializer.update(init)
            self.nodes.append(node)
            return out_name

        coeff = list(self.layer.eltwise_param.coeff)
        if len(self.layer.bottom) > len(self.layer.eltwise_param.coeff):
            coeff.extend([1.0] * (len(self.layer.bottom) - len(self.layer.eltwise_param.coeff)))
        elif len(self.layer.bottom) < len(self.layer.eltwise_param.coeff):
            logger.error(f'Less bottom than coeff in {self.layer.name}')
            exit(-1)
        bottom_var = list(self.layer.bottom)

        # The number of inputs is uncertain, thus here may generate a series of binary op
        last = bottom_var[0]
        if coeff[0] != 1.0:
            node_name = self.layer.name + '_0'
            output_name = _add_mul(node_name, last, coeff[0])
            last = output_name

        for i, name in enumerate(bottom_var[1:]):
            node_name = self.layer.name + '_' + str(i + 1)
            output_name = node_name + '_out' if (i + 2 != len(bottom_var)) else self.layer.top[0]
            op_type = self.op_type
            if coeff[i + 1] == -1.0:
                op_type = 'Sub'
            elif coeff[i + 1] != 1.0:
                mul_op_name = _add_mul(node_name, name, coeff[i + 1])
                name = mul_op_name
            sub_node = {'name': node_name, 'inputs': [last, name], 'outputs': [output_name], 'op_type': op_type,
                        'attribute': self.get_attr()}
            last = output_name
            self.nodes.append(sub_node)

        if self.nodes[-1]['outputs'] != self.layer.top:
            # If Eltwise op only has one bottom and coeff != 1.0, the output name will be changed and
            # lead to missing key error
            assert len(self.nodes[-1]['outputs']) == len(self.layer.top)
            self.nodes[-1]['outputs'] = list(self.layer.top)
            logger.debug(f'Output of {self.nodes[-1]} != {self.layer.top}, Change it!')

        self.infer_shape()
        return build_temp_graph(self.initializer, self.nodes, graph=self.graph)


@register_class
class Reshape(CaffeOpBuilder):
    def trans(self):
        dim = self.layer.reshape_param.shape.dim
        init_name = self.layer.name + '_param'
        self.initializer = {init_name: {'dims': [len(dim)], 'value': np.array(dim, dtype='int64')}}
        self.nodes = [{'name': self.layer.name, 'inputs': list(self.layer.bottom) + [init_name],
                       'outputs': list(self.layer.top), 'op_type': self.layer.type, 'attribute': {}}]
        self.infer_shape()
        return build_temp_graph(self.initializer, self.nodes, graph=self.graph)


@register_class
class ReLU6(CaffeOpBuilder):
    def get_attr(self):
        self.settings = {'min': 0.0, 'max': 6.0}
        return self.settings

@register_class
class Clip(CaffeOpBuilder):
    def get_attr(self):
        param = self.layer.clip_param
        self.settings = {'min': param.min, 'max': param.max}
        return self.settings

@register_class
class Mul(CaffeOpBuilder):
    pass

@register_class
class Add(CaffeOpBuilder):
    pass

@register_class
class InnerProduct(CaffeOpBuilder):
    def get_attr(self):
        self.settings = {'transB': 1}
        if self.layer.inner_product_param.HasField('axis'):
            self.settings['axis'] = self.layer.inner_product_param.axis
        else:
            self.settings['axis'] = 1
        return self.settings

    def trans(self):
        ip_param = self.layer.inner_product_param
        assert ip_param.num_output == self.layer.blobs[0].shape.dim[0], 'Given num_output is not same with weight shape'
        input_var_list = list(self.layer.bottom)

        '''
        if not (len(self.input_shape[0]) == 2 and ip_param.axis == 1):
            # Insert an reshape op whether needed or not due to no shape information
            flatten_shape = [0, -1]
            reshape_name = self.layer.name + '_flatten_param'
            output_name = self.layer.name + '_flatten_output'
            self.initializer[reshape_name] = {'dims': [2],
                                              'value': np.array(flatten_shape, dtype='int64')}
            self.nodes.append({'name': self.layer.name + '_flatten', 'inputs': input_var_list + [reshape_name],
                               'outputs': [output_name], 'op_type': 'Reshape', 'attribute': {}})
            input_var_list = [output_name]
        '''

        # process the shape of bias, let [num_output, 1] to [num_output]
        if len(self.layer.blobs) == 2:
            bias_shape = self.layer.blobs[1].shape.dim
            if bias_shape == [ip_param.num_output, 1]:
                self.layer.blobs[1].shape.dim.pop()

        sub_int, param_name = self.param_process()
        self.initializer.update(sub_int)
        self.nodes.append({'name': self.layer.name, 'inputs': input_var_list + param_name,
                           'outputs': list(self.layer.top), 'op_type': self.op_type, 'attribute': self.get_attr()})
        self.infer_shape()
        return build_temp_graph(self.initializer, self.nodes, graph=self.graph)


@register_class
class ReflectionPad(CaffeOpBuilder):
    def get_attr(self):
        pad = self.layer.pad_param
        self.settings = {'pads': [pad.pad_h, pad.pad_w] * 2, 'mode': 'reflect'}
        return self.settings


@register_class
class NNUpsample(CaffeOpBuilder):
    # Upsample is deprecated in onnx-11, it's covered by Resize
    def trans(self):
        # Only support input is 4-D
        param = self.layer.nn_upsample_param
        resize = param.resize
        roi = self.layer.name + '_roi'
        scales = self.layer.name + '_scales'
        self.initializer = {roi: {'dims': [4], 'value': np.array([0, 0, 0, 0], dtype='f')},
                            scales: {'dims': [4], 'value': np.array([1, 1, resize, resize], dtype='f')}}

        self.nodes = [{'name': self.layer.name, 'inputs': list(self.layer.bottom) + [roi, scales],
                       'outputs': list(self.layer.top), 'op_type': self.op_type, 'attribute': {'mode': 'nearest'}}]

        self.infer_shape()
        return build_temp_graph(self.initializer, self.nodes, graph=self.graph)


@register_class
class Sigmoid(CaffeOpBuilder):
    pass


@register_class
class Slice(CaffeOpBuilder):
    def trans(self):
        param = self.layer.slice_param
        axis = param.axis
        axis_name = self.layer.name + '_axes'
        self.initializer[axis_name] = {'dims': [1], 'value': np.array([axis], dtype='int64')}

        slice_points = [0] + list(param.slice_point) + [self.input_shape[0][axis]]
        splits = [(slice_points[i], slice_points[i + 1]) for i in range(len(slice_points) - 1)]
        for i, pairs in enumerate(splits):
            starts = self.layer.name + '_starts_' + str(i)
            ends = self.layer.name + '_ends_' + str(i)
            self.initializer[starts] = {'dims': [1], 'value': np.array([pairs[0]], dtype='int64')}
            self.initializer[ends] = {'dims': [1], 'value': np.array([pairs[1]], dtype='int64')}
            self.nodes.append(
                {'name': self.layer.name + '_' + str(i), 'inputs': list(self.layer.bottom) + [starts, ends, axis_name],
                 'outputs': [list(self.layer.top)[i]], 'op_type': self.op_type, 'attribute': {}})
        self.infer_shape()
        return build_temp_graph(self.initializer, self.nodes, graph=self.graph)


@register_class
class Flatten(CaffeOpBuilder):
    def get_attr(self):
        flatten = self.layer.flatten_param
        self.settings['axis'] = flatten.axis
        if flatten.end_axis not in [-1, len(self.input_shape) - 1]:
            logger.error('Only support flattening inputs into 2-D matrix')
            exit(-1)
        return self.settings


@register_class
class Interp(CaffeOpBuilder):
    def trans(self):
        self.initializer = {}
        self.nodes = [{'name': self.layer.name, 'inputs': list(self.layer.bottom),
                       'outputs': list(self.layer.top), 'op_type': self.op_type, 'attribute': self.get_attr()}]
        self.infer_shape()
        return build_temp_graph(self.initializer, self.nodes, graph=self.graph)

    def get_attr(self):
        interp = self.layer.interp_param
        self.settings['shrink_factor'] = interp.shrink_factor
        self.settings['zoom_factor'] = interp.zoom_factor
        self.settings['pad_beg'] = interp.pad_beg
        self.settings['pad_end'] = interp.pad_end
        self.settings['height'] = interp.height
        self.settings['width'] = interp.width
        self.settings['align_corners'] = interp.align_corners
        self.settings['mode'] = 'linear'
        # For caffe ops which don't mapping to onnx compute in torch_backend.py
        # input_shape will be saved in attributes for dump_to_onnx func
        self.settings['input_shape'] = self.input_shape
        return self.settings


@register_class
class Tile(CaffeOpBuilder):
    def get_attr(self):
        tile = self.layer.tile_param
        self.settings['axis'] = tile.axis
        self.settings['tiles'] = tile.tiles
        return self.settings


@register_class
class SubpixelDown(CaffeOpBuilder):
    def get_attr(self):
        subpixel_down = self.layer.subpixel_down_param
        self.settings['blocksize'] = subpixel_down.downsample
        return self.settings


@register_class
class SubpixelUp(CaffeOpBuilder):
    def get_attr(self):
        subpixel_up = self.layer.subpixel_up_param
        self.settings['blocksize'] = subpixel_up.upsample
        return self.settings


@register_class
class Scale(CaffeOpBuilder):
    def get_attr(self):
        scale_param = self.layer.scale_param
        self.settings['axis'] = scale_param.axis
        self.settings['num_axes'] = scale_param.num_axes
        self.settings['bias_term'] = scale_param.bias_term
        # For caffe ops which don't mapping to onnx compute in torch_backend.py
        # input_shape will be saved in attributes for dump_to_onnx func
        self.settings['input_shape'] = self.input_shape
        return self.settings


@register_class
class TanH(CaffeOpBuilder):
    # TODO
    # parameter 'engine' is  not added yet
    pass


@register_class
class Power(CaffeOpBuilder):
    def get_attr(self):
        power_param = self.layer.power_param
        self.settings['power'] = power_param.power
        self.settings['scale'] = power_param.scale
        self.settings['shift'] = power_param.shift
        return self.settings


@register_class
# !Crop in ppl_caffe is different from caffe
class Crop(CaffeOpBuilder):
    def get_attr(self):
        crop_param = self.layer.crop_param
        self.settings['crop_param'] = crop_param
        return self.settings


@register_class
class ChannelShuffle(CaffeOpBuilder):
    def get_attr(self):
        channel_shuffle_param = self.layer.channel_shuffle_param
        self.settings['group'] = channel_shuffle_param.group
        # For caffe ops which don't mapping to onnx compute in torch_backend.py
        # input_shape will be saved in attributes for dump_to_onnx func
        self.settings['input_shape'] = self.input_shape
        return self.settings


@register_class
class InstanceNorm(CaffeOpBuilder):
    def get_attr(self):
        instance_norm_param = self.layer.instance_norm_param
        self.settings['num_features'] = instance_norm_param.num_features
        self.settings['eps'] = instance_norm_param.eps
        self.settings['affine'] = instance_norm_param.affine
        return self.settings


@register_class
class Parameter(CaffeOpBuilder):
    def get_attr(self):
        parameter_param = self.layer.parameter_param
        self.settings['m'] = parameter_param.m
        self.settings['n'] = parameter_param.n
        self.settings['batch'] = parameter_param.batch
        self.settings['channel'] = parameter_param.channel
        self.settings['height'] = parameter_param.height
        self.settings['width'] = parameter_param.width
        return self.settings


@register_class
class ArgMax(CaffeOpBuilder):
    def get_attr(self):
        argmax_param = self.layer.argmax_param
        self.settings['out_max_val'] = argmax_param.out_max_val
        self.settings['top_k'] = argmax_param.top_k
        self.settings['axis'] = argmax_param.axis if argmax_param.HasField('axis') else None
        return self.settings

@register_class
class HSwish(CaffeOpBuilder):
    pass

@register_class
class HSigmoid(CaffeOpBuilder):
    pass

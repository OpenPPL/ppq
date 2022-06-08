import numpy as np
from ppq.core.data import (DataType, convert_any_to_numpy,
                           convert_any_to_python_primary_type)
from ppq.IR import Operation
from ppq.log import NaiveLogger

from . import ppl_caffe_pb2

logger = NaiveLogger.get_logger('PPQ')
caffe_export_map = {}


def register_class(cls):
    caffe_export_map[cls.__name__] = cls
    return cls

def refine_value(attribute):
    if isinstance(attribute, DataType):
        attribute = attribute.value
    return convert_any_to_python_primary_type(attribute)

class CaffeOpExporter(object):
    onnx_to_caffe = {
        'Conv': 'Convolution',
        'BatchNormalization': 'BN',
        'Relu': 'ReLU',
        'PRelu': 'PReLU',
        'LeakyRelu': 'ReLU',
        'GlobalAveragePool': 'Pooling',
        'GlobalMaxPool': 'Pooling',
        'MaxPool': 'Pooling',
        'AveragePool': 'Pooling',
        'Mul': 'Eltwise',
        'Add': 'Eltwise',
        'Max': 'Eltwise',
        'Sub': 'Eltwise',
        'Gemm': 'InnerProduct',
        'Pad': 'ReflectionPad',
        'ConvTranspose': 'Deconvolution',
        'InstanceNormalization': 'InstanceNorm',
        'Pow': 'Power',
        'Tanh': 'TanH',
        'CaffeArgMax': 'ArgMax',
        'DepthToSpace': 'SubpixelUp',
        'SpaceToDepth': 'SubpixelDown',
        'HardSwish': 'HSwish',
        'HardSigmoid': 'HSigmoid'
    }

    def __init__(self, op: Operation):
        self.op = op
        self.op_type = self.set_type()
        self.layer = ppl_caffe_pb2.LayerParameter(type=self.op_type, name=self.op.name)

    def set_type(self):
        op_type = self.op.type
        if op_type in CaffeOpExporter.onnx_to_caffe:
            op_type = CaffeOpExporter.onnx_to_caffe[op_type]
        return op_type

    def set_attr(self):
        pass

    def parse(self) -> ppl_caffe_pb2.LayerParameter:
        self.set_attr()
        self.layer.bottom[:] = [var.name for var in self.op.inputs if not var.is_parameter]
        self.layer.top[:] = [var.name for var in self.op.outputs]

        for var in self.op.parameters:
            blob = ppl_caffe_pb2.BlobProto()
            value = var.value
            value = convert_any_to_numpy(value)
            if var.meta is not None:
                shape = var.meta.shape
                dtype = DataType.to_numpy(var.meta.dtype)
            else:
                shape, dtype = value.shape, value.dtype
            blob.shape.dim.extend(shape)
            blob.data.extend(value.astype(dtype).flat)
            self.layer.blobs.extend([blob])

        return self.layer


@register_class
class Conv(CaffeOpExporter):
    def set_attr(self):
        kernel_h, kernel_w = refine_value(self.op.attributes.get('kernel_shape'))
        stride_h, stride_w = refine_value(self.op.attributes.get('strides', [1, 1]))
        dilations_h, dilations_w = refine_value(self.op.attributes.get('dilations', [1, 1]))
        pads = refine_value(self.op.attributes.get('pads', [0, 0]))
        if len(pads) == 2:
            pad_h, pad_w = pads
        elif len(pads) == 4:
            begin_pad = pads[:2]
            end_pad = pads[2:]
            if begin_pad == end_pad:
                pad_h, pad_w = begin_pad
            else:
                logger.error('Caffe only support begin_pad == end_pad in layer')
        else:
            logger.error(f'Unsupported pads attributes with the length of {len(pads)} in Caffe')

        self.layer.convolution_param.num_output = self.op.parameters[0].value.shape[0]
        self.layer.convolution_param.group = refine_value(self.op.attributes.get('group', 1))
        self.layer.convolution_param.kernel_h = kernel_h
        self.layer.convolution_param.kernel_w = kernel_w
        self.layer.convolution_param.pad_h = pad_h
        self.layer.convolution_param.pad_w = pad_w
        self.layer.convolution_param.stride_h = stride_h
        self.layer.convolution_param.stride_w = stride_w
        self.layer.convolution_param.hole_h = dilations_h
        self.layer.convolution_param.hole_w = dilations_w

        if len(self.op.parameters) == 2:
            self.layer.convolution_param.bias_term = True
        else:
            self.layer.convolution_param.bias_term = False


@register_class
class BatchNormalization(CaffeOpExporter):
    def set_attr(self):
        self.layer.bn_param.moving_average = bool(refine_value(self.op.attributes.get('training_mode', 0)))
        self.layer.bn_param.var_eps = refine_value(self.op.attributes.get('epsilon', 1e-05))
        if self.layer.bn_param.moving_average:
            self.layer.bn_param.decay = 1 - refine_value(self.op.attributes.get('momentum', 0.9))

    def parse(self):
        super(BatchNormalization, self).parse()
        channel = len(self.layer.blobs[3].data)
        for i in range(4):
            self.layer.blobs[i].shape.ClearField('dim')
            self.layer.blobs[i].shape.dim.extend([1, channel, 1, 1])
        return self.layer


@register_class
class Relu(CaffeOpExporter):
    pass


@register_class
class PRelu(CaffeOpExporter):
    pass


@register_class
class LeakyRelu(CaffeOpExporter):
    def set_attr(self):
        self.layer.relu_param.negative_slope = refine_value(self.op.attributes.get('alpha', 0.01))


class _Pooling(CaffeOpExporter):
    def set_attr(self):
        kernel_h, kernel_w = refine_value(self.op.attributes.get('kernel_shape'))
        stride_h, stride_w = refine_value(self.op.attributes.get('strides', [1, 1]))
        ceil_mode = refine_value(self.op.attributes.get('ceil_mode', 0))
        pads = refine_value(self.op.attributes.get('pads', [0, 0]))
        if len(pads) == 2:
            pad_h, pad_w = pads
        elif len(pads) == 4:
            begin_pad = pads[:2]
            end_pad = pads[2:]
            if begin_pad == end_pad:
                pad_h, pad_w = begin_pad
            else:
                logger.error('Caffe only support begin_pad == end_pad in layer')
        else:
            logger.error(f'Unsupported pads attributes with the length of {len(pads)} in Caffe')

        self.layer.pooling_param.kernel_h = kernel_h
        self.layer.pooling_param.kernel_w = kernel_w
        self.layer.pooling_param.pad_h = pad_h
        self.layer.pooling_param.pad_w = pad_w
        self.layer.pooling_param.stride_h = stride_h
        self.layer.pooling_param.stride_w = stride_w
        if ceil_mode == 0:
            # ceil_mode is True by CaffeOpExporter in caffe
            self.layer.pooling_param.ceil_mode = False


@register_class
class GlobalAveragePool(_Pooling):
    def set_attr(self):
        self.layer.pooling_param.global_pooling = True
        self.layer.pooling_param.pool = ppl_caffe_pb2.PoolingParameter.AVE


@register_class
class AveragePool(_Pooling):
    def set_attr(self):
        super(AveragePool, self).set_attr()
        self.layer.pooling_param.global_pooling = False
        self.layer.pooling_param.pool = ppl_caffe_pb2.PoolingParameter.AVE


@register_class
class GlobalMaxPool(_Pooling):
    def set_attr(self):
        self.layer.pooling_param.global_pooling = True
        self.layer.pooling_param.pool = ppl_caffe_pb2.PoolingParameter.MAX


@register_class
class MaxPool(_Pooling):
    def set_attr(self):
        super(MaxPool, self).set_attr()
        self.layer.pooling_param.global_pooling = False
        self.layer.pooling_param.pool = ppl_caffe_pb2.PoolingParameter.MAX


@register_class
class Concat(CaffeOpExporter):
    def set_attr(self):
        self.layer.concat_param.axis = refine_value(self.op.attributes['axis'])


@register_class
class Softmax(CaffeOpExporter):
    def set_attr(self):
        axis = refine_value(self.op.attributes.get('axis', -1))
        if not (axis == -1 or axis == len(self.op.inputs[0].meta.shape) - 1):
            logger.warning(f'Converting to caffe Softmax, the axis={axis}, which is not the last axis. '
                           'This may result to incorrect caffe model')
        self.layer.softmax_param.axis = axis


@register_class
class Transpose(CaffeOpExporter):
    def set_attr(self):
        perm = refine_value(self.op.attributes['perm'])
        self.layer.transpose_param.dim.extend(perm)


@register_class
class ReduceL2(CaffeOpExporter):
    def set_attr(self):
        self.layer.reducel2_param.axes = refine_value(self.op.attributes.get('axes'))
        self.layer.reducel2_param.keepdims = refine_value(self.op.attributes.get('keepdims', 1))


@register_class
class ReduceMean(CaffeOpExporter):
    def set_attr(self):
        axis = None
        if 'axis' in self.op.attributes:
            axis = self.op.attributes.get('axis')
        elif 'axes' in self.op.attributes:
            axis = self.op.attributes.get('axes')
        if isinstance(axis, list):
            assert len(axis) == 1, (
                'You are trying to dump a RuduceMean op to caffe, '
                f'however caffe support 1 axis only, your mean operation has {len(axis)} working axis')
            axis = axis[0]
        self.layer.reduce_param.axis = axis

@register_class
class Div(CaffeOpExporter):
    pass


@register_class
class Mul(CaffeOpExporter):
    # TODO: Can optimize some case to Scale + Reshape
    #  (lhs_shape[1] == rhs_shape[1] and all(i == 1 for i in rhs_shape[2:]))
    def set_attr(self):
        self.layer.eltwise_param.operation = ppl_caffe_pb2.EltwiseParameter.PROD

    def parse(self):
        self.set_attr()
        self.layer.bottom[:] = [var.name for var in self.op.inputs if not var.is_parameter]
        self.layer.top[:] = [var.name for var in self.op.outputs]

        for var in self.op.parameters:
            value = convert_any_to_numpy(var.value)
            if value.size != 1:
                raise AttributeError(f'Now don\'t support Mul op with initializer in shape {value.shape} convert to caffe')
            # Mul only has two inputs, thus in this loop means the bottom has only one item
            self.layer.eltwise_param.coeff.append(value.item())

        return self.layer


@register_class
class Add(CaffeOpExporter):
    def set_attr(self):
        self.layer.eltwise_param.operation = ppl_caffe_pb2.EltwiseParameter.SUM
        # ONNX op only support no coeff add now
        # https://github.com/onnx/onnx/blob/master/docs/Operators.md#add
        self.layer.eltwise_param.coeff[:] = [1.0] * len(self.op.inputs)

    def parse(self):
        parameter_layers, extend_bottom = [], []
        for i, var in enumerate(self.op.parameters):
            param_layer = ppl_caffe_pb2.LayerParameter(type='Parameter', name=self.op.name + '_param_' + str(i))
            param_layer.top[:] = [var.name]
            extend_bottom.append(var.name)

            blob = ppl_caffe_pb2.BlobProto()
            value = convert_any_to_numpy(var.value)
            if var.meta is not None:
                shape = var.meta.shape
                dtype = DataType.to_numpy(var.meta.dtype)
            else:
                shape, dtype = value.shape, value.dtype
            blob.shape.dim.extend(shape)
            blob.data.extend(value.astype(dtype).flat)
            param_layer.blobs.extend([blob])

            shape_param = param_layer.parameter_param
            if len(shape) == 3:
                shape_param.batch, shape_param.m, shape_param.n = shape
            elif len(shape) == 4:
                shape_param.batch, shape_param.channel, shape_param.height, shape_param.width = shape
            else:
                raise AttributeError(f'Cannot convert {self.op.name} to Eltwise op.')

            parameter_layers.append(param_layer)

        super(Add, self).parse()
        if len(extend_bottom) != 0:
            self.layer.bottom.extend(extend_bottom)
            self.layer.eltwise_param.coeff.extend([1.0] * len(extend_bottom))
        return [*parameter_layers, self.layer]


@register_class
class Max(CaffeOpExporter):
    def set_attr(self):
        self.layer.eltwise_param.operation = ppl_caffe_pb2.EltwiseParameter.MAX


@register_class
class Sub(CaffeOpExporter):
    def set_attr(self):
        self.layer.eltwise_param.operation = ppl_caffe_pb2.EltwiseParameter.SUM
        # ONNX op only support no coeff sub now
        # https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
        self.layer.eltwise_param.coeff[:] = [1.0, -1.0]


@register_class
class Reshape(CaffeOpExporter):
    def set_attr(self):
        if len(self.op.inputs) == 0:
            raise AttributeError(f'{self.op.name} has no inputs. Cannot convert to caffe op. '
                        'Please optimize the onnx model.')
        shape = convert_any_to_numpy(self.op.parameters[0].value)
        self.layer.reshape_param.shape.dim.extend(shape)

    def parse(self):
        self.set_attr()
        self.layer.bottom[:] = [var.name for var in self.op.inputs if not var.is_parameter]
        self.layer.top[:] = [var.name for var in self.op.outputs]
        return self.layer


@register_class
class Clip(CaffeOpExporter):
    def parse(self):
        self.layer.bottom[:] = [var.name for var in self.op.inputs if not var.is_parameter]
        self.layer.top[:] = [var.name for var in self.op.outputs]

        min_val = refine_value(self.op.attributes.get('min'))
        max_val = refine_value(self.op.attributes.get('max'))
        if len(self.op.parameters) == 2:
            min_val = convert_any_to_numpy(self.op.parameters[0].value).item()
            max_val = convert_any_to_numpy(self.op.parameters[1].value).item()

        if min_val == 0.0 and max_val == 6.0:
            self.layer.type = 'ReLU6'
        else:
            self.layer.clip_param.min = min_val
            self.layer.clip_param.max = max_val
        return self.layer


@register_class
class Gemm(CaffeOpExporter):
    def parse(self):
        super(Gemm, self).parse()
        # Whether need to add transpose layer
        transpose_layer = None
        if refine_value(self.op.attributes.get('transA', 0)) != 0:
            A = self.op.inputs[0]
            shape = A.meta.shape
            if len(shape) == 2:
                transpose_layer = ppl_caffe_pb2.LayerParameter(type='Transpose', name=self.op.name + '_transposed')
                transpose_layer.bottom[:] = [A.name]
                transpose_layer.top[:] = [A.name + '_trans']
                transpose_layer.transpose_param.dim[:] = [1, 0]
                # Modify InnerProduct input
                self.layer.bottom[:] = [A.name + '_trans']
            else:
                raise ValueError('Cannot support transposed gemm with non-2D input.')

        if self.op.attributes.get('transB', 0) == 0:
            B = convert_any_to_numpy(self.op.parameters[0].value)
            BT_value = np.transpose(B, [1, 0])
            self.layer.blobs[0].shape.dim[:] = BT_value.shape
            self.layer.blobs[0].data[:] = BT_value.astype('float32').flat
        
        self.layer.inner_product_param.axis = self.op.attributes.get('axis', 1)
        self.layer.inner_product_param.num_output = self.layer.blobs[0].shape.dim[0]
        self.layer.inner_product_param.bias_term = True if len(self.op.parameters) == 2 else False

        if transpose_layer is None:
            return self.layer
        else:
            return [transpose_layer, self.layer]


@register_class
class Pad(CaffeOpExporter):
    def set_attr(self):
        mode = refine_value(self.op.attributes.get('mode', 'constant'))
        if mode != 'reflect':
            raise TypeError(f'Unsupport pad mode {mode} in caffe op')

        pads = convert_any_to_numpy(self.op.inputs[1].value) if len(self.op.inputs) > 1 else refine_value(self.op.attributes['pads'])
        if len(pads) == 2:
            pads = [pads[0], 0, pads[1], 0]
        phs, pws, phe, pwe = pads
        assert phs == phe and pws == pwe
        self.layer.pad_param.pad_h = phs
        self.layer.pad_param.pad_w = pws


# TODO: InterP and other cases
@register_class
class Resize(CaffeOpExporter):
    def __init__(self, op):
        self.mode = refine_value(op.attributes.get('mode'))
        self.scales = convert_any_to_numpy(op.inputs[2].value) if len(op.inputs) > 2 else []
        self.sizes = convert_any_to_numpy(op.inputs[-1].value) if len(op.inputs) == 4 else []
        super().__init__(op)

    def set_type(self):
        if self.mode == 'nearest' and len(self.sizes) == 0 and len(self.scales) > 0:
            return 'NNUpsample'
        elif self.mode == 'linear' and len(self.scales) == 0 and len(self.sizes) > 0:
            return 'Interp'
        else:
            raise TypeError(f'Cannot convert {self.op.name} to caffe op')

    def set_attr(self):
        if self.op_type == 'NNUpsample':
            assert len(self.scales) == 4
            valid_flag = (self.scales[0] == 1.0) and (self.scales[1] == 1.0) and (self.scales[2] == self.scales[3])
            if not valid_flag:
                raise AttributeError(f'Cannot convert {self.op.name} to NNUpsample due to different scales')
            self.layer.nn_upsample_param.resize = int(self.scales[2])
        if self.op_type == 'Interp':
            self.layer.interp_param.height = int(self.sizes[-2])
            self.layer.interp_param.width = int(self.sizes[-1])
            trans_mode = refine_value(self.op.attributes.get('coordinate_transformation_mode', 'half_pixel'))
            if trans_mode == 'align_corners':
                self.layer.interp_param.align_corners = True
            elif trans_mode == 'half_pixel':
                self.layer.interp_param.align_corners = False
            else:
                raise AttributeError(f'Cannot convert {self.op.name} in {trans_mode} mode to Interp.')

    def parse(self):
        self.set_attr()
        self.layer.bottom[:] = [var.name for var in self.op.inputs if not var.is_parameter]
        self.layer.top[:] = [var.name for var in self.op.outputs]
        return self.layer


@register_class
class ConvTranspose(CaffeOpExporter):
    def set_attr(self):
        kernel_h, kernel_w = refine_value(self.op.attributes.get('kernel_shape'))
        stride_h, stride_w = refine_value(self.op.attributes.get('strides', [1, 1]))
        dilations_h, dilations_w = refine_value(self.op.attributes.get('dilations', [1, 1]))
        pads = refine_value(self.op.attributes.get('pads', [0, 0]))
        if len(pads) == 2:
            pad_h, pad_w = pads
        elif len(pads) == 4:
            begin_pad = pads[:2]
            end_pad = pads[2:]
            if begin_pad == end_pad:
                pad_h, pad_w = begin_pad
            else:
                logger.error('Caffe only support begin_pad == end_pad in layer')
        else:
            logger.error(f'Unsupported pads attributes with the length of {len(pads)} in Caffe')

        self.layer.convolution_param.num_output = self.op.parameters[0].value.shape[1] * self.op.attributes.get('group', 1)
        self.layer.convolution_param.group = refine_value(self.op.attributes.get('group', 1))
        self.layer.convolution_param.kernel_h = kernel_h
        self.layer.convolution_param.kernel_w = kernel_w
        self.layer.convolution_param.pad_h = pad_h
        self.layer.convolution_param.pad_w = pad_w
        self.layer.convolution_param.stride_h = stride_h
        self.layer.convolution_param.stride_w = stride_w
        self.layer.convolution_param.hole_h = dilations_h
        self.layer.convolution_param.hole_w = dilations_w

        if len(self.op.parameters) == 2:
            self.layer.convolution_param.bias_term = True
        else:
            self.layer.convolution_param.bias_term = False


@register_class
class Sigmoid(CaffeOpExporter):
    pass


@register_class
class Slice(CaffeOpExporter):
    def parse(self):
        # assert (len(self.op.inputs) == 1)
        input_shape = self.op.inputs[0].meta.shape
        starts, ends = convert_any_to_numpy(self.op.parameters[0].value), convert_any_to_numpy(self.op.parameters[1].value)
        axes = convert_any_to_numpy(self.op.parameters[2].value) if len(self.op.parameters) >= 3 else [i for i in range(len(input_shape))]
        if len(self.op.parameters) >= 4 and any(convert_any_to_numpy(self.op.parameters[3].value) != 1):
            raise AttributeError('Slice op with steps cannot dump to caffe model')

        layers = []
        self.layer = ppl_caffe_pb2.LayerParameter(type=self.op_type, name=self.op.name)

        for i, (start_point, end_point, axis) in enumerate(list(zip(starts, ends, axes))):
            current_layer = ppl_caffe_pb2.LayerParameter(type=self.op_type, name=self.op.name + '_' + str(i))
            current_layer.slice_param.axis = axis
            current_layer.bottom[:] = [var.name for var in self.op.inputs if not var.is_parameter]
            current_layer.top[:] = [var.name for var in self.op.outputs]

            slice_points = [start_point, end_point]
            if start_point == 0:
                slice_points.remove(start_point)
            else:
                current_layer.top.insert(0, self.op.outputs[0].name + '_front')
            if end_point == -1 or end_point == input_shape[axis]:
                slice_points.remove(end_point)
            else:
                current_layer.top.append(self.op.outputs[-1].name + '_behind')
            current_layer.slice_param.slice_point.extend(slice_points)
            layers.append(current_layer)

        return layers


@register_class
class Tanh(CaffeOpExporter):
    pass


@register_class
class Pow(CaffeOpExporter):
    def set_attr(self):
        self.layer.power_param.power = refine_value(self.op.attributes.get('power', 1))
        self.layer.power_param.scale = refine_value(self.op.attributes.get('scale', 1))
        self.layer.power_param.shift = refine_value(self.op.attributes.get('shift', 0))


@register_class
class Scale(CaffeOpExporter):
    def set_attr(self):
        self.layer.scale_param.axis = refine_value(self.op.attributes.get('axis', 1))
        self.layer.scale_param.num_axes = refine_value(self.op.attributes.get('num_axes', 1))
        self.layer.scale_param.bias_term = refine_value(self.op.attributes.get('bias_term', False))


@register_class
class ChannelShuffle(CaffeOpExporter):
    def set_attr(self):
        self.layer.channel_shuffle_param.group = refine_value(self.op.attributes.get('group', 1))


@register_class
class InstanceNormalization(CaffeOpExporter):
    def set_attr(self):
        self.layer.instance_norm_param.num_features = refine_value(self.op.attributes.get('num_features'))
        self.layer.instance_norm_param.eps = refine_value(self.op.attributes.get('eps', 1e-5))
        self.layer.instance_norm_param.affine = refine_value(self.op.attributes.get('affine', False))


@register_class
class Parameter(CaffeOpExporter):
    def set_attr(self):
        self.layer.parameter_param.m = refine_value(self.op.attributes.get('m', -1))
        self.layer.parameter_param.n = refine_value(self.op.attributes.get('n', -1))
        self.layer.parameter_param.batch = refine_value(self.op.attributes.get('batch', 1))
        self.layer.parameter_param.channel = refine_value(self.op.attributes.get('channel', -1))
        self.layer.parameter_param.height = refine_value(self.op.attributes.get('height', -1))
        self.layer.parameter_param.width = refine_value(self.op.attributes.get('width', -1))


@register_class
class Interp(CaffeOpExporter):
    def set_attr(self):
        if refine_value(self.op.attributes.get('shrink_factor')) != 1:
            self.layer.interp_param.shrink_factor = refine_value(self.op.attributes.get('shrink_factor'))
        if refine_value(self.op.attributes.get('zoom_factor')) != 1:
            self.layer.interp_param.zoom_factor = refine_value(self.op.attributes.get('zoom_factor'))
        if refine_value(self.op.attributes.get('width')) and refine_value(self.op.attributes.get('height')):
            self.layer.interp_param.height = refine_value(self.op.attributes.get('height'))
            self.layer.interp_param.width = refine_value(self.op.attributes.get('width'))
        self.layer.interp_param.pad_beg = refine_value(self.op.attributes.get('pad_beg'))
        self.layer.interp_param.pad_end = refine_value(self.op.attributes.get('pad_end'))
        self.layer.interp_param.align_corners = refine_value(self.op.attributes.get('align_corners'))


@register_class
class Tile(CaffeOpExporter):
    def set_attr(self):
        self.layer.tile_param.axis = refine_value(self.op.attributes.get('axis'))
        self.layer.tile_param.tiles = refine_value(self.op.attributes.get('tiles'))


@register_class
class Flatten(CaffeOpExporter):
    def set_attr(self):
        self.layer.flatten_param.axis = refine_value(self.op.attributes.get('axis', 1))
        self.layer.flatten_param.end_axis = refine_value(self.op.attributes.get('end_axis', -1))


@register_class
class SpaceToDepth(CaffeOpExporter):
    def set_attr(self):
        self.layer.subpixel_down_param.downsample = refine_value(self.op.attributes.get('blocksize', 1))


@register_class
class DepthToSpace(CaffeOpExporter):
    def set_attr(self):
        self.layer.subpixel_up_param.upsample = refine_value(self.op.attributes.get('blocksize', 1))


@register_class
class CaffeArgMax(CaffeOpExporter):
    def set_attr(self):
        self.layer.argmax_param.out_max_val = refine_value(self.op.attributes.get('out_max_val'))
        self.layer.argmax_param.top_k = refine_value(self.op.attributes.get('top_k'))
        self.layer.argmax_param.axis = refine_value(self.op.attributes.get('axis'))

@register_class
class HardSwish(CaffeOpExporter):
    pass

@register_class
class HardSigmoid(CaffeOpExporter):
    pass

from collections import defaultdict

import numpy as np
from ppq.log import NaiveLogger

from . import ppl_caffe_pb2

logger = NaiveLogger.get_logger('PPQ')

def de_inplace(net_def: ppl_caffe_pb2.NetParameter) -> ppl_caffe_pb2.NetParameter:
    """Remove inplace layer in netdef If the names of bottom and top are same,
    it means the computation of this layer is in place."""

    def new_name(_name):
        if current_write_times[_name] == total_write_times[_name]:
            return _name
        else:
            return f'{_name}_ver{current_write_times[_name]}'

    total_write_times = {}
    for layer in net_def.layer:
        for top in layer.top:
            total_write_times.setdefault(top, 0)
            total_write_times[top] += 1

    current_write_times = {}
    for name in net_def.input:
        # Init input var, only read in the beginning
        total_write_times[name] = 0
        current_write_times[name] = 0

    for layer in net_def.layer:
        for i, name in enumerate(layer.bottom):
            layer.bottom[i] = new_name(name)
        for i, name in enumerate(layer.top):
            current_write_times.setdefault(name, 0)
            current_write_times[name] += 1
            layer.top[i] = new_name(name)

    return net_def

def merge_batchnorm_scale(caffe_net: ppl_caffe_pb2.NetParameter) -> ppl_caffe_pb2.NetParameter:
    new_net = ppl_caffe_pb2.NetParameter()
    new_net.CopyFrom(caffe_net)
    new_net.ClearField('layer')

    idx = 0
    while idx < len(caffe_net.layer):
        if idx != len(caffe_net.layer) - 1 and caffe_net.layer[idx].type == 'BatchNorm' and caffe_net.layer[idx + 1].type == 'Scale':
            batchnorm_layer = caffe_net.layer[idx]
            scale_layer = caffe_net.layer[idx + 1]
            if not batchnorm_layer.batch_norm_param.use_global_stats:
                logger.warning('Now cannot convert BatchNorm layer with use_global_stats==False. Change it to True')
            # Generate BN layer from batchnorm + scale
            layer = ppl_caffe_pb2.LayerParameter(type='BN', name=batchnorm_layer.name)
            layer.bottom[:] = batchnorm_layer.bottom[:]
            layer.top[:] = scale_layer.top[:]
            # Set attribute
            layer.bn_param.var_eps = batchnorm_layer.batch_norm_param.eps
            # assign blobs in order: weight, bias, mean, var
            weight_blob = ppl_caffe_pb2.BlobProto()
            weight_blob.CopyFrom(scale_layer.blobs[0])
            layer.blobs.extend([weight_blob])

            bias_blob = ppl_caffe_pb2.BlobProto()
            if scale_layer.scale_param.bias_term:
                bias_blob.CopyFrom(scale_layer.blobs[1])
                layer.blobs.extend([bias_blob])
            else:
                # bias = 0
                bias_blob.data.extend(np.zeros(len(weight_blob.data), dtype='f'))
                layer.blobs.extend([bias_blob])

            mean_blob = ppl_caffe_pb2.BlobProto()
            mean_blob.CopyFrom(batchnorm_layer.blobs[0])
            layer.blobs.extend([mean_blob])
            var_blob = ppl_caffe_pb2.BlobProto()
            var_blob.CopyFrom(batchnorm_layer.blobs[1])
            layer.blobs.extend([var_blob])
            # Update running_mean and running_var according to scale_factor
            if len(batchnorm_layer.blobs) == 3:
                factor = np.array(batchnorm_layer.blobs[2].data, dtype='f')
                layer.blobs[2].ClearField('data')
                running_mean = np.array(batchnorm_layer.blobs[0].data, dtype='f') / factor
                layer.blobs[2].data.extend(running_mean)
                layer.blobs[3].ClearField('data')
                running_var = np.array(batchnorm_layer.blobs[1].data, dtype='f') / factor
                layer.blobs[3].data.extend(running_var)

            new_net.layer.extend([layer])
            idx += 2
        else:
            layer = ppl_caffe_pb2.LayerParameter()
            layer.CopyFrom(caffe_net.layer[idx])
            new_net.layer.extend([layer])
            idx += 1
    return new_net

def optimize_for_export(caffe_net: ppl_caffe_pb2.NetParameter) -> ppl_caffe_pb2.NetParameter:
    """Simplify some caffe ops Pattern 1: combine multi-slices to one Pattern
    2: combine multi-eltwise back to one."""
    slice_combine(caffe_net)
    eltwise_combine(caffe_net)
    return caffe_net


def slice_combine(caffe_net: ppl_caffe_pb2.NetParameter):
    slice_layer = [layer for layer in caffe_net.layer if layer.type == 'Slice']
    slice_opt_set = []
    matched = []
    # Get all slice_set which can be merged together
    for i in range(len(slice_layer)):
        if slice_layer[i] in matched:
            continue
        else:
            slice_set = [slice_layer[i]]
        for j in range(i + 1, len(slice_layer)):
            if slice_layer[j] not in matched:
                same_flag = (slice_layer[i].bottom == slice_layer[j].bottom) and \
                            (slice_layer[i].slice_param.axis == slice_layer[j].slice_param.axis)
                if same_flag:
                    slice_set.append(slice_layer[j])
        if len(slice_set) > 1:
            slice_opt_set.append(slice_set)
            matched.extend(slice_set)
    # Merge Slice layer
    for layer_set in slice_opt_set:
        remained_layer = layer_set[0]
        slice_point = [layer.slice_param.slice_point for layer in layer_set] + [[0]]
        slice_point = sorted(set([item for sublist in slice_point for item in sublist]))

        name_dict = defaultdict(list)
        # An example: name_dict = {slice_point_0: ['337', '338_front'], slice_point_1: ['338', '337_behind']}
        for layer in layer_set:
            points = layer.slice_param.slice_point
            cuttings = [slice_point[slice_point.index(points[0]) - 1]] + list(points)
            for name, cut_point in zip(layer.top, cuttings):
                name_dict[cut_point].append(name)

        changed_name_dict = {}
        # An example: changed_name_dict = {'338_front': '337', '337_behind': '338'}
        new_top = []
        for names in name_dict.values():
            key = [name for name in names if ('_front' not in name) and ('_behind' not in name)]
            if not key:
                key = names[0]
                logger.warning(f'Cannot find origin name in slice combination. Use {key}')
            else:
                assert len(key) == 1, 'Find multiple origin names in slice combination. Only support one.'
                key = key[0]
            new_top.append(key)
            names.remove(key)
            changed_name_dict.update([(i, key) for i in names])

        remained_layer.top[:] = new_top
        remained_layer.slice_param.slice_point[:] = slice_point[1:]  # remove 0 in slice_point

        # update changed name
        for layer in caffe_net.layer:
            for i, item in enumerate(layer.bottom):
                if item in changed_name_dict:
                    layer.bottom[i] = changed_name_dict[item]
        # delete other slice layer
        for i in range(1, len(layer_set)):
            caffe_net.layer.remove(layer_set[i])


def eltwise_combine(caffe_net: ppl_caffe_pb2.NetParameter):
    eltwise_layer = [layer for layer in caffe_net.layer if layer.type == 'Eltwise']
    # Step 1: Combine mul op generated from coeff != 1.0 in origin caffe eltwise op
    eltwise_opt_set = []
    matched = []
    # Get all eltwise_set which can be merged together
    for i in range(len(eltwise_layer)):
        if '_mul' not in eltwise_layer[i].name:
            eltwise_set = [eltwise_layer[i]]
            find_out = eltwise_layer[i].bottom
        else:
            continue

        for layer in eltwise_layer:
            if layer not in matched:
                # Eltwise only has 1 output
                same_flag = ('_mul' in layer.name) and (layer.top[0] in find_out)
                if same_flag:
                    eltwise_set.append(layer)

        if len(eltwise_set) > 1:
            eltwise_opt_set.append(eltwise_set)
            matched.extend(eltwise_set)
    # Merge mul op
    for eltwise_set in eltwise_opt_set:
        remained_layer = [layer for layer in eltwise_set if '_mul' not in layer.name]
        assert len(remained_layer) == 1, f'Cannot find root layer for {eltwise_set[0]}'
        remained_layer = remained_layer[0]
        eltwise_set.remove(remained_layer)

        new_bottom = {}
        for layer in eltwise_set:
            assert len(layer.bottom) == len(layer.top) == 1
            new_bottom[layer.top[0]] = (layer.bottom[0], layer.eltwise_param.coeff[0])
            # remove layer
            eltwise_layer.remove(layer)
            caffe_net.layer.remove(layer)

        if len(remained_layer.eltwise_param.coeff) == 0:
            remained_layer.eltwise_param.coeff[:] = [1.0] * len(remained_layer.bottom)
        for i, item in enumerate(remained_layer.bottom):
            if item in new_bottom:
                remained_layer.bottom[i] = new_bottom[item][0]
                remained_layer.eltwise_param.coeff[i] = new_bottom[item][1]

    # Step 2:Combine binary op to multi-input eltwise
    eltwise_opt_set = []
    matched = []
    # Get all eltwise_set which can be merged together
    for layer_target in eltwise_layer:
        eltwise_set = [layer_target]
        for layer_find in eltwise_layer:
            # the layer_target must only have one successor
            same_flag = (layer_target.eltwise_param.operation == layer_find.eltwise_param.operation) and \
                (layer_target.top[0] in layer_find.bottom) and \
                len([layer for layer in caffe_net.layer if layer_target.top[0] in layer.bottom]) == 1
            if same_flag:
                eltwise_set.append(layer_find)

        if len(eltwise_set) == 2:
            eltwise_opt_set.append(eltwise_set)
            matched.extend(eltwise_set)
    # Merge binary op
    for eltwise_set in reversed(eltwise_opt_set):
        remained_layer, combine_layer = eltwise_set
        if len(remained_layer.eltwise_param.coeff) == 0:
            remained_layer.eltwise_param.coeff[:] = [1.0] * len(remained_layer.bottom)
        if len(combine_layer.eltwise_param.coeff) == 0:
            combine_layer.eltwise_param.coeff[:] = [1.0] * len(remained_layer.bottom)

        idx = list(combine_layer.bottom).index(remained_layer.top[0])
        if combine_layer.eltwise_param.coeff[idx] != 1.0:
            continue
        else:
            combine_layer.bottom.pop(idx)
            combine_layer.eltwise_param.coeff.pop(idx)
            remained_layer.bottom.extend(combine_layer.bottom)
            remained_layer.eltwise_param.coeff.extend(combine_layer.eltwise_param.coeff)
            remained_layer.top[:] = combine_layer.top[:]
            caffe_net.layer.remove(combine_layer)

from typing import List

from ppq.core import (DataType, NetworkFramework, QuantizationProperty,
                      QuantizationStates)
from ppq.IR import BaseGraph, GraphExporter, QuantableOperation

from .caffe_exporter import CaffeExporter
from .onnx_exporter import OnnxExporter
from .util import convert_value


class NCNNExporter(GraphExporter):
    
    def export_raw_quant_config(self, config_path: str, graph: BaseGraph):
        ''' ncnn table format when version <= 20220420 '''
        fd = open(config_path, 'w+')
        topo_order =  graph.topological_sort()
        for op in topo_order:
            if op.is_computing_op and isinstance(op, QuantableOperation):
                fd.write(f'{op.name}_param_0 ')
                param_cfg = op.config.input_quantization_config[1]
                assert param_cfg.state in {QuantizationStates.BAKED, QuantizationStates.ACTIVATED}\
                    and param_cfg.observer_algorithm in {'minmax', 'Minmax'} and \
                        param_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL)
                # a workaround for depthwise conv in ncnn
                # will cause mis-alignment between ppq and ncnn
                if op.type == 'Conv' and op.attributes.get('group', 1) > 1:
                    group  = op.attributes.get('group', 1)
                    scale  = param_cfg.scale.reshape(group, -1).max(dim=1)[0]
                else:
                    scale  = param_cfg.scale
                scale = convert_value(1 / scale, False, DataType.FP32)
                for s in scale:
                    fd.write('%f '% s)
                fd.write('\n')
        for op in topo_order:
            if op.is_computing_op and isinstance(op, QuantableOperation):
                fd.write(f'{op.name} ')
                input_cfg = op.config.input_quantization_config[0]
                assert input_cfg.state == QuantizationStates.ACTIVATED and\
                    input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)
                scale = convert_value(1 / input_cfg.scale, True, DataType.FP32)
                fd.write('%f '% scale)
                fd.write('\n')
        fd.close()


    def export_toml_quant_config(self, config_path: str, graph: BaseGraph):
        ''' toml is human readable format '''
        import toml
        from ppq.core.config import PPQ_CONFIG
        
        table = {'source': '{} {}'.format(PPQ_CONFIG.NAME, PPQ_CONFIG.VERSION)}
        order =  graph.topological_sort()
        
        for op in order:
            if hasattr(op, 'config'):
                item = dict()
                # avoiding Gather to Crop, we cannot deduce opr_type from opr_name
                item['type'] = op.type
                if op.type in {'Conv', 'Gemm'}:
                    param_cfg = op.config.input_quantization_config[1]
                    assert param_cfg.state in {QuantizationStates.BAKED, QuantizationStates.ACTIVATED}\
                        and param_cfg.observer_algorithm in {'minmax', 'Minmax'} and \
                            param_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL)
                    # a workaround for depthwise conv in ncnn
                    # will cause mis-alignment between ppq and ncnn
                    if op.type == 'Conv' and op.attributes.get('group', 1) > 1:
                        group  = op.attributes.get('group', 1)
                        scale  = param_cfg.scale.reshape(group, -1).max(dim=1)[0]
                    else:
                        scale  = param_cfg.scale
                    item['weight'] = convert_value(1 / scale, False, DataType.FP32)
                    
                    input_cfg = op.config.input_quantization_config[0]
                    assert input_cfg.state == QuantizationStates.ACTIVATED and \
                        input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)
                    item['input_scale'] = convert_value(1 / input_cfg.scale, True, DataType.FP32)

                elif op.type in {'Add'}:
                    # Add may have multiple input node
                    input_scale = []
                    zero_point = []
                    
                    for cfg in op.config.input_quantization_config:
                        assert cfg.state in {QuantizationStates.BAKED, QuantizationStates.ACTIVATED,  QuantizationStates.SLAVE} \
                            and cfg.observer_algorithm in {'minmax', 'Minmax'}
                        input_scale.append(convert_value(1.0 / cfg.scale, True, DataType.FP32))
                        zero_point.extend(convert_value(cfg.offset, False, DataType.INT32))
                    
                    item['input_scale'] = input_scale
                    item['zero_point'] = zero_point

                elif op.type in {'LayerNorm', 'Gelu'}:
                    cfg = op.config.input_quantization_config[0]

                    assert cfg.state in {QuantizationStates.BAKED, QuantizationStates.ACTIVATED} \
                        and cfg.observer_algorithm in {'minmax', 'Minmax'}
                    item['input_scale'] = convert_value(1.0 / cfg.scale, False, DataType.FP32)
                    item['zero_point'] = convert_value(cfg.offset, False, DataType.INT32)

                elif op.type == 'MultiHeadAttention':
                    # write input scale
                    cfg_q_in = op.config.input_quantization_config[0]
                    cfg_k_in = op.config.input_quantization_config[1]
                    cfg_v_in = op.config.input_quantization_config[2]
                    
                    item['input_scale_q'] =  convert_value(1.0 / cfg_q_in.scale, True, DataType.FP32)
                    item['input_scale_k'] =  convert_value(1.0 / cfg_k_in.scale, True, DataType.FP32)
                    item['input_scale_v'] =  convert_value(1.0 / cfg_v_in.scale, True, DataType.FP32)
                    
                    # write input/output weight scale, per-channel
                    cfg_q_w = op.config.input_quantization_config[3]
                    cfg_k_w = op.config.input_quantization_config[5]
                    cfg_v_w = op.config.input_quantization_config[7]
                    cfg_o_w = op.config.input_quantization_config[9]
                    
                    item['weight_q'] = convert_value(1 / cfg_q_w.scale, False, DataType.FP32)
                    item['weight_k'] = convert_value(1 / cfg_k_w.scale, False, DataType.FP32)
                    item['weight_v'] = convert_value(1 / cfg_v_w.scale, False, DataType.FP32)
                    item['weight_o'] = convert_value(1 / cfg_o_w.scale, False, DataType.FP32)
                    
                    # write internal scale
                    cfg_q = op.config.output_quantization_config[1]
                    cfg_k = op.config.output_quantization_config[2]
                    cfg_v = op.config.output_quantization_config[3]
                    cfg_energy = op.config.output_quantization_config[4]
                    cfg_feat = op.config.output_quantization_config[5]

                    item['internal_scale_q'] =  convert_value(1.0 / cfg_q.scale, True, DataType.FP32)
                    item['internal_scale_k'] =  convert_value(1.0 / cfg_k.scale, True, DataType.FP32)
                    item['internal_scale_v'] =  convert_value(1.0 / cfg_v.scale, True, DataType.FP32)
                    item['internal_scale_energy'] =  convert_value(1.0 / cfg_energy.scale, True, DataType.FP32)
                    item['internal_scale_feat'] =  convert_value(1.0 / cfg_feat.scale, True, DataType.FP32)
                
                else:
                    print('unknown quant type {} name {} during write weight scale'.format(op.type, op.name))

                table[op.name] = item

        toml.dump(table, open(config_path, 'w+'))


    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        toml_style = True
        if toml_style:
            self.export_toml_quant_config(config_path=config_path, graph=graph)
        else:
            self.export_raw_quant_config(config_path=config_path, graph=graph)


    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):
        if config_path is not None:
            self.export_quantization_config(config_path, graph)

        # no pre-determined export format, we export according to the
        # original model format
        if graph._built_from == NetworkFramework.CAFFE:
            exporter = CaffeExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None, input_shapes=input_shapes)

        elif graph._built_from == NetworkFramework.ONNX:
            exporter = OnnxExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None)

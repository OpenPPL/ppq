from google.protobuf import text_format
from ppq.core import NetworkFramework, is_file_exist
from ppq.IR import BaseGraph, GraphBuilder
from ppq.log import NaiveLogger

from .caffe import ppl_caffe_pb2
from .caffe.caffe_graph_optim import de_inplace, merge_batchnorm_scale
from .caffe.caffe_import_utils import caffe_import_map, get_input_shape

logger = NaiveLogger.get_logger('PPQ')

class CaffeParser(GraphBuilder):
    def load_graph_and_format(self, prototxt_path: str, caffemodel_path: str) -> ppl_caffe_pb2.NetParameter:
        if not is_file_exist(prototxt_path):
            raise FileNotFoundError(f'file {prototxt_path} not exist, please check your file path')
        elif not is_file_exist(caffemodel_path):
            raise FileNotFoundError(f'file {caffemodel_path} not existm please check your file path')
        network = ppl_caffe_pb2.NetParameter()
        with open(prototxt_path) as f:
            text_format.Merge(f.read(), network)
        weight = ppl_caffe_pb2.NetParameter()
        with open(caffemodel_path, 'rb') as f:
            weight.ParseFromString(f.read())

        network = de_inplace(network)

        for i in network.layer:
            for j in weight.layer:
                if i.name == j.name:
                    i.ClearField('blobs')
                    i.blobs.MergeFrom(j.blobs)
                    break

        network = merge_batchnorm_scale(network)
        return network

    def build(self, prototxt_path: str, caffemodel_path: str) -> BaseGraph:
        network = self.load_graph_and_format(prototxt_path, caffemodel_path)
        graph = BaseGraph(name=network.name, built_from=NetworkFramework.CAFFE)
        input_shape = get_input_shape(network)
        input_names = list(input_shape.keys())

        activation_shape = input_shape
        top_name_set = set()
        for layer in network.layer:
            if layer.type not in caffe_import_map:
                logger.error(f'{layer.type} Caffe OP is not supported in PPQ import parser yet')
                raise NotImplementedError(f'{layer.type} Caffe OP is not supported in PPQ import parser yet')
            input_shape = [activation_shape[k] for k in layer.bottom]
            caffe_layer = caffe_import_map[layer.type](graph, layer, input_shape)
            graph = caffe_layer.trans()
            activation_shape.update([(k, v) for k, v in zip(layer.top, caffe_layer.out_shape)])

            # statistic top_name and get final out var name
            for name in layer.bottom:
                if name in top_name_set:
                    top_name_set.remove(name)
            for name in layer.top:
                top_name_set.add(name)

        # add input and output for graph
        try:
            for var_name in input_names:
                if var_name not in graph.variables: continue
                graph.inputs[var_name] = graph.variables[var_name]
            for var_name in top_name_set:
                graph.outputs[var_name] = graph.variables[var_name]
        except KeyError as e:
            raise KeyError(
                'seems you got an input/output variable that is not linked to any operation.')
        return graph

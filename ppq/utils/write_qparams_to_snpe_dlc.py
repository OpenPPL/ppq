import argparse
import json
import os
import snpe
import qti.aisw.dlc_utils as dlc

parser = argparse.ArgumentParser(description='Write ppq qparams to snpe dlc')
parser.add_argument('--input_dlc_model', default='snpe_quantized.dlc', help='path to snpe quantized dlc model')
parser.add_argument('--output_dlc_model', default='ppq_export.dlc', help='path to export quantized dlc')
parser.add_argument('--qparam', default='quantized.json', help='path to ppq qparams json')

def json_load(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def write_qparams_to_dlc_model(input_dlc, output_dlc, activation_qparams):
    model = dlc.modeltools.Model()
    model.load(input_dlc)
    model.set_tf_encoding_type("TF")

    for snpe_layer in model.get_layers():
        print('\n write qparams to {}'.format(snpe_layer['name']))
        for snpe_layer_out_ind, snpe_layer_out in enumerate(snpe_layer['output_names']):
            layer_name = snpe_layer['name']
            print('original quant encodings : ', model.get_tf_output_encoding_by_index(name=layer_name, index=snpe_layer_out_ind))
            top = snpe_layer['output_names'][0]

            if top not in activation_qparams.keys():
                # Before the Reshape layer, SNPE will insert the shape conversion layer(xxx.ncs)
                # Because the SNPE data is arranged as NHWC
                assert top.endswith('.ncs'), '{} ranges not exists'.format(top)
                bottom = snpe_layer['input_names'][0]
                new_enc = activation_qparams[bottom][0] #List[dict]
            else:
                new_enc = activation_qparams[top][0] #List[dict]

            model.set_tf_output_encoding_by_index(name=layer_name, index=snpe_layer_out_ind, bitwidth=8, min=new_enc["min"], max=new_enc["max"])
            print('ppq quant encodings : ', model.get_tf_output_encoding_by_index(name=layer_name, index=snpe_layer_out_ind))
    model.quantize_weights(should_quantize=True)
    model.save(output_dlc)

if __name__ == '__main__':
    args = parser.parse_args()
    act_ranges = json_load(args.qparam)['activation_encodings']
    write_qparams_to_dlc_model(args.input_dlc_model, args.output_dlc_model, act_ranges)

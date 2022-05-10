from tmodel  import *
from tscheme import *
from ppq     import *
from ppq.api import *
import sys

DEVICE = 'cuda'

for scheme in TEST_SCHEMES:
    for case in TORCH_TEST_CASES:
        try:
            print(f'PPQ System test start with model {case.model_name}, Scheme: {scheme.name}')
            dataset = [case.input_generator().to(DEVICE) for _ in range(8)]
            model = case.model_builder().to(DEVICE)

            quantized = quantize_torch_model(
                model=model,
                calib_dataloader=dataset,
                calib_steps=8,
                input_shape=case.input_generator().shape,
                platform=scheme.quant_platform,
                setting=scheme.setting)

            if (case.deploy_platforms is None or
                scheme.export_platform in case.deploy_platforms):
                export_ppq_graph(
                    graph=quantized,
                    platform=scheme.export_platform,
                    graph_save_to='tworkingspace/export',
                    config_save_to='tworkingspace/export.json')
        except NotImplementedError as e:
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} | Error occurred: {e}')
            sys.exit(1)

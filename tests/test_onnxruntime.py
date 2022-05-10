import onnxruntime
from ppq.IR.morph import GraphFormatter
from tests.tmodel  import *
from tests.tscheme import *
from ppq     import *
from ppq.api import *
from ppq import layerwise_error_analyse
import sys

DEVICE = 'cuda'
PLATFORM = TargetPlatform.ORT_OOS_INT8

for case in TORCH_TEST_CASES:
    try:
        print(f'PPQ System test(Onnxruntime) start with model {case.model_name}')
        dataset = [case.input_generator().to(DEVICE) for _ in range(8)]
        model = case.model_builder().to(DEVICE)

        quantized = quantize_torch_model(
            model=model,
            calib_dataloader=dataset,
            calib_steps=8,
            input_shape=case.input_generator().shape,
            platform=PLATFORM,
            setting=QuantizationSettingFactory.default_setting())

        executor = TorchExecutor(quantized)
        sample_output = [(executor(sample)[0].cpu().unsqueeze(0)) for sample in dataset]

        export_ppq_graph(
            graph=quantized,
            platform=TargetPlatform.ONNXRUNTIME,
            graph_save_to='onnxruntime',
            config_save_to='export.json')

        # graph has only 1 input and output.
        for name in quantized.outputs: output_name = name
        for name in quantized.inputs: input_name = name
        session = onnxruntime.InferenceSession('onnxruntime.onnx', providers=onnxruntime.get_available_providers())
        onnxruntime_outputs = [session.run([output_name], {input_name: convert_any_to_numpy(sample)}) for sample in dataset]
        onnxruntime_outputs = [convert_any_to_torch_tensor(sample) for sample in onnxruntime_outputs]

        error = []
        for ref, real in zip(sample_output, onnxruntime_outputs):
            error.append(torch_snr_error(ref, real))
        error = sum(error) / len(error) * 100

        print(f'Simulating Error: {error: .4f}%')
        assert error < 1
    except NotImplementedError as e:
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} | Error occurred: {e}')
        sys.exit(1)

for case in TORCH_TEST_CASES:
    try:
        print(f'PPQ System test(Onnxruntime) start with model {case.model_name}')
        dataset = [case.input_generator().to(DEVICE) for _ in range(8)]
        model = case.model_builder().to(DEVICE)

        quantized = quantize_torch_model(
            model=model,
            calib_dataloader=dataset,
            calib_steps=8,
            input_shape=case.input_generator().shape,
            platform=PLATFORM,
            setting=QuantizationSettingFactory.default_setting())

        '''
        quantized.outputs.clear()
        quantized.mark_variable_as_graph_output(quantized.variables['onnx::Conv_26'])
        processor = GraphFormatter(quantized)
        processor.truncate_on_var(quantized.variables['onnx::Conv_26'], mark_as_output=True)
        '''

        executor = TorchExecutor(quantized)
        sample_output = [(executor(sample)[0].cpu().unsqueeze(0)) for sample in dataset]

        export_ppq_graph(
            graph=quantized,
            platform=TargetPlatform.ORT_OOS_INT8,
            graph_save_to='onnx',
            config_save_to='export.json')

        # graph has only 1 input and output.
        for name in quantized.outputs: output_name = name
        for name in quantized.inputs: input_name = name
        session = onnxruntime.InferenceSession('onnx.onnx', providers=onnxruntime.get_available_providers())
        onnxruntime_outputs = [session.run([output_name], {input_name: convert_any_to_numpy(sample)}) for sample in dataset]
        onnxruntime_outputs = [convert_any_to_torch_tensor(sample) for sample in onnxruntime_outputs]

        error = []
        for ref, real in zip(sample_output, onnxruntime_outputs):
            error.append(torch_snr_error(ref, real))
        error = sum(error) / len(error) * 100

        print(f'Simulating Error: {error: .4f}%')
        # assert error < 1

    except NotImplementedError as e:
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} | Error occurred: {e}')
        sys.exit(1)

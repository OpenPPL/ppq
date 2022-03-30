import os
import shutil

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppq import *
from ppq.api import *
from ppq.api import quantize_caffe_model
from ppq.quantization.analyise.util import MeasurePrinter, MeasureRecorder
from ppq.utils.fetch import tensor_random_fetch


def load_calibration_dataset(directory: str, input_shape: List[int], batchsize: int) -> Iterable:
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise FileNotFoundError(f'Can not load calibration dataset from path {directory}. '
                                'It is not a valid directory, check your input again.')

    num_of_file, samples, sizes = 0, [], set()
    for file in os.listdir(os.path.join(directory, 'data')):
        sample = None
        if file.endswith('.npy'):
            sample = np.load(os.path.join(directory, 'data', file))
            num_of_file += 1
        elif file.endswith('.bin'):
            sample = np.fromfile(os.path.join(directory, 'data', file), dtype=np.float)
            sample = sample.reshape(input_shape)
            num_of_file += 1
        else:
            ppq_warning(f'文件格式不可读: {os.path.join(directory, "data", file)}, 该文件已经被忽略.')

        sample = convert_any_to_torch_tensor(sample)
        sample = sample.float()
        if sample.ndim == 3: sample = sample.unsqueeze(0)
        assert sample.shape[1] == 1 or sample.shape[1] == 3 or sample.shape[1] == 4, (
            f'你的文件 {os.path.join(directory, file)} 拥有 {sample.shape[1]} 个输入通道. '
            '这是合理的图像文件吗?(是否忘记了输入图像应当是 NCHW 的)')

        assert sample.shape[0] == 1 or batchsize == 1, (
            f'你的输入图像似乎已经有了预设好的 batchsize, 因此我们不会再尝试对你的输入进行打包。')

        sizes.add((sample.shape[-2], sample.shape[-1]))
        samples.append(sample)

    if len(sizes) != 1:
        ppq_warning('你的输入图像似乎包含动态的尺寸，因此 CALIBRATION BATCHSIZE 被强制设置为 1')
        batchsize = 1

    # create batches
    batches, batch = [], []
    if batchsize != 1:
        for sample in samples:
            if len(batch) < batchsize:
                batch.append(sample)
            else:
                batches.append(torch.cat(batch, dim=0))
                batch = [sample]
        if len(batch) != 0:
            batches.append(torch.cat(batch, dim=0))
    else:
        batches = samples

    print(f'{num_of_file} File(s) Loaded.')
    for idx, tensor in enumerate(samples[: 5]):
        print(f'Loaded sample {idx}, shape: {tensor.shape}')
    assert len(batches) > 0, '你送入了空的数据集'
    
    print(f'Batch Shape: {batches[0].shape}')
    return batches


def quantize(working_directory: str, setting: QuantizationSetting, model_type: NetworkFramework,
             executing_device: str, input_shape: List[int], target_platform: TargetPlatform,
             dataloader: DataLoader, calib_steps: int=32) -> BaseGraph:
    if model_type == NetworkFramework.ONNX:
        if not os.path.exists(os.path.join(working_directory, 'model.onnx')):
            raise FileNotFoundError(f'无法找到你的模型: {os.path.join(working_directory, "model.onnx")}，'
                                    '如果你使用caffe的模型，请设置MODEL_TYPE为CAFFE')
        return quantize_onnx_model(
            onnx_import_file=os.path.join(working_directory, 'model.onnx'),
            calib_dataloader=dataloader, calib_steps=calib_steps, input_shape=input_shape, setting=setting,
            platform=target_platform, device=executing_device, collate_fn=lambda x: x.to(executing_device)
        )
    if model_type == NetworkFramework.CAFFE:
        if not os.path.exists(os.path.join(working_directory, 'model.caffemodel')):
            raise FileNotFoundError(f'无法找到你的模型: {os.path.join(working_directory, "model.caffemodel")}，'
                                    '如果你使用ONNX的模型，请设置MODEL_TYPE为ONNX')
        return quantize_caffe_model(
            caffe_proto_file=os.path.join(working_directory, 'model.prototxt'),
            caffe_model_file=os.path.join(working_directory, 'model.caffemodel'),
            calib_dataloader=dataloader, calib_steps=calib_steps, input_shape=input_shape, setting=setting,
            platform=target_platform, device=executing_device, collate_fn=lambda x: x.to(executing_device)
        )


def export(working_directory: str, quantized: BaseGraph, platform: TargetPlatform):
    if platform == TargetPlatform.NATIVE:
        export_ppq_graph(
            graph=quantized, platform=platform,
            graph_save_to=os.path.join(working_directory, 'quantized.native'),
            config_save_to=os.path.join(working_directory, 'quantized.config')
        )
    elif platform == TargetPlatform.CAFFE:
        export_ppq_graph(
            graph=quantized, platform=platform,
            graph_save_to=os.path.join(working_directory, 'quantized.caffe'),
            config_save_to=os.path.join(working_directory, 'quantized.json')
        )
    else:
        export_ppq_graph(
            graph=quantized, platform=platform,
            graph_save_to=os.path.join(working_directory, 'quantized.onnx'),
            config_save_to=os.path.join(working_directory, 'quantized.json')
        )


def load_from_file(
    filename: str, dtype: np.dtype = np.float32, 
    shape: List[int] = None, format: str = '.npy') -> np.ndarray:
    if not os.path.exists(filename) or os.path.isdir(filename):
        raise FileNotFoundError(f'Can not load data from file: {filename}. '
                                'File not exist or it is a directory.')

    if format == '.npy':
        return np.load(filename)
    elif format == '.dat':
        raw = np.fromfile(file=filename, dtype=dtype)
        if shape is not None: raw = np.reshape(raw, shape)
        return raw
    else:
        raise ValueError(
            'file format not understand, support .npy and .dat only, '
            f'{format} was given.')


def dump_to_file(
    filename: str, data: Union[torch.Tensor, np.ndarray], 
    format: str = '.npy') -> None:
    if os.path.isdir(filename):
        raise FileExistsError(f'Can not dump data to file: {filename}',
                              ' Cause it is a directory.')
    if os.path.exists(filename):
        ppq_warning(f'Overwritting file {filename} ...')
    
    data = convert_any_to_numpy(data)
    if format == '.npy':
        np.save(file=filename, arr=data)
    elif format == '.dat':
        data.tofile(os.path.join(filename))
    else:
        raise ValueError(
            'file format not understand, support .npy and .dat only, '
            f'{format} was given.')


def create_dir(dir: str):
    try: os.mkdir(dir)
    except Exception as e: pass
    finally:
        if not os.path.isdir(dir):
            raise FileNotFoundError(f'Can not create directory at {dir}')


def compare_cosine_similarity_between_results(
    ref_dir: str, target_dir: str, method: str = 'cosine'):
    print(f'正从 {ref_dir} 与 {target_dir} 加载数据 ...')
    if not os.path.exists(ref_dir) or not os.path.isdir(ref_dir):
        raise FileNotFoundError(f'找不到路径 {ref_dir}')
    if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
        raise FileNotFoundError(f'找不到路径 {target_dir}')

    samples, outputs = [], []
    for sample_name in os.listdir(target_dir):
        if os.path.isdir(os.path.join(target_dir, sample_name)):
            samples.append(sample_name)

    for output in os.listdir(os.path.join(target_dir, samples[-1])):
        outputs.append(output)

    print(f'总计 {len(samples)} 个 Sample 被检出，{len(outputs)} 个待测输出 Variable')
    recorders = {name: MeasureRecorder(measurement=method) for name in outputs}
    
    for sample_name in samples:
        for output in outputs:

            ref_file    = os.path.join(ref_dir, sample_name, output)
            target_file = os.path.join(target_dir, sample_name, output)
            
            ref    = load_from_file(filename=ref_file, format='.dat')[:1000]
            target = load_from_file(filename=target_file, format='.dat')[:1000]

            recorders[output].update(
                y_real=convert_any_to_torch_tensor(ref), 
                y_pred=convert_any_to_torch_tensor(target))

    results = {}
    for output, recorder in recorders.items():
        results[output] = recorder.measure
    
    method_str = 'MEASUREMENT'
    if method == 'snr': method_str = 'NOISE:SIGNAL POWER RATIO'
    if method == 'cosine': method_str = 'COSINE SIMILARITY'
    if method == 'mse': method_str = 'MSE LOSS(UNSCALED)'
    MeasurePrinter(results, order='large_to_small', measure=method_str).print()


def dump_internal_results(
    graph: BaseGraph, dataloader: Iterable, 
    dump_dir: str, executing_device: str, sample: bool = True):

    i_dir  = os.path.join(dump_dir, 'inputs')
    o_dir  = os.path.join(dump_dir, 'outputs')

    create_dir(i_dir)
    create_dir(o_dir)

    # 找出所有量化点，抽出所有中间结果.
    for var in graph.variables.values():
        if isinstance(var, QuantableVariable):
            if (var.source_op_config is not None and 
                var.source_op_config.state == QuantizationStates.ACTIVATED):
                graph.outputs[var.name] = var # 直接标记为网络输出

    executor = TorchExecutor(graph, device=executing_device)
    for batch_idx, batch in tqdm(enumerate(dataloader), 
                                 total=len(dataloader), desc='Dumping Results ...'):
        batch = batch.to(executing_device)
        outputs = executor.forward(batch)

        create_dir(os.path.join(o_dir, str(batch_idx)))
        for name, output in zip(graph.outputs, outputs):

            # 如果数字太多就抽样
            if output.numel() > 10000 and sample:
                output = tensor_random_fetch(
                    tensor=output, seed=10086, # 保证随机种子一致才能比对结果 
                    num_of_fetches=4096)

            dump_to_file(
                filename=os.path.join(o_dir, str(batch_idx), name + '.dat'), 
                data=output, format='.dat')

        dump_to_file(
            filename=os.path.join(i_dir, str(batch_idx) + '.npy'),
            data=batch, format='.npy')

def split_result_to_directory(raw_dir: str, to_dir: str):
    print(f'正从 {raw_dir} 分割数据')
    data_files, sample_names = [], set()
    for file_name in os.listdir(raw_dir):
        if file_name.endswith('.dat'): 
            sample_name = file_name.split('_')[0]
            sample_names.add(sample_name)
            data_files.append((sample_name, file_name))
        
    print(f'加载 {len(data_files)} 数据文件，总计 {len(sample_names)} 个样本')
    create_dir(to_dir)
    for sample_name in sample_names:
        create_dir(os.path.join(to_dir, sample_name))
    for sample_name, file_name in tqdm(data_files, total=len(data_files), desc='正复制文件中 ...'):
        file_path = os.path.join(raw_dir, file_name)
        processed = os.path.join(to_dir, sample_name, file_name[file_name.index('_') + 1: ])
        shutil.copy(file_path, processed)

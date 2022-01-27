import os
import numpy as np
from ppq import *
from ppq.api import *
from ppq.api import quantize_caffe_model
from torch.utils.data import DataLoader


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
             dataloader: DataLoader) -> BaseGraph:
    if model_type == NetworkFramework.ONNX:
        if not os.path.exists(os.path.join(working_directory, 'model.onnx')):
            raise FileNotFoundError(f'无法找到你的模型: {os.path.join(working_directory, "model.onnx")}，'
                                    '如果你使用caffe的模型，请设置MODEL_TYPE为CAFFE')
        return quantize_onnx_model(
            onnx_import_file=os.path.join(working_directory, 'model.onnx'),
            calib_dataloader=dataloader, calib_steps=32, input_shape=input_shape, setting=setting,
            platform=target_platform, device=executing_device, collate_fn=lambda x: x.to(executing_device)
        )
    if model_type == NetworkFramework.CAFFE:
        if not os.path.exists(os.path.join(working_directory, 'model.caffemodel')):
            raise FileNotFoundError(f'无法找到你的模型: {os.path.join(working_directory, "model.caffemodel")}，'
                                    '如果你使用ONNX的模型，请设置MODEL_TYPE为ONNX')
        return quantize_caffe_model(
            caffe_proto_file=os.path.join(working_directory, 'model.prototxt'),
            caffe_model_file=os.path.join(working_directory, 'model.caffemodel'),
            calib_dataloader=dataloader, calib_steps=32, input_shape=input_shape, setting=setting,
            platform=target_platform, device=executing_device, collate_fn=lambda x: x.to(executing_device)
        )


def export(working_directory: str, quantized: BaseGraph, platform: TargetPlatform):
    export_ppq_graph(
        graph=quantized, platform=platform,
        graph_save_to=os.path.join(working_directory, 'quantized.onnx'),
        config_save_to=os.path.join(working_directory, 'quantized.config')
    )
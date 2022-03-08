import os
from typing import Any, Callable, List

import torch
from ppq.core import (NetworkFramework, TargetPlatform, empty_ppq_cache,
                      ppq_warning)
from ppq.executor import TorchExecutor
from ppq.IR import (BaseGraph, GraphCommand, GraphCommandType, GraphFormatter,
                    GraphMerger)
from ppq.IR.morph import GraphDeviceSwitcher
from ppq.parser import dump_graph_to_file, load_graph
from ppq.quantization.quantizer import (ACADEMIC_INT4_Quantizer,
                                        ACADEMIC_Mix_Quantizer,
                                        ACADEMICQuantizer, BaseQuantizer,
                                        ExtQuantizer,
                                        MetaxChannelwiseQuantizer,
                                        MetaxTensorwiseQuantizer,
                                        NXP_Quantizer, ORT_PerChannelQuantizer,
                                        ORT_PerTensorQuantizer,
                                        PPL_DSP_Quantizer,
                                        PPLCUDA_INT4_Quantizer,
                                        PPLCUDAMixPrecisionQuantizer,
                                        PPLCUDAQuantizer, TensorRTQuantizer)
from ppq.scheduler import DISPATCHER_TABLE
from torch.utils.data import DataLoader

from .setting import *

QUANTIZER_COLLECTION = {
    TargetPlatform.DSP_INT8: PPL_DSP_Quantizer,
    TargetPlatform.TRT_INT8: TensorRTQuantizer,
    TargetPlatform.NXP_INT8: NXP_Quantizer,
    TargetPlatform.ORT_OOS_INT8: ORT_PerTensorQuantizer,
    TargetPlatform.METAX_INT8_C: MetaxChannelwiseQuantizer,
    TargetPlatform.METAX_INT8_T: MetaxTensorwiseQuantizer,
    # TargetPlatform.ORT_OOS_INT8: ORT_PerChannelQuantizer,
    TargetPlatform.PPL_CUDA_INT8: PPLCUDAQuantizer,
    TargetPlatform.EXTENSION: ExtQuantizer,
    TargetPlatform.PPL_CUDA_MIX: PPLCUDAMixPrecisionQuantizer,
    TargetPlatform.PPL_CUDA_INT4: PPLCUDA_INT4_Quantizer,
    TargetPlatform.ACADEMIC_INT8: ACADEMICQuantizer,
    TargetPlatform.ACADEMIC_INT4: ACADEMIC_INT4_Quantizer,
    TargetPlatform.ACADEMIC_MIX: ACADEMIC_Mix_Quantizer
}

def load_onnx_graph(onnx_import_file: str, setting: QuantizationSetting) -> BaseGraph:
    """
        从一个指定位置加载 onnx 计算图
        load onnx graph from the specified location
    Args:
        onnx_import_file (str): onnx 计算图的保存位置 the specified location

    Returns:
        BaseGraph: 解析 onnx 获得的 ppq 计算图对象 the parsed ppq IR graph
    """
    ppq_ir = load_graph(onnx_import_file, from_framework=NetworkFramework.ONNX)
    return format_graph(graph=ppq_ir, setting=setting)

def load_caffe_graph(prototxt_path: str, caffemodel_path: str, 
                     setting: QuantizationSetting) -> BaseGraph:
    """
        从一个指定位置加载 caffe 计算图
        load caffe graph from the specified location
    Args:
        prototxt_path (str): caffe prototxt的保存位置 the specified location of caffe prototxt
        caffemodel_path (str): caffe weight的保存位置 the specified lcoation of caffe weight

    Returns:
        BaseGraph: 解析 caffe 获得的 ppq 计算图对象 the parsed ppq IR graph
    """
    ppq_ir = load_graph(file_path=prototxt_path, caffemodel_path=caffemodel_path, from_framework=NetworkFramework.CAFFE)
    return format_graph(graph=ppq_ir, setting=setting)

def dump_torch_to_onnx(
    model: torch.nn.Module, 
    onnx_export_file: str, 
    input_shape: List[int], 
    input_dtype: torch.dtype, 
    inputs: List[Any] = None,
    device: str = 'cuda'):
    """
        转换一个 torch 模型到 onnx，并保存到指定位置
        convert a torch model to onnx and save to the specified location
    Args:
        model (torch.nn.Module): 被转换的 torch 模型 torch model used for conversion

        onnx_export_file (str): 保存文件的路径 the path to save onnx model

        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
            a list of ints indicating size of input, for multiple inputs, please use keyword arg inputs for 
            direct parameter passing and this should be set to None 

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                   the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                   for direct parameter passing and this should be set to None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                    for multiple inputs, please give the specified inputs directly in the form of
                                    a list of arrays

        device (str, optional): 转换过程的执行设备 the execution device, defaults to 'cuda'.
    """

    # set model to eval mode, stablize normalization weights.
    assert isinstance(model, torch.nn.Module), (
        f'Model must be instance of torch.nn.Module, however {type(model)} is given.')
    model.eval()

    if inputs is None:
        dummy_input = torch.zeros(size=input_shape, device=device, dtype=input_dtype)
    else: dummy_input = inputs

    torch.onnx.export(
        model=model, args=dummy_input,
        verbose=False, f=onnx_export_file, opset_version=11,
    )

@ empty_ppq_cache
def quantize_onnx_model(
    onnx_import_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    input_dtype: torch.dtype = torch.float,
    inputs: List[Any] = None,
    setting: QuantizationSetting = None,
    collate_fn: Callable = None,
    platform: TargetPlatform = TargetPlatform.DSP_INT8,
    device: str = 'cuda',
    verbose: int = 0,
    do_quantize: bool = True,
) -> BaseGraph:
    """
        量化一个 onnx 原生的模型
            输入一个 onnx 模型的文件路径
            返回一个量化后的 PPQ.IR.BaseGraph
        quantize onnx model, input onnx model and return quantized ppq IR graph

    Args:
        onnx_import_file (str): 被量化的 onnx 模型文件路径 onnx model location
 
        calib_dataloader (DataLoader): 校准数据集 calibration data loader

        calib_steps (int): 校准步数 calibration steps

        collate_fn (Callable): 校准数据的预处理函数 batch collate func for preprocessing
        
        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                a list of ints indicating size of input, for multiple inputs, please use 
                                keyword arg inputs for direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                for direct parameter passing and this should be set to None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                for multiple inputs, please give the specified inputs directly in the form of
                                a list of arrays

        setting (OptimSetting): 量化配置信息，用于配置量化的各项参数，设置为 None 时加载默认参数。
                                Quantization setting, default setting will be used when set None

        do_quantize (Bool, optional): 是否执行量化 whether to quantize the model, defaults to True.


        platform (TargetPlatform, optional): 量化的目标平台 target backend platform, defaults to TargetPlatform.DSP_INT8.
                                        
        device (str, optional): 量化过程的执行设备 execution device, defaults to 'cuda'.

        verbose (int, optional): 是否打印详细信息 whether to print details, defaults to 0.

    Raises:
        ValueError: 给定平台不可量化 the given platform doesn't support quantization
        KeyError: 给定平台不被支持 the given platform is not supported yet

    Returns:
        BaseGraph: 量化后的IR，包含了后端量化所需的全部信息 
                   The quantized IR, containing all information needed for backend execution
    """
    if not TargetPlatform.is_quantized_platform(platform=platform):
        raise ValueError(f'Target Platform {platform} is an non-quantable platform.')
    if platform not in QUANTIZER_COLLECTION:
        raise KeyError(f'Target Platform {platform} is not supported by ppq right now.')
    if do_quantize:
        if calib_dataloader is None or calib_steps is None:
            raise TypeError('Quantization needs a valid calib_dataloader and calib_steps setting.')

    if setting is None:
        setting = QuantizationSettingFactory.default_setting()

    ppq_ir = load_onnx_graph(onnx_import_file=onnx_import_file, setting=setting)

    if inputs is None:
        dummy_input = torch.zeros(size=input_shape, device=device, dtype=input_dtype)
    else: dummy_input = inputs

    quantizer = QUANTIZER_COLLECTION[platform](graph=ppq_ir)

    assert isinstance(quantizer, BaseQuantizer)
    executor = TorchExecutor(graph=quantizer._graph, device=device)
    if do_quantize:
        quantizer.quantize(
            inputs=dummy_input,
            calib_dataloader=calib_dataloader,
            executor=executor,
            setting=setting,
            calib_steps=calib_steps,
            collate_fn=collate_fn
        )
        if verbose: quantizer.report()
        return quantizer._graph
    else:
        return quantizer._graph

@ empty_ppq_cache
def quantize_torch_model(
    model: torch.nn.Module,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    input_dtype: torch.dtype = torch.float,
    setting: QuantizationSetting = None,
    collate_fn: Callable = None,
    inputs: List[Any] = None,
    do_quantize: bool = True,
    platform: TargetPlatform = TargetPlatform.DSP_INT8,
    onnx_export_file: str = 'onnx.model',
    device: str = 'cuda',
    verbose: int = 0,
    ) -> BaseGraph:
    """
        量化一个 Pytorch 原生的模型
            输入一个 torch.nn.Module
            返回一个量化后的 PPQ.IR.BaseGraph
        
        quantize a pytorch model, input pytorch model and return quantized ppq IR graph
    Args:
        model (torch.nn.Module): 被量化的 torch 模型(torch.nn.Module) the pytorch model

        calib_dataloader (DataLoader): 校准数据集 calibration dataloader

        calib_steps (int): 校准步数 calibration steps

        collate_fn (Callable): 校准数据的预处理函数 batch collate func for preprocessing
        
        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                a list of ints indicating size of input, for multiple inputs, please use 
                                keyword arg inputs for direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                for direct parameter passing and this should be set to None

        setting (OptimSetting): 量化配置信息，用于配置量化的各项参数，设置为 None 时加载默认参数。
                                Quantization setting, default setting will be used when set None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                for multiple inputs, please give the specified inputs directly in the form of
                                a list of arrays

        do_quantize (Bool, optional): 是否执行量化 whether to quantize the model, defaults to True, defaults to True.

        platform (TargetPlatform, optional): 量化的目标平台 target backend platform, defaults to TargetPlatform.DSP_INT8.
                                        
        device (str, optional): 量化过程的执行设备 execution device, defaults to 'cuda'.

        verbose (int, optional): 是否打印详细信息 whether to print details, defaults to 0.

    Raises:
        ValueError: 给定平台不可量化 the given platform doesn't support quantization
        KeyError: 给定平台不被支持 the given platform is not supported yet

    Returns:
        BaseGraph: 量化后的IR，包含了后端量化所需的全部信息 
                   The quantized IR, containing all information needed for backend execution
    """
    # dump pytorch model to onnx
    dump_torch_to_onnx(model=model, onnx_export_file=onnx_export_file, 
        input_shape=input_shape, input_dtype=input_dtype, 
        inputs=inputs, device=device)

    return quantize_onnx_model(onnx_import_file=onnx_export_file, 
        calib_dataloader=calib_dataloader, calib_steps=calib_steps, collate_fn=collate_fn, 
        input_shape=input_shape, input_dtype=input_dtype, inputs=inputs, setting=setting, 
        platform=platform, device=device, verbose=verbose, do_quantize=do_quantize)

@ empty_ppq_cache
def quantize_caffe_model(
    caffe_proto_file: str,
    caffe_model_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    input_dtype: torch.dtype = torch.float,
    setting: QuantizationSetting = None,
    collate_fn: Callable = None,
    inputs: List[Any] = None,
    do_quantize: bool = True,
    platform: TargetPlatform = TargetPlatform.DSP_INT8,
    device: str = 'cuda',
    verbose: int = 0,
) -> BaseGraph:
    """
        量化一个 caffe 原生的模型
            输入一个 caffe 模型的文件路径和权重路径
            返回一个量化后的 PPQ.IR.BaseGraph
        quantize caffe model, input caffe prototxt and weight path, return a quantized ppq graph
    Args:
        caffe_proto_file (str): 被量化的 caffe 模型文件 .prototxt 路径
                                caffe prototxt location

        caffe_model_file (str): 被量化的 caffe 模型文件 .caffemodel 路径
                                caffe weight location

        calib_dataloader (DataLoader): 校准数据集 calibration data loader

        calib_steps (int): 校准步数 calibration steps

        collate_fn (Callable): 校准数据的预处理函数 batch collate func for preprocessing

        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                a list of ints indicating size of input, for multiple inputs, please use 
                                keyword arg inputs for direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                for direct parameter passing and this should be set to None

        setting (OptimSetting): 量化配置信息，用于配置量化的各项参数，设置为 None 时加载默认参数。
                                Quantization setting, default setting will be used when set None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                for multiple inputs, please give the specified inputs directly in the form of
                                a list of arrays

        do_quantize (Bool, optional): 是否执行量化 whether to quantize the model, defaults to True, defaults to True.

        platform (TargetPlatform, optional): 量化的目标平台 target backend platform, defaults to TargetPlatform.DSP_INT8.
                                        
        device (str, optional): 量化过程的执行设备 execution device, defaults to 'cuda'.

        verbose (int, optional): 是否打印详细信息 whether to print details, defaults to 0.

    Raises:
        ValueError: 给定平台不可量化 the given platform doesn't support quantization
        KeyError: 给定平台不被支持 the given platform is not supported yet

    Returns:
        BaseGraph: 量化后的IR，包含了后端量化所需的全部信息 
                   The quantized IR, containing all information needed for backend execution
    """
    if not TargetPlatform.is_quantized_platform(platform=platform):
        raise ValueError(f'Target Platform {platform} is an non-quantable platform.')
    if platform not in QUANTIZER_COLLECTION:
        raise KeyError(f'Target Platform {platform} is not supported by ppq right now.')
    if do_quantize:
        if calib_dataloader is None or calib_steps is None:
            raise TypeError('Quantization needs a valid calib_dataloader and calib_steps setting.')
    
    if setting is None:
        setting = QuantizationSettingFactory.default_setting()

    ppq_ir = load_graph(file_path=caffe_proto_file, 
                        caffemodel_path=caffe_model_file, 
                        from_framework=NetworkFramework.CAFFE)
    
    ppq_ir = format_graph(ppq_ir, setting=setting)

    if inputs is None:
        dummy_input = torch.zeros(size=input_shape, device=device, dtype=input_dtype)
    else: dummy_input = inputs

    quantizer = QUANTIZER_COLLECTION[platform](graph=ppq_ir)

    assert isinstance(quantizer, BaseQuantizer)
    executor = TorchExecutor(graph=quantizer._graph, device=device)
    if do_quantize:
        quantizer.quantize(
            inputs=dummy_input,
            calib_dataloader=calib_dataloader,
            executor=executor,
            setting=setting,
            calib_steps=calib_steps,
            collate_fn=collate_fn
        )
        if verbose: quantizer.report()
        return quantizer._graph
    else:
        return quantizer._graph

def export_ppq_graph(
    graph: BaseGraph, platform: TargetPlatform, 
    graph_save_to: str, config_save_to: str = None, **kwargs) -> None:
    """
    使用这个函数将 PPQ ir 保存到文件，同时导出 PPQ 的量化配置信息。
        该函数可以将 PPQ ir 保存为不同格式的模型文件。
    this func dumps ppq IR to file, and exports quantization setting information simultaneously

    详细的支持情况请参考：ppq.parser.__ini__.py
    for details please refer to ppq.parser.__ini__.py

    Args:
        graph (BaseGraph): 被保存的 ir 
                           the ppq IR graph

        platform (TargetPlatform): 期望部署的目标平台
                           target backend platform

        graph_save_to (str): 模型保存文件名
                           filename to save

        config_save_to (str): 量化配置信息保存文件名。
            注意部分平台导出时会将量化配置信息直接写入模型，在这种情况下设置此参数无效
            note that some of platforms requires to write quantization setting
            directly into the model file, this parameter won't have effect at
            this situation
    """
    for save_path in [graph_save_to, config_save_to]:
        if save_path is None: continue
        if os.path.exists(save_path):
            if os.path.isfile(save_path):
                ppq_warning(f'File {save_path} has already exist, ppq exporter will overwrite it.')
            if os.path.isdir(save_path):
                raise FileExistsError(f'File {save_path} has already exist, and it is a directory, '
                                    'ppq exporter can not create file here.')
    dump_graph_to_file(file_path=graph_save_to, config_path=config_save_to, 
                       target_platform=platform, graph=graph)

def format_graph(graph: BaseGraph, setting: QuantizationSetting) -> BaseGraph:
    """

    这个函数将对读入的计算图进行预处理 this func will preprocess the loaded computational graph
    
    所有的算子将被规范化，将符合 PPQ 的定义标准 all operators will be regularized 

    计算图将被切分并调度到不同设备 operators will be dispatched to different devices
    
    这不是一个可重入函数，如果你需要手动调用这个函数，则不能够使用 ppq.api.load_from_onnx 等函数加载模型！
    This is not an reenterable function, do not invoke this twice.
    if you are using functions like ppq.api.load_from_onnx, notice they will invoke this function automatically.

    """

    # do graph level optimization
    formatter = GraphDeviceSwitcher(GraphFormatter(GraphMerger(graph)))
    if str(setting.dispatcher).lower() not in DISPATCHER_TABLE:
        raise ValueError(f'Can not found dispatcher type "{setting.dispatcher}", check your input again.')
    dispatcher = DISPATCHER_TABLE[str(setting.dispatcher).lower()]

    formatter(GraphCommand(GraphCommandType.FORMAT_CONSTANT_INPUT))
    formatter(GraphCommand(GraphCommandType.FUSE_BN))
    formatter(GraphCommand(GraphCommandType.FORMAT_PARAMETERS))
    formatter(GraphCommand(GraphCommandType.FORMAT_CAST))
    formatter(GraphCommand(GraphCommandType.DELETE_ISOLATED))

    # dispatching.
    dispatching_table = dispatcher.dispatch(
        graph, quant_platform=TargetPlatform.UNSPECIFIED, 
        fp32_platform=TargetPlatform.FP32, 
        SOI_platform=TargetPlatform.SHAPE_OR_INDEX)

    # override dispatching result with setting
    dispatching_override = setting.dispatching_table
    for dispatching in dispatching_override.dispatchings:
        if dispatching.operation not in graph.operations: continue
        assert isinstance(dispatching.platform, int), (
            f'Your dispatching table contains a invalid setting of operation {dispatching.operation}, '
            'All platform setting given in dispatching table is expected given as int, '
            f'however {type(dispatching.platform)} was given.')
        dispatching_table[dispatching.operation] = TargetPlatform(dispatching.platform)
    
    for operation in graph.operations.values():
        assert operation.name in dispatching_table, (
            f'Internal Error, Can not find operation {operation.name} in dispatching table.')
        operation.platform = dispatching_table[operation.name]
    
    # insert necessary device switchers.
    formatter(GraphCommand(GraphCommandType.INSERT_SWITCHER))
    return graph

__all__ = ['load_onnx_graph', 'load_caffe_graph', 'dump_torch_to_onnx', 'quantize_onnx_model', 
           'quantize_torch_model', 'quantize_caffe_model', 'export_ppq_graph', 'format_graph']

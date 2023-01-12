from time import time

import numpy as np
import openvino.runtime as ov
from tqdm import tqdm

if ov.get_version() < '2022.1.0':
    raise Exception('Please Install Openvino >= 2022.1.0')

def Benchmark(ir_or_onnx_file: str, samples: int = 500, jobs: int = 4) -> float:
    """ Run Performance Benckmark with given onnx model. (Or Openvino IR)
    
    By default this function will run with Async Mode.
    """
    # https://docs.openvino.ai/latest/api/ie_python_api/_autosummary/openvino.runtime.InferRequest.html
    core = ov.Core()
    # core.add_extension("path_to_extension_library.so")
    model = core.read_model(ir_or_onnx_file)
    compiled_model = core.compile_model(model, 'CPU')

    infer_request = compiled_model.create_infer_request()
    print(f'Openvino Model Loaded: {len(infer_request.input_tensors)} Input Tensors, {len(infer_request.output_tensors)} Output Tensors')

    feed_dict = []
    for tensor in infer_request.input_tensors:
        feed_dict.append(np.random.random(size=tensor.shape).astype(tensor.element_type.to_dtype()))

    # Start async inference on a single infer request
    infer_request.start_async()
    # Wait for 1 milisecond
    infer_request.wait_for(1)
    # Wait for inference completion
    infer_request.wait()
    infer_queue = ov.AsyncInferQueue(compiled_model, jobs=jobs)

    tick = time()
    for _ in tqdm(range(samples)):
        # Wait for at least one available infer request and start asynchronous inference
        infer_queue.start_async(feed_dict)
    # Wait for all requests to complete
    infer_queue.wait_all()
    tok = time()
    
    print(f'Time span: {tok - tick  : .4f} sec')
    return tick - tok

    """
    infer_request.infer(feed_dict)
    for record in infer_request.get_profiling_info():
        print(record.node_name, record.node_type, record.cpu_time.total_seconds(), record.real_time.total_seconds())
    """
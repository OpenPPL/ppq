from ppq.utils.TensorRTUtil import Benchmark, Profiling

Benchmark(engine_file='Output/INT8.engine')
Benchmark(engine_file='Output/FP16.engine')
Benchmark(engine_file='Output/FP32.engine')

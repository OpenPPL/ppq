from ppq.utils.TensorRTUtil import Benchmark, Profiling

print('Profiling with Int8 Model')
Profiling(engine_file='Output/INT8.engine')
print('-------------------------------------------')

print('Profiling with Fp16 Model')
Profiling(engine_file='Output/FP16.engine')
print('-------------------------------------------')

print('Profiling with Fp32 Model')
Profiling(engine_file='Output/FP32.engine')
print('-------------------------------------------')
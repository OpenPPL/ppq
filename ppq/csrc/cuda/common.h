# include <cuda.h>
# include <cuda_runtime.h>
# include <math.h>
# include <torch/extension.h>
# pragma once

using at::Tensor;

# define CUDA_NUM_THREADS 512

# define ROUND_HALF_EVEN     0
# define ROUND_HALF_UP       1
# define ROUND_HALF_DOWN     2
# define ROUND_HALF_TOWARDS_ZERO    3
# define ROUND_HALF_FAR_FORM_ZERO   4
# define ROUND_TO_NEAR_INT   5
# define ROUND_UP            6
# define ROUND_DOWN          7

# define VALUE_CLIP(v, max, min) (v > max ? max: (v > min ? v : min))
# include "isotone.h"

namespace PPQ_Crt{

float SolveIsotoneScale(
    const float *first_largest_arr, 
    const float *second_largest_arr, 
    const int64 length, const int quant_max){
    /**
     * Solving isotonic quantization scale
     * Algorithm Complexity is 0(n^2), where n denotes the num of batches
     * This can be optimized further to reach O(n log n) complexity.
     */
    float *candidates = new float[length * 2];
    float *s_maxs = new float[length];
    float *s_mins = new float[length];

    for(int64 i = 0; i < length; i++){
        auto f = first_largest_arr[i];
        auto s = second_largest_arr[i];
        float s_min = s / (quant_max - 1);
        float s_max = 2 * (f - s);

        s_maxs[i] = s_max;
        s_mins[i] = s_min;

        candidates[2 * i] = s_min;
        candidates[2 * i + 1] = s_max;
    }

    int64 best_satisified = 0;
    float best_scale      = 1.0f;

    for(int64 i = 0; i < length * 2; i++){
        auto c = candidates[i]; int64 satisified = 0;
        for(int64 j = 0; j < length; j++){
            float s_max = s_maxs[j];
            float s_min = s_mins[j];
            satisified += (c <= s_max) + (c >= s_min);
        }
        if (satisified > best_satisified){
            best_satisified = satisified;
            best_scale = c;
        }
    }

    delete [] candidates;
    delete [] s_maxs;
    delete [] s_mins;
    return best_scale;
}

} // end of namespace
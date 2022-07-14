# include "hist_mse.h"

float compute_mse_loss(
    const vector<int64_t> &hist,
    const int start,
    const int step,
    const int end){
    int64_t num_of_elements = 0; float loss = 0.0;
    for(auto v: hist) num_of_elements += v;
    for(int idx = 0; idx < hist.size(); idx++){
        float error = 0.0f;
        int64_t bin = hist[idx];
        if(idx < start) error = start - idx - 1 + 0.5;
        else if(idx > end) error =idx - end + 0.5;
        else{
            int64_t l_idx = (idx - start) % step;
            int64_t r_idx = step - l_idx - 1;
            if(l_idx == r_idx) error = l_idx + 0.25;
            else{
                float l_err = (l_idx + 0.5);
                float r_err = (r_idx + 0.5);
                error = l_err < r_err ? l_err: r_err;
            }
        }
        loss += (bin * error * error) / num_of_elements;
    }
    return loss;
}
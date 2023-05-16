# include <vector>
# include <cstdlib>

using std::vector;
float compute_mse_loss(
    const vector<int64_t> &hist,
    const int start,
    const int step,
    const int end);

from typing import Tuple

import torch
from ppq.core import (CUDA, OBSERVER_KL_COMPUTING_DEVICE,
                      OBSERVER_KL_HIST_BINS,
                      OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE, OBSERVER_MIN_SCALE,
                      OBSERVER_MIN_SCALE_MANUL_OVERRIDE,
                      OBSERVER_MSE_COMPUTE_INTERVAL, OBSERVER_MSE_HIST_BINS,
                      OBSERVER_PERCENTILE, OBSERVER_PERCENTILE_MANUL_OVERRIDE,
                      OBSERVER_WARNING, PPQ_CONFIG,
                      ChannelwiseTensorQuantizationConfig,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TensorQuantizationConfig, convert_any_to_numpy,
                      ppq_quant_param_computing_function, ppq_warning)
from ppq.IR import Variable
from ppq.quantization.measure import torch_KL_divergence
from ppq.utils.round import ppq_numerical_round, ppq_round_to_power_of_2

from .base import BaseTensorObserver


@ ppq_quant_param_computing_function
def minmax_to_scale_offset(
    min_val: float, max_val: float,
    config: TensorQuantizationConfig,
    scale_threshold: float=OBSERVER_MIN_SCALE
) -> Tuple[float, float]:
    if OBSERVER_MIN_SCALE_MANUL_OVERRIDE in config.detail:
        scale_threshold = config.detail[OBSERVER_MIN_SCALE_MANUL_OVERRIDE]
    
    scale, offset = 1, 0
    if min_val > 0: min_val = 0
    if max_val < 0: max_val = 0

    if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
        range = float(max_val - min_val)
        scale  = range / (config.quant_max - config.quant_min)
        if scale < scale_threshold and OBSERVER_WARNING: 
            ppq_warning('Numeric instability detected: '
                        'ppq find there is a scale value < 1e-7, '
                        'which probably cause numeric underflow in further computation.')
        scale = max(scale, scale_threshold)
        offset = ppq_numerical_round(-min_val / scale)

    elif config.policy.has_property(QuantizationProperty.SYMMETRICAL):
        range = 2 * float(max(abs(max_val), abs(min_val)))
        scale  = range / (config.quant_max - config.quant_min)
        if scale < scale_threshold and OBSERVER_WARNING: 
            ppq_warning('Numeric instability detected: '
                        'ppq find there is a scale value < 1e-7, '
                        'which probably cause numeric underflow in further computation.')
        scale = max(scale, scale_threshold)
        offset = 0

    else:
        raise TypeError('Tensor Min Max Observer Excepts either ASYMMETRICAL or SYMMETRICAL quantization config.')
    if config.policy.has_property(QuantizationProperty.POWER_OF_2):
        scale = ppq_round_to_power_of_2(scale, policy=RoundingPolicy.ROUND_UP)
    return scale, offset


class TorchMinMaxObserver(BaseTensorObserver):
    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        self._min_val_collector = []
        self._max_val_collector = []

    @ torch.no_grad()
    def observe(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor), 'TorchMinMaxObserver can only deal with torch Tensor values'
        assert value.numel() > 0, (f'You are observing an empty tensor({self._watch_on.name}).')
        if self._quant_cfg.state == QuantizationStates.INITIAL:
            if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                self._min_val_collector.append(value.min().reshape(shape=[1, ]))
                self._max_val_collector.append(value.max().reshape(shape=[1, ]))
            elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                assert isinstance(self._quant_cfg, ChannelwiseTensorQuantizationConfig), \
                    'Your quantization config has PER_CHANNEL while it is not a '\
                    'ChannelwiseTensorQuantizationConfig instance.'
                channel_axis = self._quant_cfg.channel_axis
                channelwise_view = value.transpose(dim0=0, dim1=channel_axis).unsqueeze(-1)
                channelwise_view = torch.flatten(channelwise_view, start_dim=1)
                self._min_val_collector.append(torch.min(channelwise_view, dim=1, keepdim=True)[0])
                self._max_val_collector.append(torch.max(channelwise_view, dim=1, keepdim=True)[0])
            else:
                raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')

    def render_quantization_config(self):
        if len(self._max_val_collector) == 0:
            raise ValueError('Can not render quantization config yet, Observer data collator is empty. ' \
                'Invoke observe() function before render config.')
        device = self._max_val_collector[-1].device

        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            scale, offset = minmax_to_scale_offset(
                min_val = torch.min(torch.cat(self._min_val_collector, dim=0)).cpu().item(),
                max_val = torch.max(torch.cat(self._max_val_collector, dim=0)).cpu().item(),
                config=self._quant_cfg)
            self._quant_cfg.scale  = torch.tensor([scale], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([offset], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED

        elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            min_vals = torch.min(torch.cat(self._min_val_collector, dim=-1), dim=-1, keepdim=False)[0].cpu().numpy()
            max_vals = torch.max(torch.cat(self._max_val_collector, dim=-1), dim=-1, keepdim=False)[0].cpu().numpy()
            assert(len(min_vals) == len(max_vals)), 'Min values and max values should at same length.'
            scales, offsets = [], []
            for min_val, max_val in zip(min_vals, max_vals):
                scale, offset = minmax_to_scale_offset(
                    min_val=min_val, max_val=max_val, config=self._quant_cfg)
                scales.append(scale)
                offsets.append(offset)

            # scale, offset here only deployed on cpu
            # we will move them towards target device through RunnableGraph
            self._quant_cfg.scale  = torch.tensor(scales, dtype=torch.float32, device=device)
            self._quant_cfg.offset = torch.tensor(offsets, dtype=torch.float32, device=device)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        else:
            raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')


class TorchHistObserver(TorchMinMaxObserver):
    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig,
                 hist_bins: int = OBSERVER_KL_HIST_BINS):
        self._phase = 'Detecting Minmax'
        self._hist  = None
        self._hist_scale = None
        self._min   = None
        self._max   = None
        if OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE in quant_cfg.detail:
            hist_bins = quant_cfg.detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE]
        self._hist_bins  = hist_bins
        super().__init__(watch_on, quant_cfg)

    def observe(self, value: torch.Tensor):
        assert value.numel() > 0, (f'You are observing an empty tensor({self._watch_on.name}).')

        if self._phase == 'Detecting Minmax':
            return super().observe(value) # collect min, max

        elif self._phase == 'Collating Hist':
            if self._hist is None:
                self._hist = torch.zeros(size=(self._hist_bins,), dtype=torch.int32, device=value.device)

            if self._quant_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                # ASYMMETRICAL Hist
                if PPQ_CONFIG.USING_CUDA_KERNEL and value.is_cuda:
                    CUDA.Histogram_Asymmetric_T(self._min, self._max, tensor=value, histogram=self._hist)
                else:
                    hist = torch.histc(value, self._hist_bins, min=self._min, max=self._max)
                    self._hist += hist.int()

            elif self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRICAL):
                # SYMMETRICAL Hist
                if PPQ_CONFIG.USING_CUDA_KERNEL and value.is_cuda:
                    CUDA.Histogram_T(tensor=value, histogram=self._hist, scale=self._hist_scale)
                else:
                    hist = torch.histc(torch.abs(value), self._hist_bins, min=0, max=self._hist_scale * self._hist_bins)
                    self._hist += hist.int()

            else:
                raise TypeError('Quantization Property is invalid, '
                                'expect either ASYMMETRICAL or SYMMETRICAL config here.')

    @ ppq_quant_param_computing_function
    def hist_to_scale_offset(
        self, histogram: torch.Tensor, hist_bins: int, hist_scale: float,
        config: TensorQuantizationConfig, computing_device: str = OBSERVER_KL_COMPUTING_DEVICE,
        scale_threshold: float=OBSERVER_MIN_SCALE
    ) -> Tuple[float, int]:
        """
        PPQ core quant parameter computing method - Histogram to scale & offset

        With a pre-defined histogram,
        this function will automatically search best clip value
        to minimize KL divergence between quantized result and fp32 input.

        only work for per-tensor symmetrical quantization policy for now.
        see also https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

        Args:
            histogram (torch.Tensor): histogram records activation's statistics.
            hist_bins (int): how many bins are included in histogram(also known as histogram length)
            hist_scale (float): histogram step size. it can be solved by histogram.max_val / histogram.bins
            config (TensorQuantizationConfig): quantization config.
            computing_device (str, optional): computing device. Defaults to 'cpu'.

        Raises:
            ValueError: given quantization config is invalid.

        Returns:
            Tuple[float, int]: scale(fp32) and offset(int).
        """
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            raise PermissionError('KL observer is not designed for ASYMMETRICAL quantization')
        
        if OBSERVER_MIN_SCALE_MANUL_OVERRIDE in config.detail:
            scale_threshold = config.detail[OBSERVER_MIN_SCALE_MANUL_OVERRIDE]

        # move histogram to cpu, speedup computation.
        histogram = histogram.to(computing_device).float()

        # compute symmtrical kl-divergence.
        # Here is a simple example: reference distribution P consisting of 8 bins, we want to quantize into 2 bins:
        # P = [ 1, 0, 2, 3, 5, 3, 1, 7]
        # we merge into 2 bins (8 / 2 = 4 consecutive bins are merged into one bin)
        # [1 + 0 + 2 + 3 , 5 + 3 + 1 + 7] = [6, 16]
        # then proportionally expand back to 8 bins, we preserve empty bins from the original distribution P:
        # Q = [ 6/3, 0, 6/3, 6/3, 16/4, 16/4, 16/4, 16/4] = [ 2, 0, 2, 2, 4, 4, 4, 4]
        # now we should normalize both distributions, after that we can compute KL_divergence
        # P /= sum(P) Q /= sum(Q)
        # result = KL_divergence(P, Q)
        # see also
        # https://github.com/NVIDIA/TensorRT/blob/3835424af081db4dc8cfa3ff3c9f4a8b89844421/tools/pytorch-quantization/pytorch_quantization/calib/histogram.py#L147

        losses, quant_bins = [], 2 ** (config.num_of_bits - 1)

        # following code is curcial, do not move
        histogram[: int(hist_bins * .002)] = 0
        histogram[int(hist_bins * .002)] = 1

        hist_sum = torch.sum(histogram)
        for bin_range in range(quant_bins, hist_bins + quant_bins - 1, quant_bins):
            p_hist = torch.zeros(size=(bin_range, ), dtype=torch.float, device=computing_device)
            p_hist[: bin_range].copy_(histogram[: bin_range])
            p_hist[bin_range - 1] += torch.sum(histogram[bin_range: ])
            p_hist = p_hist / hist_sum

            expand_ratio = int(bin_range / quant_bins)
            q_hist = histogram[: bin_range].clone()
            q_hist = q_hist.reshape((quant_bins, expand_ratio))
            positive_map = q_hist > 0
            positive_cnt = positive_map.sum(axis=1, keepdim=True)
            positive_cnt[positive_cnt == 0] = 1
            q_hist = torch.div(q_hist.sum(axis=1, keepdim=True), positive_cnt)
            q_hist = q_hist.repeat([1, expand_ratio])
            q_hist = q_hist * positive_map
            q_hist = q_hist / torch.sum(q_hist)
            q_hist = q_hist.flatten()

            losses.append({
                'kl': torch_KL_divergence(p_hist, q_hist),
                'bin_range': bin_range
            })

        best_bin_range = sorted(losses, key=lambda x: x['kl'])[0]['bin_range']
        scale, offset = (best_bin_range / self._hist_bins) * hist_scale * (self._hist_bins / quant_bins), 0
        
        if scale < scale_threshold and OBSERVER_WARNING: 
            ppq_warning('Numeric instability detected: '
                        'ppq find there is a scale value < 1e-7, '
                        'which probably cause numeric underflow in further computation.')
        scale = max(scale, scale_threshold)

        if config.policy.has_property(QuantizationProperty.POWER_OF_2):
            scale = ppq_round_to_power_of_2(scale, policy=RoundingPolicy.ROUND_HALF_UP)
        return scale, offset

    def render_quantization_config(self):
        if not self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            raise ValueError('Hist observer can only apply with per-tensor quantization config.')

        if self._phase == 'Detecting Minmax':
            min_val = torch.min(torch.cat(self._min_val_collector, dim=0)).cpu().item()
            max_val = torch.max(torch.cat(self._max_val_collector, dim=0)).cpu().item()
            if self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRICAL):
                hist_range = float(max(abs(max_val), abs(min_val)))
            else:
                hist_range = max_val - min_val
            self._min = min_val
            self._max = max_val
            self._hist_scale = hist_range / self._hist_bins
            self._phase = 'Collating Hist'
        elif self._phase == 'Collating Hist':
            scale, offset = self.hist_to_scale_offset(
                histogram=self._hist, hist_bins=self._hist_bins,
                hist_scale=self._hist_scale, config=self._quant_cfg)
            device = self._hist.device
            self._quant_cfg.scale  = torch.tensor([scale], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([offset], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED


class TorchPercentileObserver(BaseTensorObserver):
    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        if not OBSERVER_PERCENTILE_MANUL_OVERRIDE in quant_cfg.detail:
            self._percentile = OBSERVER_PERCENTILE
        else: self._percentile = quant_cfg.detail[OBSERVER_PERCENTILE_MANUL_OVERRIDE]
        self._percentile_collector = []
        self._percentile_maxs = []
        self._percentile_mins = []

    @ torch.no_grad()
    def observe(self, value: torch.Tensor):
        assert value.numel() > 0, (f'You are observing an empty tensor({self._watch_on.name}).')
        assert isinstance(value, torch.Tensor), 'TorchMinMaxObserver can only deal with torch Tensor values'
        if self._quant_cfg.state == QuantizationStates.INITIAL:
            if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                if not PPQ_CONFIG.USING_CUDA_KERNEL or (not value.is_cuda):
                    numel = value.numel()
                    
                    min_idx, max_idx = int(numel * (1 - self._percentile)), int(numel * (self._percentile))
                    # torch.kthvalue needs index from 1 to numel ...
                    min_idx = max(0, min_idx) + 1
                    max_idx = min(max_idx, numel - 1) + 1
                    _min = torch.kthvalue(value.flatten(), k = min_idx, dim=0)[0].view(1, -1)
                    _max = torch.kthvalue(value.flatten(), k = max_idx, dim=0)[0].view(1, -1)
                    self._percentile_collector.append(torch.cat([_max, _min], dim=-1))
                else:
                    self._percentile_collector.append(CUDA.Quantile(value, self._percentile).view(1, -1))
            elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                raise PermissionError('Percentile observer can not deal with per channel quantization.')
                assert isinstance(self._quant_cfg, ChannelwiseTensorQuantizationConfig), (
                    'Your quantization config has PER_CHANNEL while it is not a '
                    'ChannelwiseTensorQuantizationConfig instance.')
                channel_axis = self._quant_cfg.channel_axis
                channelwise_view = value.transpose(dim0=0, dim1=channel_axis)
                channelwise_view = torch.flatten(channelwise_view, start_dim=1)
                self._percentile_mins.append(-torch.quantile(-channelwise_view, q=self._percentile, dim=1, keepdim=True)[0])
                self._percentile_maxs.append(torch.quantile(channelwise_view, q=self._percentile, dim=1, keepdim=True)[0])
            else:
                raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')

    def render_quantization_config(self):
        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            if len(self._percentile_collector) == 0:
                raise ValueError('Can not render quantization config yet, Observer data collator is empty. ' \
                    'Invoke observe() function before render config.')
            device = self._percentile_collector[-1].device
            self._percentile_collector = torch.cat(self._percentile_collector, dim=0).mean(dim=0).cpu()
            scale, offset = minmax_to_scale_offset(
                min_val = self._percentile_collector[1].item(),
                max_val = self._percentile_collector[0].item(),
                config=self._quant_cfg)

            self._quant_cfg.scale  = torch.tensor([scale], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([offset], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError('Percentile observer can not deal with per channel quantization.')
            if len(self._percentile_maxs) == 0:
                raise ValueError('Can not render quantization config yet, Observer data collator is empty. ' \
                    'Invoke observe() function before render config.')
            device = self._percentile_maxs[-1].device

            min_vals = torch.mean(torch.cat(self._percentile_mins, dim=-1), dim=-1, keepdim=False)
            max_vals = torch.mean(torch.cat(self._percentile_maxs, dim=-1), dim=-1, keepdim=False)

            min_vals = min_vals.cpu()
            max_vals = max_vals.cpu()

            assert(len(min_vals) == len(max_vals)), 'Min values and max values should at same length.'
            scales, offsets = [], []
            for min_val, max_val in zip(min_vals, max_vals):
                scale, offset = minmax_to_scale_offset(
                    min_val=min_val, max_val=max_val, config=self._quant_cfg)
                scales.append(scale)
                offsets.append(offset)

            self._quant_cfg.scale  = torch.tensor(scales, dtype=torch.float32, device=device)
            self._quant_cfg.offset = torch.tensor(offsets, dtype=torch.int32, device=device)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        else:
            raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')


class TorchMSEObserver(TorchHistObserver):
    """Histogram accelerated MSE Observer, inspired by LightGBM This observer
    will collect data in histogram firstly, all mse computing will directly use
    histogram rather than data itself.

        Time complexity: O(Iteration * Num_of_Batch * Length(Data)) -> O(Iteration * Length(histogram))
        Space complexity: O(Num_of_Batch * Length(Data)) -> O(Iteration * Length(histogram))

    Args:
        TorchHistObserver ([type]): [description]
    """
    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig,
                 bins: int = OBSERVER_MSE_HIST_BINS):
        super().__init__(watch_on, quant_cfg)
        self._hist_bins = bins

    def compute_mse_loss(self, histogram: list, start: int, step: int, end: int):
        if PPQ_CONFIG.USING_CUDA_KERNEL:
            from ppq.core import CUDA
            return CUDA.compute_mse_loss(histogram=histogram, start=start, step=step, end=end)
        else:
            # 如果你觉得 mse 太慢，想办法加速这段代码就可以了
            # 求解 mse 时，我们假设每一个 bin 里面的数据都是均匀分布的
            # 我们需要给一个直方图，并用 start, end, step 给出量化表示的范围
            # losses = [0 for _ in histogram]  debug
            num_of_elements = sum(histogram)
            loss = 0
            for idx, bin in enumerate(histogram):
                if idx < start:
                    # 如果所选的 bin 已经超出了起点，那从 bin 的中心到起点的距离即
                    # ((idx 到 起点的距离) + 0.5)
                    # 注意 hist 统计时是直接取 floor 的，因此会在这里额外 - 1
                    error = ((start - idx - 1) + 0.5)
                elif idx > end:
                    # 注意 hist 统计时是直接取 floor 的
                    error = ((idx - end) + 0.5)
                else:
                    # 分别计算左右两侧的 err
                    l_idx = (idx - start) % step
                    r_idx = step - l_idx - 1
                    if l_idx == r_idx:
                        error = (l_idx + 0.25)
                    else:
                        l_err = (l_idx + 0.5)
                        r_err = (r_idx + 0.5)
                        error = min(l_err, r_err)
                loss += (bin * error * error) / num_of_elements
                # losses[idx] = bin * error * error
            return loss

    @ ppq_quant_param_computing_function
    def hist_to_scale_offset(
        self, histogram: torch.Tensor,
        hist_bins: int, hist_scale: float,
        config: TensorQuantizationConfig,
        scale_threshold: float = OBSERVER_MIN_SCALE
    ) -> Tuple[float, int]:
        from tqdm import tqdm
        if OBSERVER_MIN_SCALE_MANUL_OVERRIDE in config.detail:
            scale_threshold = config.detail[OBSERVER_MIN_SCALE_MANUL_OVERRIDE]
        histogram = convert_any_to_numpy(histogram).tolist()
        num_of_quant_levels = (self._quant_cfg.quant_max - self._quant_cfg.quant_min) + 1

        losses = []
        if (config.policy.has_property(QuantizationProperty.ASYMMETRICAL) and
            config.policy.has_property(QuantizationProperty.PER_TENSOR)):
            
            # at least we can have a min-max result
            step = hist_bins // num_of_quant_levels + 1
            loss = self.compute_mse_loss(histogram=histogram, start=0, step=step, end=num_of_quant_levels * step)
            losses.append({'mse': loss, 'start': 0, 'end': num_of_quant_levels * step})

            for start in range(0, hist_bins, OBSERVER_MSE_COMPUTE_INTERVAL):
                if (start * hist_scale) + self._min > 0:
                    break # start can not > 0

                for step in range(1, hist_bins // num_of_quant_levels + 1):
                    end = start + num_of_quant_levels * step
                    if end > (hist_bins + num_of_quant_levels): break
                    loss = self.compute_mse_loss(histogram=histogram, start=start, step=step, end=end)
                    losses.append({'mse': loss, 'start': start, 'end': end})
        
            best_policy = sorted(losses, key=lambda x: x['mse'])[0]
            best_start  = best_policy['start']
            best_end    = best_policy['end']

            # translate start & end to scale & offset.
            range_min, range_max = (best_start * hist_scale) + self._min, (best_end * hist_scale) + self._min
            scale, offset = minmax_to_scale_offset(range_min, range_max, config, scale_threshold)
            return scale, offset

        elif config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError('Torch Mse observer do not support PER_CHANNEL policy now, please wait.')

        elif (config.policy.has_property(QuantizationProperty.SYMMETRICAL) and
              config.policy.has_property(QuantizationProperty.PER_TENSOR)):
            # at least we can have a min-max result
            step = hist_bins // num_of_quant_levels + 1
            loss = self.compute_mse_loss(histogram=histogram, start=0, step=step, end=num_of_quant_levels * step)
            losses.append({'mse': loss, 'end': num_of_quant_levels * step})

            for step in range(1, hist_bins // num_of_quant_levels + 1):
                end = num_of_quant_levels * step
                if end > (hist_bins + num_of_quant_levels): break
                loss = self.compute_mse_loss(histogram=histogram, start=0, step=step, end=end)
                losses.append({'mse': loss, 'end': end})

            best_policy = sorted(losses, key=lambda x: x['mse'])[0]
            best_end    = best_policy['end']

            # translate start & end to scale & offset.
            range_min, range_max = -(best_end * hist_scale), (best_end * hist_scale)
            scale, offset = minmax_to_scale_offset(range_min, range_max, config, scale_threshold)
            return scale, offset

        raise Exception('Oops, there might be some mistakes.')

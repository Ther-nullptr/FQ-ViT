# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from .bit_type import BIT_TYPE_DICT
from .observer import build_observer
from .quantizer import build_quantizer

import copy


def softmax_func(x, base, dim=-1):
    max_val, _ = torch.max(x, dim, keepdims = True)
    return torch.pow(base, x - max_val) / torch.pow(base, x - max_val).sum(dim, keepdim = True)

class QConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'conv_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        weight = self.quantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight)
        return F.linear(x, weight, self.bias)


class QAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return x
        x = self.quantizer(x)
        return x


class QIntLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(normalized_shape, eps,
                                            elementwise_affine)
        assert isinstance(normalized_shape, int)
        self.mode = 'ln'

    def get_MN(self, x):
        bit = 7
        N = torch.clamp(bit - torch.floor(torch.log2(x)), 0, 31)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2**(bit + 1) - 1)
        return M, N

    def forward(self,
                x,
                in_quantizer=None,
                out_quantizer=None,
                in_scale_expand=1):
        if self.mode == 'ln':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.mode == 'int':
            in_scale = in_quantizer.scale
            if in_scale_expand != 1:
                in_scale = in_scale.unsqueeze(-1).expand(
                    -1, in_scale_expand).T.reshape(-1)
            out_scale = out_quantizer.scale
            assert in_scale is not None and out_scale is not None
            channel_nums = x.shape[-1]
            in_scale = in_scale.reshape(1, 1, -1)
            out_scale = out_scale.reshape(1, 1, -1)
            x_q = (x / in_scale).round()
            in_scale1 = in_scale.min()
            in_scale_mask = (in_scale / in_scale1).round()

            x_q = x_q * in_scale_mask

            mean_x_q = x_q.mean(dim=-1) * in_scale1
            std_x_q = (in_scale1 / channel_nums) * torch.sqrt(
                channel_nums * (x_q**2).sum(dim=-1) - x_q.sum(dim=-1)**2) + self.eps

            A = (in_scale1 / std_x_q).unsqueeze(-1) * \
                self.weight.reshape(1, 1, -1) / out_scale
            A_sign = A.sign()
            M, N = self.get_MN(A.abs())
            B = ((self.bias.reshape(1, 1, -1) -
                  (mean_x_q / std_x_q).unsqueeze(-1) *
                  self.weight.reshape(1, 1, -1)) / out_scale *
                 torch.pow(2, N)).round()

            x_q = ((A_sign * M * x_q + B) / torch.pow(2, N)).round()
            x = x_q * out_scale
        else:
            raise NotImplementedError
        return x


class QIntSoftmax(nn.Module):

    def __init__(self,
                 log_int_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QIntSoftmax, self).__init__()

        self.log_int_softmax = log_int_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor

        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale):
        if self.log_int_softmax and scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = torch.round(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2**self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            x = x.softmax(dim=-1)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x


class QIntSoftermax(nn.Module):

    def __init__(self,
                 log_int_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform',
                 input_int_bit=6,
                 input_frac_bit=2,
                 local_max_int_bit=6,
                 local_max_frac_bit=2,
                 unnormed_int_bit=1,
                 unnormed_frac_bit=15,
                 pow_sum_int_bit=10,
                 pow_sum_frac_bit=6,
                 recip_int_bit=1,
                 recip_frac_bit=7,
                 output_int_bit=1,
                 output_frac_bit=15,
                 base=2):
        super(QIntSoftermax, self).__init__()
        self.log_int_softmax = log_int_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)
        self.input_int_bit = input_int_bit
        self.input_frac_bit = input_frac_bit        
        self.local_max_int_bit = local_max_int_bit
        self.local_max_frac_bit = local_max_frac_bit
        self.unnormed_int_bit = unnormed_int_bit
        self.unnormed_frac_bit = unnormed_frac_bit
        self.pow_sum_int_bit = pow_sum_int_bit
        self.pow_sum_frac_bit = pow_sum_frac_bit
        self.recip_int_bit = recip_int_bit
        self.recip_frac_bit = recip_frac_bit
        self.output_int_bit = output_int_bit
        self.output_frac_bit = output_frac_bit
        self.base = base
    
    def history_max(self, x):
        x = torch.round(x.clone())
        x[..., 0] = torch.ceil(x[..., 0])
        for i in range(1, x.shape[-1]):
            x[..., i] = torch.ceil(torch.max(x[..., i], x[..., i - 1]))
        return x

    def convert(self, x, int_bit=6, frac_bit=2):
        return torch.round(torch.clamp(x * 2 ** frac_bit, max=2 ** (int_bit + frac_bit))) / 2 ** frac_bit

    def get_sum_and_power(self, x, m, base=2):
        m_diff = torch.diff(m, dim=-1, prepend=torch.zeros_like(m[..., :1]))
        x_m_diff = x - m
        pow_x_m_diff = torch.pow(base, x_m_diff)
        d = copy.deepcopy(pow_x_m_diff[..., 0])
        for i in range(1, x.shape[-1]):
            d *= torch.pow(2, -m_diff[..., i])
            d += pow_x_m_diff[..., i]
        return d, pow_x_m_diff

    def forward(self, x, scale):
        if self.log_int_softmax and scale is not None:
            # Inp.
            x = self.convert(x, self.input_int_bit, self.input_frac_bit)
            m = self.history_max(torch.ceil(x))
            m_v = m[..., -1]
            m_v_i_diff = m_v.unsqueeze(-1) - m
            d, pow_x_m_diff = self.get_sum_and_power(x, m, base = self.base)
            # PowSum
            d = self.convert(d, self.pow_sum_int_bit, self.pow_sum_frac_bit)
            # Recip.
            recip_d = self.convert(1 / d, self.recip_int_bit, self.recip_frac_bit)
            # Unnormed
            pow_x_m_diff = self.convert(pow_x_m_diff, self.unnormed_int_bit, self.unnormed_frac_bit)
            out = pow_x_m_diff * torch.pow(self.base, -m_v_i_diff) * recip_d.unsqueeze(-1)
            # Out.
            out = self.convert(out, self.output_int_bit, self.output_frac_bit)
            return out
        else:
            x = softmax_func(x, self.base)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x

class QIntSoftmaxUniform(nn.Module):
    def __init__(self,
                 log_int_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QIntSoftmaxUniform, self).__init__()

        self.log_int_softmax = log_int_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):
        M = 25

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor

        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        x_out = torch.round(exp_int * (2 ** M) / exp_int_sum)
        x_out = x_out / (2 ** M)
        return x_out, exp_scaling_factor

    def forward(self, x, scale):
        if self.log_int_softmax and scale is not None:
            x, scale = self.int_softmax(x, scale)
            return x
        else:
            x = x.softmax(dim=-1)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x


class QIntGELU(nn.Module):
    def __init__(self,
                i_gelu=False,
                quant=False,
                calibrate=False,
                last_calibrate=False,
                bit_type=BIT_TYPE_DICT['int8'],
                calibration_mode='layer_wise',
                observer_str='minmax',
                quantizer_str='uniform'):
        super(QIntGELU, self).__init__()

        self.i_gelu = i_gelu
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                        self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                            self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_gelu(x, scaling_factor):
        
        coef = [-0.2888, -1.769, 1.] # a(x+b)**2 + c
        sqrt2 = 1.41421356

        def int_polynomial(x_int, scaling_factor):
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / (coef[0] * scaling_factor**2))
            z = (x_int + b_int) ** 2 + c_int
            scaling_factor = coef[0] * scaling_factor ** 2
            return z, scaling_factor

        def int_erf(x_int, scaling_factor):
            x_int_sgn = torch.sign(x_int)
            x_int = torch.clip(torch.abs(x_int), max = -coef[1] / scaling_factor)
            x_int, scaling_factor = int_polynomial(x_int, scaling_factor)
            x_int = x_int_sgn * x_int
            return x_int, scaling_factor

        x_int = x / scaling_factor
        x_erf, scaling_factor_erf = int_erf(x_int, scaling_factor / sqrt2)
        x_int_1 = torch.floor(1. / scaling_factor_erf)
        x_int = x_int * (x_erf + x_int_1)
        scaling_factor = scaling_factor_erf * scaling_factor / 2
        return x_int, scaling_factor

    def forward(self, x, scale):
        if self.i_gelu and scale is not None:
            x_int, scale = self.int_gelu(x, scale)
            return x_int * scale
        else:
            x = F.gelu(x)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x


class QIntSoftmaxShift(nn.Module):
    def __init__(self,
                log_int_softmax=False,
                quant=False,
                calibrate=False,
                last_calibrate=False,
                bit_type=BIT_TYPE_DICT['int8'],
                calibration_mode='layer_wise',
                observer_str='minmax',
                quantizer_str='uniform'):
        super(QIntSoftmaxShift, self).__init__()

        self.log_int_softmax = log_int_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                        self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                            self.observer, self.module_type)
        self.bit = self.bit_type.bits
        

    @staticmethod
    def shiftmax(x, scaling_factor, bit):
        M = 30
        N = 20

        def intdiv(x_int, y_int):
            x_out = torch.floor((2 ** M) /  y_int) * x_int * 2 ** (-M + (bit - 1))
            s_out = 2 ** (-(bit - 1))
            return x_out, s_out

        def shiftexp(x_int, scaling_factor):
            Ip = x_int + x_int * 0.5 - x_int * 0.0625
            I0 = torch.round(1 / scaling_factor)
            q = torch.floor(Ip / -I0)
            r = -(Ip + q * I0)
            Ib = ((-r) * 0.5) + I0
            Iexp = Ib * 2 ** (N - q)
            Sexp = scaling_factor * (2 ** (-N))
            return Iexp, Sexp           

        I = x / scaling_factor
        I_delta = I - torch.max(I)
        Iexp, Sexp = shiftexp(I_delta, scaling_factor)
        Iout, Sout = intdiv(Iexp, torch.sum(Iexp))
        return Iout, Sout

    def forward(self, x, scale):
        if self.log_int_softmax and scale is not None:
            x_int, scale = self.shiftmax(x, scale, self.bit)
            return x_int * scale
        else:
            x = F.softmax(x, dim=-1)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x


class QIntGELUShift(nn.Module):
    def __init__(self,
            i_gelu=False,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='layer_wise',
            observer_str='minmax',
            quantizer_str='uniform'):
        super(QIntGELUShift, self).__init__()

        self.i_gelu = i_gelu
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                        self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                            self.observer, self.module_type)

        self.bit = self.bit_type.bits

    @staticmethod
    def shiftgelu(x, scaling_factor, bit):
        M = 30
        N = 20

        def intdiv(x_int, y_int):
            x_out = torch.floor((2 ** M) /  y_int) * x_int * 2 ** (-M + (bit - 1))
            s_out = 2 ** (-(bit - 1))
            return x_out, s_out

        def shiftexp(x_int, scaling_factor):
            Ip = x_int + x_int * 0.5 - x_int * 0.0625
            I0 = torch.round(1 / scaling_factor)
            q = torch.floor(Ip / -I0)
            r = -(Ip + q * I0)
            Ib = ((-r) * 0.5) + I0
            Iexp = Ib * 2 ** (N - q)
            Sexp = scaling_factor * (2 ** - N)
            return Iexp, Sexp

        I = x / scaling_factor
        S = scaling_factor
        Ip = I + I * 0.5 + I * 0.125 + I * 0.0625
        I_delta = Ip - torch.max(Ip)
        Iexp, Sexp = shiftexp(I_delta, S)
        Iexp_1, Sexp_1 = shiftexp(-torch.max(Ip), S)
        Idiv, Sdiv = intdiv(Iexp, Iexp + Iexp_1)
        Iout, Sout = Idiv * I, S * Sdiv
        return Iout, Sout

    def forward(self, x, scale):
        if self.i_gelu and scale is not None:
            x_int, scale = self.shiftgelu(x, scale, self.bit)
            return x_int * scale
        else:
            x = F.gelu(x)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x
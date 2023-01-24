import torch
import copy

def convert(x, int_bit = 6, frac_bit = 2):
    return torch.round(torch.clamp(x * 2 ** frac_bit, max = 2 ** (int_bit + frac_bit))) / 2 ** frac_bit

def softermax_func(x, dim=-1):
    max_val, _ = torch.max(x, dim, keepdims = True)
    return torch.pow(2, x - max_val) / torch.pow(2, x - max_val).sum(dim, keepdim = True)

def history_max(x):
    x = x.clone()
    x[..., 0] = torch.ceil(x[..., 0])
    for i in range(1, x.shape[-1]):
        x[..., i] = torch.ceil(torch.max(x[..., i], x[..., i-1]))
    return x

def sum_d(x, m, base = 2):
    m_diff = torch.diff(m, dim=-1, prepend=torch.zeros_like(m[..., :1]))
    x_m_diff = x - m
    # print(x_m_diff)
    pow_x_m_diff = torch.pow(base, x_m_diff)
    # print(pow_x_m_diff)
    d = copy.deepcopy(pow_x_m_diff[..., 0])
    for i in range(1, x.shape[-1]):
        d *= torch.pow(2, -m_diff[..., i])
        d += pow_x_m_diff[..., i]
    return d, pow_x_m_diff

if __name__ == '__main__':
    #ipdb.set_trace()
    I = torch.randint(0, 100, (1, 2, 10)).float()
    s = 0.034
    I = convert(I * s)
    m = history_max(torch.ceil(I))
    m_v = m[..., -1]
    m_v_i_diff = m_v.unsqueeze(-1) - m
    
    d, pow_x_m_diff = sum_d(I, m, base = 2)
    # (32, 0) -> (10, 6)
    d = torch.round(torch.clamp(d * (2 ** 6), max = 2 ** 16)) / 2 ** 6
    # recip_d 
    recip_d = torch.round(torch.clamp((1 / d) * (2 ** 7), max = 2 ** 8)) / 2 ** 7
    # (32, 0) -> (1, 15)
    pow_x_m_diff = torch.round(torch.clamp(pow_x_m_diff * (2 ** 15), max = 2 ** 16)) / 2 ** 15
    # print(pow_x_m_diff, recip_d)
    out = torch.round(torch.clamp(pow_x_m_diff * torch.pow(2, -m_v_i_diff) * recip_d.unsqueeze(-1) * (2 ** 7), max = 2 ** 8)) / 2 ** 7
    print(out)
    print(softermax_func(I, dim=-1))
    
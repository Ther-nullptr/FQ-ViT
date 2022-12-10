import torch
import torch.nn.functional as F
import sys

sys.path.append('..')
from models.ptq.layers import QIntGELU, QIntGELUShift, QIntSoftmaxUniform, QIntSoftmaxShift

def softermax(x):
    m = torch.tensor(float('-inf'))
    d = 0
    for item in x:
        m_new = torch.ceil(m) if torch.ceil(m) > torch.ceil(item) else torch.ceil(item)
        d = d * torch.exp2(m - m_new) + torch.exp2(item - m_new)
        m = m_new
    return torch.exp2(x - m) / d

if __name__ == '__main__':
    qint_gelu = QIntGELU(i_gelu=True)
    qint_gelu_shift = QIntGELUShift(i_gelu=True)
    qint_softmax = QIntSoftmaxUniform(log_int_softmax=True)
    qint_softmax_shift = QIntSoftmaxShift(log_int_softmax=True)

    # x = torch.tensor([-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    # S = torch.tensor(0.1)
    # qint_gelu_out = qint_gelu.forward(x, S)
    # qint_gelu_shift_out = qint_gelu_shift.forward(x, S)
    # qint_softmax_out = qint_softmax.forward(x, S)
    # qint_softmax_shift_out = qint_softmax_shift.forward(x, S)
    # print(f'qint_gelu_out: {qint_gelu_out}')
    # print(f'qint_gelu_shift_out: {qint_gelu_shift_out}')
    # print(f'true gelu: {torch.nn.functional.gelu(x)}')
    # print(f'qint_softmax_out: {qint_softmax_out}')
    # print(f'qint_softmax_shift_out: {qint_softmax_shift_out}')
    # print(f'true softmax: {torch.nn.functional.softmax(x, dim=-1)}')

    # plot the gelu
    import matplotlib.pyplot as plt
    import numpy as np
    x = torch.tensor(np.linspace(-5, 1, 100))
    S = torch.tensor(0.1)
    true_gelu = F.gelu(x)
    poly_gelu = qint_gelu.forward(x, S)
    shift_gelu = qint_gelu_shift.forward(x, S)
    S_2 = torch.tensor(0.4)
    poly_gelu_2 = qint_gelu.forward(x, S_2)
    shift_gelu_2 = qint_gelu_shift.forward(x, S_2)
    plt.plot(x, true_gelu, label='true_gelu')
    plt.plot(x, poly_gelu, label='poly_gelu, S = 0.1')
    plt.plot(x, shift_gelu, label='shift_gelu, S = 0.1')
    plt.plot(x, poly_gelu_2, label='poly_gelu, S = 0.4')
    plt.plot(x, shift_gelu_2, label='shift_gelu, S = 0.4')
    plt.legend()
    plt.savefig('gelu.png')

    # plot the softmax
    # case 1: x is uniform
    label = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    x = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 3.5])
    S = torch.tensor(0.1)
    true_softmax = F.softmax(x, dim=-1)
    poly_softmax = qint_softmax.forward(x, S)
    shift_softmax = qint_softmax_shift.forward(x, S)
    softermax = softermax(x)
    plt.figure()
    plt.bar(label - 1, true_softmax, label='true_softmax')
    plt.bar(label, poly_softmax, label='poly_softmax')
    plt.bar(label + 1, shift_softmax, label='shift_softmax')
    plt.bar(label + 2, softermax, label='softermax')
    plt.legend()
    plt.savefig('softmax_uniform_2.png')
import matplotlib.pyplot as plt
import numpy as np
import torch

def f(x, scale, center):
    return torch.sigmoid(scale * x - center)

if __name__ == '__main__':
    x = torch.arange(-10, 10, 0.01)
    y = f(x, 10 ,75)
    plt.plot(x, y)
    plt.savefig('sigmoid_gate_2.png')
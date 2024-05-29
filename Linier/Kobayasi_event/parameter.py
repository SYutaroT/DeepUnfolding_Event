import torch
import os

T = 30
N = 2
Folder = os.getcwd()
Dt = "\\Result\\"


epoch = 1
beta_val = 0.1
h = 30/T
lr = 1e-2
lam = 5
datasize = 16
Batch = 1
x0 = torch.tensor([[20], [10]], dtype=torch.double)
sita = 5
phi = -0.5
rho = 0.16
wmax = 0.000
wmin = -0.000
K_s = torch.tensor([[0.2463, 1.2051]], dtype=torch.double)
P = torch.tensor([[398, 672], [672, 1880]], dtype=torch.double)
evolution = 1
Q = 10*torch.eye(N, dtype=torch.double)
R = torch.eye(1, dtype=torch.double)
Qf = 10*torch.eye(N, dtype=torch.double)


def x1_dot(x1, x2, u):
    return 1 * x1 + 0.8 * x2 + 1 * u


def x2_dot(x1, x2, u):
    return 1.1 * x2 - 1 * u

# def x1_dot(x1, x2, u):
#     return 4.0*x2


# def x2_dot(x1, x2, u):
#     return x1+u

import torch
import os
# Folder = os.getcwd()
Folder = os.getcwd()
Dt = "\\Result\\"


T = 100
N = 1
epoch = 50
beta_val = 0.1
h = 20/T
lr = 1e-1
lam = 3.0
datasize = 1
Batch = 1
x0 = torch.tensor([[1]], dtype=torch.double)
sita = 3
phi = 0.5
rho = 0.05
wmax = 0.0000
wmin = -0.0000
K_s = torch.tensor([[0, -1]], dtype=torch.double)
evolution = 1


def x1_dot(x1, x2, u):

    return x1+0.05*torch.abs(x1)+0.2*u


def x2_dot(x1, x2, u):
    return u

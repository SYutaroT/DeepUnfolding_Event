import torch
import parameter
import matplotlib.pyplot as plt

T = parameter.T
N = parameter.N
h = parameter.h
x0 = parameter.x0
K = parameter.K_s
P = parameter.P
Q = parameter.Q
R = parameter.R
sigma = parameter.rho
x = torch.zeros(N, T, dtype=torch.double)
x_hat = torch.zeros(N, T, dtype=torch.double)
u = torch.zeros(1, T-1, dtype=torch.double)
mark = torch.zeros(1, T-1, dtype=torch.double)
for i in range(0, N):
    x[i, 0] = x0[i]
    x_hat[i, 0] = x0[i]

A = torch.tensor([[1, 0.8], [0, 1.1]], dtype=torch.double)
B = torch.tensor([[1], [-1]], dtype=torch.double)


def f(x, u):
    Rx = torch.zeros(2, 1, dtype=torch.double)  # 注意：次元を修正しました。
    x1 = x[0]
    x2 = x[1]
    Rx[0] = parameter.x1_dot(x1, x2, u)
    Rx[1] = parameter.x2_dot(x1, x2, u)
    return Rx


# def f(x, u):
#     return A@x+B@u

u[:, 0] = K@x_hat[:, 0]
x[:, [1]] = f(x[:, 0], u[:, 0])
mark[:, 0] = 1
for i in range(1, T-1):
    if torch.norm(x[:, i-1]-x[:, i], p=2) > sigma*torch.norm(x[:, i], p=2):
        x_hat[:, i] = x[:, i]
        mark[:, i] = 1
    else:
        x_hat[:, i] = x_hat[:, i-1]
    u[:, i] = K@x_hat[:, i]
    x[:, [i+1]] = f(x[:, i], u[:, i])
cost = torch.tensor([0], dtype=torch.double)
for i in range(0, T-1):
    cost += x[:, i].T@Q@x[:, i]+u[:, i].T@R@u[:, i]
print(cost)

plt.figure()
for i in range(0,N):
    if i == 0:
        iro="red"
    else:
        iro="blue"
    plt.plot(range(T), x[i].squeeze().tolist(),
                "-", color=iro)

plt.xlabel('T')
plt.ylabel('X')
plt.grid()
plt.legend(fontsize=20)

plt.figure()

plt.plot(range(T-1), u.squeeze().cpu().tolist(),
         '-', color="red")
for i, val in enumerate(mark.squeeze().cpu().tolist()):
    if val == 1:
        # 例えば点のサイズを12に設定
        plt.plot(i, u.squeeze().cpu()[i], 'o', color="red", markersize=9)

plt.xlabel('T')
plt.ylabel('Probability')
plt.grid()
plt.legend(fontsize=10)
plt.figure()

plt.plot(range(T-1), mark.squeeze().cpu().tolist(),
         'o', color="red")

plt.xlabel('T')
plt.ylabel('Probability')
plt.grid()
plt.legend(fontsize=10)
plt.show()

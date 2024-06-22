# ライブラリのインポート

import torch  # 機械学習用のライブラリ
import torch.nn as nn  # ニューラルネットワーク構築
import torch.optim as optim  # 最適化
from torch.optim.lr_scheduler import StepLR  # 学習率調整
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学習率調整
import numpy as np  # 計算用のライブラリ
import matplotlib.pyplot as plt  # グラフのプロット
import plotly.graph_objects as go
import copy  # コピー
from random import randrange  # ランダム
from scipy.stats import beta  # ベータ分布
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR
import os
import parameter
import time
import winsound
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import importlib.util


# 使用例

# L2とL1を使用

# ディレクトリのパス

start_time = time.time()
# ?----------------------------------------------------------------保存
path = parameter.Folder+parameter.Dt
directory = os.path.dirname(path)


# ディレクトリが存在しない場合、作成する
if not os.path.exists(directory):
    os.makedirs(directory)
# ?----------------------------------------------------------------乱数固定


def seedinit(seed):
    torch.manual_seed(int(seed))
    np.random.seed(seed)


seedinit(1)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# !----------------------------------------------------------------数値設定(必要に応じて変更)
T = parameter.T
N = parameter.N
epoch = parameter.epoch
beta_val = parameter.beta_val
h = parameter.h
lr = parameter.lr
lam = parameter.lam
Batch = parameter.Batch
datasize = parameter.datasize
x0 = parameter.x0.to(device)
xf = torch.tensor([[0.0], [0.0]], dtype=torch.double, device=device)
Q = parameter.Q
R = parameter.R
Qf = parameter.Qf
# TODO グラフ保存
path2 = "e" + str(epoch) + "B" + str(Batch) + "w"+str(parameter.wmin)+str(parameter.wmax) + \
    "lr" + str(lr) + "T" + str(T) + "h" + str(h) + "lam" + str(lam)+"\\"
imgname = path+path2
data = imgname+"L2data.txt"
directory = os.path.dirname(imgname)
if not os.path.exists(directory):
    os.makedirs(directory)
module_path = os.path.join(os.getcwd(), imgname, "result.py")
# モジュールのspecをロード
spec = importlib.util.spec_from_file_location("result", module_path)
result = importlib.util.module_from_spec(spec)

# モジュールを実行してロード
spec.loader.exec_module(result)

# ?----------------------------------------------------------------関数


def f(x, u, h):
    Rx = torch.zeros(2, 1, dtype=torch.double).to(device)
    x1 = x[0]
    x2 = x[1]
    Rx[0] = parameter.x1_dot(x1, x2, u)
    Rx[1] = parameter.x2_dot(x1, x2, u)
    return Rx


def w_for_evaluation(T):  # !評価用の外乱
    alpha = 1
    beta_1 = 1
    minw = parameter.wmin
    maxw = parameter.wmax
    return torch.tensor(beta.rvs(alpha, beta_1, size=(N, T), loc=minw, scale=maxw - minw)).to(device)


def calculate_cost(w, T, N, x0, Q, R, Qf, lam, K, h, Name, No, L2, L1, L0):
    x = torch.zeros(N, T, dtype=torch.double).to(device)
    x_hat = torch.zeros((N, 1), dtype=torch.double).to(device)
    u = torch.zeros(1, T-1, dtype=torch.double).to(device)
    mark = torch.zeros(1, T-1, dtype=torch.double).to(device)
    cost = torch.zeros(1, dtype=torch.double).to(device)
    mark = torch.zeros(1, T-1, dtype=torch.double).to(device)
    cost_time = torch.zeros(1, dtype=torch.double).to(device)
    mark_time = torch.zeros(1, T-1, dtype=torch.double).to(device)
    cost_hikaku = torch.zeros(1, dtype=torch.double).to(device)
    mark_hikaku = torch.zeros(1, T-1, dtype=torch.double).to(device)
    Rx = torch.zeros(4, 1, dtype=torch.double).to(device)
    Rx_data = torch.zeros(1, T-1, dtype=torch.double).to(device)
    cnt = 0
    for i in range(0, N):
        x[i, 0] = x0[i]
        x_hat[i, 0] = x0[i]

    for i in range(0, T-1):
        if i == 0:
            mark[:, i] = 1
            Rx_data[:, i] = 1
        else:
            Rx[0] = x[0, [i]]
            Rx[1] = x[1, [i]]
            Rx[2] = x_hat[0, [0]]
            Rx[3] = x_hat[1, [0]]
            if 0 < Rx.T@L2@Rx+L1@Rx+L0:
                x_hat[:, [0]] = x[:, [i]]
                mark[:, i] = 1
                cnt += 1
            else:
                mark[:, i] = 0
            Rx_data[:, i] = torch.sigmoid(Rx.T@L2@Rx+L1@Rx+L0)
        u[:, [i]] = K@x_hat
        x[:, [i+1]] = f(x[:, [i]], u[:, [i]], h)+w[:, [i]]
    for i in range(0, T-1):
        cost = cost + x[:, i].T@Q@x[:, i]+u[:, i]@R@u[:, i]
    # cost = cost + x[:, T-1].T@Qf@x[:, T-1]
    imgname = Name+"_ev"+str(No)
    x_hikaku = torch.zeros(N, T, dtype=torch.double).to(device)
    x_hat_hikaku = torch.zeros(N, T, dtype=torch.double)
    u_hikaku = torch.zeros(1, T-1, dtype=torch.double).to(device)
    K_s = parameter.K_s
    sigma = parameter.rho
    for i in range(0, N):
        x_hikaku[i, 0] = x0[i]
        x_hat_hikaku[i, 0] = x0[i]
    u_hikaku[:, 0] = K_s@x_hat_hikaku[:, 0]
    x_hikaku[:, [1]] = f(x_hikaku[:, 0], u_hikaku[:, 0], h)
    mark_hikaku[:, 0] = 1
    for i in range(1, T-1):
        if torch.norm(x_hikaku[:, i-1]-x_hikaku[:, i], p=2) > sigma*torch.norm(x_hikaku[:, i], p=2):
            x_hat_hikaku[:, i] = x_hikaku[:, i]
            mark_hikaku[:, i] = 1
        else:
            x_hat_hikaku[:, i] = x_hat_hikaku[:, i-1]
        u_hikaku[:, i] = K_s@x_hat_hikaku[:, i]
        x_hikaku[:, [i+1]] = f(x_hikaku[:, i], u_hikaku[:, i], h)
    cost_hikaku = torch.tensor([0], dtype=torch.double)
    for i in range(0, T-1):
        cost_hikaku += x_hikaku[:, i].T@Q@x_hikaku[:,
                                                   i]+u_hikaku[:, i].T@R@u_hikaku[:, i]

    for i in range(0, N):
        plt.figure()
        plt.plot(range(T), x_hikaku[i].squeeze().tolist(),
                 "-", color="black")
        # plt.plot(range(T), x_t[i].squeeze().tolist(),
        #          "--", color="black",)
        plt.plot(range(T), x[i].squeeze().tolist(),
                 "-", color="red")

        plt.xlabel('T')
        plt.ylabel('X')
        plt.grid()
        plt.legend(fontsize=20)
        imgname1 = imgname + str(i+1)+".svg"
        plt.savefig(imgname1)

    plt.figure()
    # plt.plot(range(T-1), u_time.squeeze().cpu().tolist(),
    #          '--', color="blue")
    # for i, val in enumerate(mark3.squeeze().cpu().tolist()):
    #     if val == 1:
    #         plt.plot(i, u_time.squeeze().cpu()[
    #                  i], 'x', color="blue", markersize=9)
    plt.plot(range(T-1), u_hikaku.squeeze().cpu().tolist(),
             '--', color="black")
    for i, val in enumerate(mark_hikaku.squeeze().cpu().tolist()):
        if val == 1:
            plt.plot(i, u_hikaku.squeeze().cpu()[
                     i], 'x', color="black", markersize=9)
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
    imgname2 = imgname + "U.svg"
    plt.savefig(imgname2)
    plt.figure()

    plt.plot(range(T-1), mark.squeeze().cpu().tolist(),
             'o', color="red")
    plt.plot(range(T-1), mark_hikaku.squeeze().cpu().tolist(),
             'x', color="black")

    plt.xlabel('T')
    plt.ylabel('Probability')
    plt.grid()
    plt.legend(fontsize=10)
    imgname3 = imgname + "mark.svg"
    plt.savefig(imgname3)
    print(cost)
    print(cost_hikaku)
    return cost.item(), mark.sum().item()


# ?----------------------------------------------------------------
K = torch.tensor(result.K, dtype=torch.double).to(device)
L2 = torch.tensor(result.L2, dtype=torch.double).to(device)
L1 = torch.tensor(result.L1,
                  dtype=torch.double).to(device)
L0 = torch.tensor([result.L0
                   ], dtype=torch.double).to(device)

print("K", K)
print("L2", L2)
print("L1", L1)
print("L0", L0)
# ?----------------------------------------------------------------データの保存


if not os.path.exists(directory):
    os.makedirs(directory)
with open(data, "w") as file:
    file.write("Title:\n")
    file.write(f"{imgname}\n")
with open(data, "a") as file:  # 追加モード
    file.write("\nL0:\n")
    for value in L0:
        file.write(f"{value}\n")
with open(data, "a") as file:  # 追加モード
    file.write("\nL1:\n")
    for value in L1:
        file.write(f"{value}\n")
with open(data, "a") as file:  # 追加モード
    file.write("\nL2:\n")
    for value in L2:
        file.write(f"{value}\n")
with open(data, "a") as file:  # 追加モード
    file.write("\nK:\n")
    for value in K:
        file.write(f"{value}\n")
# ?----------------------------------------------------------------シミュレーション
counter = 0
total_cost = 0
total_cnt = 0
total_cost2 = 0
total_cnt2 = 0
total_Timecost = 0
w_list = []
for i in range(parameter.evolution):
    seedinit(i)
    w = w_for_evaluation(T-1)
    w_list.append(w)


for w in w_list:
    counter += 1
    cost, cnt = calculate_cost(
        w, T, N, x0, Q, R, Qf, lam, K, h, imgname, counter, L2, L1, L0
    )

    with open(data, "a") as file:  # 追加モード
        file.write("cost_ev"+str(counter)+"\n")
        file.write(f"{cost}\n")

    with open(data, "a") as file:
        file.write("cnt_ev"+str(counter)+"\n")
        file.write(f"{cnt}\n")
    # with open(data, "a") as file:  # 追加モード
    #     file.write("cost_ev_hikaku"+str(counter)+"\n")
    #     file.write(f"{cost2}\n")

    # with open(data, "a") as file:
    #     file.write("cnt_ev_hikaku"+str(counter)+"\n")
    #     file.write(f"{cnt2}\n")
    # with open(data, "a") as file:
    #     file.write("Time_cost"+str(counter)+"\n")
    #     file.write(f"{Timecost}\n")

    total_cost += cost
    total_cnt += cnt
    # total_cost2 += cost2
    # total_cnt2 += cnt2
    # total_Timecost += Timecost
average_cost = total_cost / counter
average_cnt = total_cnt / counter
# average_cost2 = total_cost2 / counter
# average_cnt2 = total_cnt2 / counter
# average_Timecost = total_Timecost / counter

with open(data, "a") as file:  # 追加モード
    file.write("cost_AVE\n")
    file.write(f"{average_cost}\n")
with open(data, "a") as file:
    file.write("cnt_AVE\n")
    file.write(f"{average_cnt}\n")
# with open(data, "a") as file:  # 追加モード
#     file.write("cost_AVE_2\n")
#     file.write(f"{average_cost2}\n")
# with open(data, "a") as file:
#     file.write("cnt_AVE_2\n")
#     file.write(f"{average_cnt2}\n")
# with open(data, "a") as file:
#     file.write("cnt_AVE_Time\n")
#     file.write(f"{average_Timecost}\n")

# L2 = L2.float()
# L1 = L1.float()
# L0 = L0.float()

# # Define the range for x1, x2, x_hat1, x_hat2
# x1_range = torch.linspace(-10, 10, 30)
# x2_range = torch.linspace(-10, 10, 30)
# x_hat1_range = torch.linspace(-10, 10, 30)
# x_hat2_range = torch.linspace(-10, 10, 30)

# # Initialize plots
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
# # Plot for x1, x_hat1, phi
# for x1 in x1_range:
#     for x_hat1 in x_hat1_range:
#         # x2 and x_hat2 are set to zero

#         Rx = torch.tensor([x1, 0, x_hat1, 0], device=device)
#         phi = Rx @ L2 @ Rx + L1 @ Rx + L0
#         ax1.scatter(x1.item(), x_hat1.item(), phi.item(), color='b')
# ax1.set_xlabel('x1')
# ax1.set_ylabel('x_hat1')
# ax1.set_zlabel('phi')
# ax1.set_title('Graph of x1, x_hat1, phi')

# # Plot for x2, x_hat2, phi
# for x2 in x2_range:
#     for x_hat2 in x_hat2_range:
#         # x1 and x_hat1 are set to zero
#         Rx = torch.tensor([0, x2, 0, x_hat2], device=device)
#         phi = Rx @ L2 @ Rx + L1 @ Rx + L0
#         ax2.scatter(x2.item(), x_hat2.item(), phi.item(), color='r')
# ax2.set_xlabel('x2')
# ax2.set_ylabel('x_hat2')
# ax2.set_zlabel('phi')
# ax2.set_title('Graph of x2, x_hat2, phi')
# plt.savefig(imgname+"graph.svg")
# # x1, x_hat1に基づくphiの計算
# X1, X_HAT1 = np.meshgrid(x1_range, x_hat1_range)
# PHI1 = np.zeros(X1.shape)
# for i in range(X1.shape[0]):
#     for j in range(X1.shape[1]):
#         Rx = torch.tensor(
#             [X1[i, j], 0, X_HAT1[i, j], 0], dtype=torch.float).to(device)
#         phi = Rx @ L2 @ Rx + L1 @ Rx + L0
#         PHI1[i, j] = phi.item()

#     # x2, x_hat2に基づくphiの計算
# X2, X_HAT2 = np.meshgrid(x2_range, x_hat2_range)
# PHI2 = np.zeros(X2.shape)
# for i in range(X2.shape[0]):
#     for j in range(X2.shape[1]):
#         Rx = torch.tensor(
#             [0, X2[i, j], 0, X_HAT2[i, j]], dtype=torch.float).to(device)
#         phi = Rx @ L2 @ Rx + L1 @ Rx + L0
#         PHI2[i, j] = phi.item()

#     # x1, x_hat1に基づく3Dプロット
# fig1 = go.Figure(data=[go.Surface(z=PHI1, x=X1, y=X_HAT1)])
# fig1.update_layout(title='Graph of x1, x_hat1, phi', autosize=False,
#                    width=500, height=500,
#                    margin=dict(l=65, r=50, b=65, t=90))

# # x2, x_hat2に基づく3Dプロット
# fig2 = go.Figure(data=[go.Surface(z=PHI2, x=X2, y=X_HAT2)])
# fig2.update_layout(title='Graph of x2, x_hat2, phi', autosize=False,
#                    width=500, height=500,
#                    margin=dict(l=65, r=50, b=65, t=90))

# グラフの表示
# fig1.show()
# fig2.show()
# ?----------------------------------------------------------------時間
time.sleep(2)  # 例として2秒間の遅延

# 処理終了後の時刻
end_time = time.time()

# 経過時間を計算
elapsed_time = end_time - start_time
print(f"Time: {elapsed_time} /s")
with open(data, "a") as file:  # Append mode
    file.write(f"\nTime: {elapsed_time} /s\n")
time.sleep(5)  # 5秒間待機するダミーの処理

# 処理終了後にビープ音を鳴らす
# winsound.Beep(1000, 1000)
plt.show()

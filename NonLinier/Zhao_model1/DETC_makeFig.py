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


# ?----------------------------------------------------------------ニューラルネットワークの構築


class DisturbanceDataset(Dataset):  # ミニバッチ学習用の学習データ作成プログラム
    def __init__(self, num_samples, N, T, minw, maxw, alpha, beta2):
        self.disturbances = [torch.tensor(beta.rvs(alpha, beta2, size=(
            N, T - 1), loc=minw, scale=maxw - minw)) for _ in range(num_samples)]

    def __len__(self):
        return len(self.disturbances)

    def __getitem__(self, idx):
        return self.disturbances[idx]


class CSTR(nn.Module):  # 深層展開のメインクラス
    def __init__(self, T, N, epoch, device, path):  # パラメータの定義
        super(CSTR, self).__init__()
        self.device = device
        self.x0 = torch.tensor([1], dtype=torch.double)
        self.xf = torch.tensor([0.0], dtype=torch.double)
        self.T = T
        self.N = N
# ?----------------------------------------------------------------最適化パラメータ

# ?----------------------------------------------------------------外乱設定
        self.alpha = 1
        self.beta = 1
        self.minw = parameter.wmin
        self.maxw = parameter.wmax
# ?----------------------------------------------------------------コスト関数

    def StageCost(self, x, u, delta):  # !ステージコスト
        return (x-self.xf).T @ self.Q @ (x-self.xf) + u.T @ self.R @ u+self.lam*delta

    def FinalCost(self, x):  # !終端コスト
        return (x-self.xf).T @ self.Qf @ (x-self.xf)


# ?----------------------------------------------------------------外乱生成


    def w_for_training(self):  # !訓練用の外乱
        return torch.tensor(beta.rvs(self.alpha, self.beta, size=(self.N, self.T - 1), loc=self.minw, scale=self.maxw - self.minw)).to(self.device)

    def w_for_evaluation(self, T):  # !評価用の外乱
        return torch.tensor(beta.rvs(self.alpha, self.beta, size=(self.N, T), loc=self.minw, scale=self.maxw - self.minw)).to(self.device)


# ?----------------------------------------------------------------ここからスタート
seedinit(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
# TODO グラフ保存
path2 = "e" + str(epoch) + "B" + str(Batch) + "w"+str(parameter.wmin)+str(parameter.wmax) + \
    "lr" + str(lr) + "T" + str(T) + "h" + str(h) + "lam" + str(lam)+"\\"
imgname = path+path2
data = imgname+"L2data.txt"
directory = os.path.dirname(imgname)
if not os.path.exists(directory):
    os.makedirs(directory)
cstr = CSTR(T, N, epoch, device, imgname)
sita = 5
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

    k1_x1 = parameter.x1_dot(x1, x2, u)
    k1_x2 = parameter.x2_dot(x1, x2, u)

    k2_x1 = parameter.x1_dot(x1, x2 + h / 2, u)
    k2_x2 = parameter.x2_dot(x1, x2 + h / 2, u)
    k3_x1 = parameter.x1_dot(x1, x2 + h / 2, u)
    k3_x2 = parameter.x2_dot(x1, x2 + h / 2, u)
    k4_x1 = parameter.x1_dot(x1, x2 + h, u)
    k4_x2 = parameter.x2_dot(x1, x2 + h, u)

    x1_dot_avg = (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1) / 6
    x2_dot_avg = (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2) / 6
    Rx[0] = x1 + h * x1_dot_avg
    Rx[1] = x2 + h * x2_dot_avg
    return Rx


def R_R(x, x_hat):
    e = x-x_hat
    norm = torch.norm(e, p=2)
    return rho(x)-norm


def phi(s):
    return -0.5*s


def rho(s):
    return 0.2*torch.norm(s, p=2)


def Z(x, x_hat, zeta):
    e = x-x_hat
    norm = torch.norm(e, p=2)
    zeta_dot = phi(zeta)+rho(x)-norm
    return zeta+h*zeta_dot


def J(x, u, Q, R):
    return x.T @ Q @ x + u.T @ R @ u


def calculate_cost(w, T, N, x0, Q, R, Qf, lam, K, h, Name, No, L2, L1, L0):
    x = torch.zeros(N, T, dtype=torch.double).to(device)
    x_hat = torch.zeros((N, 1), dtype=torch.double).to(device)
    u = torch.zeros(1, T-1, dtype=torch.double).to(device)
    mark = torch.zeros(1, T-1, dtype=torch.double).to(device)
    cost = torch.zeros(1, dtype=torch.double).to(device)
    mark = torch.zeros(1, T-1, dtype=torch.double).to(device)
    cost2 = torch.zeros(1, dtype=torch.double).to(device)
    mark2 = torch.zeros(1, T-1, dtype=torch.double).to(device)
    cost3 = torch.zeros(1, dtype=torch.double).to(device)
    mark3 = torch.zeros(1, T-1, dtype=torch.double).to(device)
    Rx = torch.zeros(2*N, 1, dtype=torch.double).to(device)
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
            cntA = cntB = 0
            for j in range(0, 2*N):
                if j < N:
                    Rx[j] = x[cntA, [i]]
                    cntA = cntA + 1
                else:
                    Rx[j] = x_hat[cntB, [0]]
                    cntB = cntB+1
            print(Rx)
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
        cost = cost + x[:, i].T@Q@x[:, i]+u[:, i]@R@u[:, i]+lam*mark[:, i]
    cost = cost + x[:, T-1].T@Qf@x[:, T-1]
    imgname = Name+"_ev"+str(No)

    x_t = torch.zeros(N, T, dtype=torch.double).to(device)
    u_t = torch.zeros(1, T-1, dtype=torch.double).to(device)
    xk = torch.zeros(N, dtype=torch.double).to(device)
    zeta = torch.zeros(1, T, dtype=torch.double).to(device)
    K_t = torch.tensor([[0, -1]], dtype=torch.double).to(device)

    for i in range(0, N):
        x_t[i, 0] = x0[i]
        xk[i] = x0[i]
    for i in range(T-1):
        u_t[:, i] = K_t@xk
        if zeta[:, i]+sita*R_R(x_t[:, i], xk) <= 0:
            xk = x_t[:, i]
            cnt += 1
            mark2[:, i] = 1
        for j in range(N):
            x_t[j, i+1] = f(x_t[:, i], u_t[:, i], h)[j]+w[j, i]
        zeta[0, i+1] = Z(x_t[:, i], xk, zeta[:, i])

    for i in range(0, T-1):
        cost2 = cost2 + x_t[:, i].T@Q@x_t[:, i] + \
            u_t[:, i]@R@u_t[:, i]+lam*mark2[:, i]
    cost2 = cost2 + x_t[:, T-1].T@Qf@x_t[:, T-1]

    x_time = torch.zeros(N, T, dtype=torch.double).to(device)
    xk_time = torch.zeros(N, dtype=torch.double).to(device)
    u_time = torch.zeros(1, T-1, dtype=torch.double).to(device)
    for i in range(0, N):
        x_time[i, 0] = x0[i]
        xk_time[i] = x0[i]
    for i in range(T-1):
        u_time[:, i] = K_t@xk_time
        xk_time = x_time[:, i]
        for j in range(N):
            x_time[j, i+1] = f(x_time[:, i], u_time[:, i], h)[j]+w[j, i]
        mark3[:, i] = 1
    for i in range(0, T-1):
        cost3 = cost3 + x_time[:, i].T@Q@x_time[:, i] + \
            u_time[:, i]@R@u_time[:, i]+lam*mark3[:, i]
    cost3 = cost3 + x_time[:, T-1].T@Qf@x_time[:, T-1]
    plt.figure()
    for i in range(0, N):

        # plt.plot(range(T), x_time[i].squeeze().tolist(),
        #          "-", color="blue")
        if i == 0:
            sen = "-"
        else:
            sen = "--"
        plt.plot(range(T), x_t[i].squeeze().tolist(),
                 sen, color="black",)
        plt.plot(range(T), x[i].squeeze().tolist(),
                 sen, color="red")

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
    #  i], 'x', color="blue", markersize=9)
    plt.plot(range(T-1), u_t.squeeze().cpu().tolist(),
             '--', color="black")
    for i, val in enumerate(mark2.squeeze().cpu().tolist()):
        if val == 1:
            plt.plot(i, u_t.squeeze().cpu()[
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
    imgname = imgname + "U.svg"
    plt.savefig(imgname)
    return cost.item(), mark.sum().item(), cost2.item(), mark2.sum(), cost3.item()


# ?----------------------------------------------------------------トレーニング
cstr.x0 = x0
cstr.xf = xf

cstr.h = h
cstr.lr = lr
cstr.lam = lam
cstr.batchSize = Batch
cstr.datasize = datasize

# ?----------------------------------------------------------------変数

K = torch.tensor([result.K
                  ], dtype=torch.double).to(device)

cnt = 0  # 更新回数
# ?----------------------------------------------------------------
L2 = torch.tensor(result.L2, dtype=torch.double).to(device)
L1 = torch.tensor(result.L1,
                  dtype=torch.double).to(device)
L0 = torch.tensor([result.L0
                   ], dtype=torch.double).to(device)
Q = parameter.Q.to(device)
Qf = parameter.Qf.to(device)
R = parameter.R.to(device)

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
    w = cstr.w_for_evaluation(T-1)
    w_list.append(w)


for w in w_list:
    counter += 1
    cost, cnt, cost2, cnt2, Timecost = calculate_cost(
        w, T, N, x0, Q, R, Qf, lam, K, h, imgname, counter, L2, L1, L0
    )

    with open(data, "a") as file:  # 追加モード
        file.write("cost_ev"+str(counter)+"\n")
        file.write(f"{cost}\n")

    with open(data, "a") as file:
        file.write("cnt_ev"+str(counter)+"\n")
        file.write(f"{cnt}\n")
    with open(data, "a") as file:  # 追加モード
        file.write("cost_ev_hikaku"+str(counter)+"\n")
        file.write(f"{cost2}\n")

    with open(data, "a") as file:
        file.write("cnt_ev_hikaku"+str(counter)+"\n")
        file.write(f"{cnt2}\n")
    with open(data, "a") as file:
        file.write("Time_cost"+str(counter)+"\n")
        file.write(f"{Timecost}\n")

    total_cost += cost
    total_cnt += cnt
    total_cost2 += cost2
    total_cnt2 += cnt2
    total_Timecost += Timecost
average_cost = total_cost / counter
average_cnt = total_cnt / counter
average_cost2 = total_cost2 / counter
average_cnt2 = total_cnt2 / counter
average_Timecost = total_Timecost / counter

with open(data, "a") as file:  # 追加モード
    file.write("cost_AVE\n")
    file.write(f"{average_cost}\n")
with open(data, "a") as file:
    file.write("cnt_AVE\n")
    file.write(f"{average_cnt}\n")
with open(data, "a") as file:  # 追加モード
    file.write("cost_AVE_2\n")
    file.write(f"{average_cost2}\n")
with open(data, "a") as file:
    file.write("cnt_AVE_2\n")
    file.write(f"{average_cnt2}\n")
with open(data, "a") as file:
    file.write("cnt_AVE_Time\n")
    file.write(f"{average_Timecost}\n")

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

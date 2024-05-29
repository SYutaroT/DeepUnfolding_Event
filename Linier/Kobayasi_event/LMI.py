import cvxpy as cp
import numpy as np

# システムの行列を定義
A = np.array([[1.0, 0.8], [0, 1.1]])
B = np.array([[1.0], [-1.0]])
Q = 10 * np.eye(2)
R = np.eye(1)
sigma = 0.16
n = A.shape[0]  # 状態の次元
m = B.shape[1]  # 制御入力の次元

# 変数の定義
S = cp.Variable((n, n), symmetric=True)
W = cp.Variable((m, n))
alpha = cp.Variable()
X = cp.vstack([S, W])

zero = np.zeros((n, n))
Y = cp.vstack([zero, W])
I_n = np.eye(n)
I_m = np.eye(m)
Q_d = np.block([
    [Q, np.zeros((n, m))],
    [np.zeros((m, n)), R]
])
print(Q_d)
# 目的関数
objective = cp.Minimize(cp.trace(S**-1))

kakunin = cp.bmat([
    [S, zero, (A@S+B@W).T, S.T, (Q_d**0.5@cp.vstack([S, W])).T],
    [zero, 2*S-alpha*I_n, (B@W).T, zero, (Q_d**0.5@cp.vstack([zero, W])).T],
    [A@S+B@W, B@W, S, zero, np.zeros((n, n+m))],
    [S, zero, zero, (alpha/sigma**2)*I_n, np.zeros((n, n+m))],
    [Q_d**0.5@cp.vstack([S, W]), Q_d**0.5@cp.vstack([zero, W]),
     np.zeros((n+m, n)), np.zeros((n+m, n)), np.eye(n+m)]

])

# 制約条件
constraints = [
    kakunin >> 0,
    S == S.T,
    S >> 0,
    alpha >= 0.0000000001
]

# 問題を定義して解く
prob = cp.Problem(objective, constraints)
prob.solve()
# 結果の表示
print("Status:", prob.status)
P = S**-1
K = W@P
print("P", P.value)
print("K", K.value)

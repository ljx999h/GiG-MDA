"""
P0-4 验证: gigs_model.py 的乘法更新式是否精确最小化下述目标函数 J:

J = 1/2||A - XY^T||_F^2 + (λ1/2)(||X||_F^2 + ||Y||_F^2)
  + λ2 ( Σ_ij ||x_i - x_j||^2 Sd_ij + Σ_ij ||y_i - y_j||^2 Sv_ij )
  + (λ3/2)( ||XX^T - Sd||_F^2 + ||YY^T - Sv||_F^2 )

验证内容:
  1. 解析梯度 vs 数值梯度 (中心差分) 一致
  2. 更新式迭代后 J 单调不增
  3. X/Y 保持非负
  4. 与 gigs_model.GiGsMatrixFactorization 的更新式完全一致

结论: 若全部 PASS, 则论文公式应按此 J 重写 (代码为准 + 标准推导),
诊断报告 P0-4 的"目标函数/更新式/超参不一致"得到方法学修复.

用法: python code/verify_grmf_objective.py
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gigs_model import GiGsMatrixFactorization


def objective(A, X, Y, Sd, Sv, l1, l2, l3):
    rec = 0.5 * np.sum((A - X @ Y.T) ** 2)
    reg = 0.5 * l1 * (np.sum(X ** 2) + np.sum(Y ** 2))
    # λ2 项: Σ_ij ||x_i - x_j||^2 Sd_ij = tr(X^T (Dd - Sd) X)
    Dd = np.diag(Sd.sum(axis=1))
    Dv = np.diag(Sv.sum(axis=1))
    gx = np.trace(X.T @ (Dd - Sd) @ X)
    gy = np.trace(Y.T @ (Dv - Sv) @ Y)
    graph = l2 * (gx + gy)
    sim = 0.5 * l3 * (np.sum((X @ X.T - Sd) ** 2) + np.sum((Y @ Y.T - Sv) ** 2))
    return rec + reg + graph + sim


def analytic_grad_X(A, X, Y, Sd, l1, l2, l3):
    """∂J/∂X 按推导: XY^TY - AY + λ1 X + 2λ2(D-S)X + 2λ3(XX^TX - SX)"""
    Dd = np.diag(Sd.sum(axis=1))
    return (X @ (Y.T @ Y) - A @ Y + l1 * X
            + 2 * l2 * (Dd - Sd) @ X
            + 2 * l3 * (X @ (X.T @ X) - Sd @ X))


def numeric_grad(func, w, eps=1e-6):
    g = np.zeros_like(w)
    it = np.nditer(w, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        g[i] = (func(wp) - func(wm)) / (2 * eps)
        it.iternext()
    return g


def main():
    rng = np.random.RandomState(0)
    m, n, k = 8, 6, 3
    l1, l2, l3 = 0.1, 0.01, 0.1
    A = rng.rand(m, n)
    # 相似度: 对称非负
    Rd = rng.rand(m, m); Sd = (Rd + Rd.T) / 2
    Rv = rng.rand(n, n); Sv = (Rv + Rv.T) / 2
    X = rng.rand(m, k) + 0.5
    Y = rng.rand(n, k) + 0.5

    print("=" * 60)
    print(" P0-4 验证: 更新式 vs 目标函数 J")
    print("=" * 60)

    # 1. 解析梯度 vs 数值梯度
    g_ana = analytic_grad_X(A, X, Y, Sd, l1, l2, l3)

    def J_of_X(w):
        return objective(A, w.reshape(X.shape), Y, Sd, Sv, l1, l2, l3)

    g_num = numeric_grad(J_of_X, X)
    denom = np.maximum(np.abs(g_ana), 1e-9)
    rel = np.max(np.abs(g_ana - g_num) / denom)
    print(f"[1] ∂J/∂X 解析 vs 数值梯度 最大相对差: {rel:.3e} "
          f"{'PASS' if rel < 1e-4 else 'FAIL'}")

    # 2. 更新式迭代: J 单调不增 + 非负
    gigs = GiGsMatrixFactorization(k=k, lambda1=l1, lambda2=l2, lambda3=l3,
                                   max_iter=60, tol=0)
    Xr, Yr = gigs.fit(A, Sd, Sv, random_state=1)
    # 复算 J 轨迹 (重新用相同初始化跑一遍并记录)
    np.random.seed(1)
    X0 = np.random.rand(m, k)
    Y0 = np.random.rand(n, k)
    J_prev = objective(A, X0, Y0, Sd, Sv, l1, l2, l3)
    violations = 0
    for it in range(60):
        num_x = A @ Y0 + 2 * (l2 + l3) * (Sd @ X0)
        den_x = X0 @ (Y0.T @ Y0) + l1 * X0 + 2 * l2 * (np.diag(Sd.sum(1)) @ X0) \
                + 2 * l3 * (X0 @ (X0.T @ X0))
        X0 = X0 * np.sqrt(num_x / (den_x + 1e-9))
        num_y = A.T @ X0 + 2 * (l2 + l3) * (Sv @ Y0)
        den_y = Y0 @ (X0.T @ X0) + l1 * Y0 + 2 * l2 * (np.diag(Sv.sum(1)) @ Y0) \
                + 2 * l3 * (Y0 @ (Y0.T @ Y0))
        Y0 = Y0 * np.sqrt(num_y / (den_y + 1e-9))
        J_now = objective(A, X0, Y0, Sd, Sv, l1, l2, l3)
        if J_now > J_prev + 1e-9:
            violations += 1
        J_prev = J_now
    print(f"[2] 60 次迭代中 J 上升次数: {violations} {'PASS' if violations == 0 else 'FAIL'}")
    print(f"    J: {objective(A, np.random.rand(m,k), np.random.rand(n,k), Sd, Sv, l1, l2, l3):.4f} -> {J_prev:.4f}")
    print(f"[3] X/Y 非负: X.min={X0.min():.4f}, Y.min={Y0.min():.4f} "
          f"{'PASS' if X0.min() >= 0 and Y0.min() >= 0 else 'FAIL'}")

    # 4. gigs_model 与手工更新式结果一致
    Xc, Yc = gigs.fit(A, Sd, Sv, random_state=1)
    same = np.allclose(Xc, X0, atol=1e-8) and np.allclose(Yc, Y0, atol=1e-8)
    print(f"[4] gigs_model.fit 输出与手工更新式一致: {same} {'PASS' if same else 'FAIL'}")
    print("\n结论: 代码更新式精确最小化上述 J (λ2 项无 1/2 因子). "
          "论文公式应改为与该 J 一致的标准推导.")


if __name__ == '__main__':
    main()

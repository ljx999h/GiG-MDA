#实现 GiGs 的核心算法，基于图正则化的矩阵分解。
import numpy as np

class GiGsMatrixFactorization:
    def __init__(self, k=70, lambda1=16, lambda2=0.125, lambda3=16, max_iter=200, tol=1e-4):
        """
        初始化 GiGs 模型参数
        :param k: 潜在特征维度 (Latent feature dimension)
        :param lambda1: 正则化参数 (Regularization parameter)
        :param lambda2: 图正则化参数 (Graph regularization parameter)
        :param lambda3: 相似度约束参数 (Similarity constraint parameter)
        """
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.max_iter = max_iter
        self.tol = tol
        self.X = None
        self.Y = None

    def fit(self, A, Sd, Sv):
        """
        训练模型
        :param A: 关联矩阵 (m x n), m=drugs, n=diseases
        :param Sd: 药物相似度矩阵 (m x m)
        :param Sv: 疾病相似度矩阵 (n x n)
        """
        m, n = A.shape
        
        # 1. 初始化 X 和 Y (随机初始化)
        np.random.seed(42)
        self.X = np.random.rand(m, self.k)
        self.Y = np.random.rand(n, self.k)
        
        # 2. 预计算 R (权重矩阵, 这里简化为 R=1 for all, 或者可以对已知关联加权)
        # GiGs 原文中 R 对于已知关联为 1，未知为 1 (或者是加权的，这里简化处理)
        # 为了矩阵运算方便，我们使用乘法更新规则
        
        last_loss = float('inf')

        print(f"开始 GiGs 矩阵分解训练 (Dimensions: {m}x{n}, Latent K: {self.k})...")
        
        for epoch in range(self.max_iter):
            # --- 更新 X (Drugs) ---
            # 分子: (A * Y) + 2*(lambda2 + lambda3) * (Sd * X)
            # 注意：A @ Y 计算量较大，但必须进行
            numerator_x = (A @ self.Y) + 2 * (self.lambda2 + self.lambda3) * (Sd @ self.X)
            
            # 分母: (X * Y^T * Y) + lambda1 * X + 2*lambda2*(Dd * X) + 2*lambda3*(X * X^T * X)
            # 简化版 GiGs 更新规则 (假设 Dd 为度矩阵等同于 Sd 的行和)
            # 为了数值稳定性，分母加上一个小 epsilon
            XYTY = self.X @ (self.Y.T @ self.Y)
            denominator_x = XYTY + self.lambda1 * self.X + \
                            2 * self.lambda2 * (np.diag(np.sum(Sd, axis=1)) @ self.X) + \
                            2 * self.lambda3 * (self.X @ (self.X.T @ self.X))
            
            self.X = self.X * np.sqrt(numerator_x / (denominator_x + 1e-9))

            # --- 更新 Y (Diseases) ---
            # 分子
            numerator_y = (A.T @ self.X) + 2 * (self.lambda2 + self.lambda3) * (Sv @ self.Y)
            
            # 分母
            YXTX = self.Y @ (self.X.T @ self.X)
            denominator_y = YXTX + self.lambda1 * self.Y + \
                            2 * self.lambda2 * (np.diag(np.sum(Sv, axis=1)) @ self.Y) + \
                            2 * self.lambda3 * (self.Y @ (self.Y.T @ self.Y))
            
            self.Y = self.Y * np.sqrt(numerator_y / (denominator_y + 1e-9))

            # --- 简单的收敛检查 (可选) ---
            if epoch % 10 == 0:
                # 计算重构误差 ||A - XY^T||^2 (仅做参考，计算很耗时，可跳过)
                diff = np.mean((A - self.X @ self.Y.T)**2)
                print(f"Epoch {epoch}/{self.max_iter}, MSE Loss: {diff:.6f}")
                if abs(last_loss - diff) < self.tol:
                    print("收敛！")
                    break
                last_loss = diff
        
        print("GiGs 训练完成。")
        return self.X, self.Y

    def predict_score(self, drug_idx, disease_idx):
        """获取指定对的预测分数"""
        if self.X is None:
            raise ValueError("Model not trained yet!")
        # 向量点积
        return np.dot(self.X[drug_idx], self.Y[disease_idx])
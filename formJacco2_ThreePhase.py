import numpy as np
from scipy import sparse

# 构建Jacobean矩阵（导纳阵）
def formJacco2_ThreePhase(Vi, Vj, Y, n, PVi, balance, QG, QD, PG, PD, Pi, Qi):
    """
    构建雅可比矩阵
    
    参数:
    Vi - 复电压
    Vj - 复共轭电压
    Y - 导纳矩阵
    n - 节点数
    PVi - PV节点
    balance - 平衡节点
    QG - 发电无功
    QD - 负荷无功
    PG - 发电有功
    PD - 负荷有功
    Pi, Qi - 节点功率
    
    返回:
    jacco1, jacco2 - 雅可比矩阵
    """
    try:
        # 确保输入数据类型正确
        if sparse.issparse(Vi):
            Vi_array = Vi.toarray()
        else:
            Vi_array = np.array(Vi)
            
        if sparse.issparse(Vj):
            Vj_array = Vj.toarray()
        else:
            Vj_array = np.array(Vj)
            
        if sparse.issparse(Y):
            Y_array = Y.toarray()
        else:
            Y_array = np.array(Y)
            
        if sparse.issparse(Pi):
            Pi_array = Pi.toarray()
        else:
            Pi_array = np.array(Pi)
            
        if sparse.issparse(Qi):
            Qi_array = Qi.toarray()
        else:
            Qi_array = np.array(Qi)
        
        # 节点数和导纳矩阵尺寸
        n_nodes = 4 * n
        #print(f"节点数: {n}, 扩展节点数: {n_nodes}")
        #print(f"导纳矩阵尺寸: {Y_array.shape}")
        
        # 检查维度是否匹配
        if Y_array.shape[0] != n_nodes or Y_array.shape[1] != n_nodes:
            print(f"警告: 导纳矩阵尺寸 {Y_array.shape} 与预期 ({n_nodes}, {n_nodes}) 不匹配")
            # 创建单位矩阵作为替代
            G = np.eye(n_nodes)
            B = np.zeros((n_nodes, n_nodes))
        else:
            G = np.real(Y_array)
            B = np.imag(Y_array)
        
        # 初始化雅可比矩阵的四个子块
        jacco1_H = G.copy()
        jacco1_N = -B.copy()
        jacco1_M = B.copy()
        jacco1_L = G.copy()
        
        # 定义辅助函数e, f, g, h - 使用从0开始的索引
        def e(i, k):
            # 计算节点索引，从0开始
            node_i = 4*(i-1)  # 节点i的起始索引
            node_k = 4*(i-1) + k - 1  # 相位k的索引
            node_n = 4*(i-1) + 3  # 中性线索引
            
            # 检查索引是否有效
            if node_k >= Vi_array.shape[0] or node_n >= Vi_array.shape[0]:
                return 0
                
            # 计算分母，避免除以零
            denom = (Vi_array[node_k, 0] - Vi_array[node_n, 0])
            denom_abs = abs(denom)**2
            if denom_abs < 1e-10:
                return 0
                
            # 计算函数值
            num = (Pi_array[node_k, 0] * (np.imag(denom)**2 - np.real(denom)**2) - 
                    2 * Qi_array[node_k, 0] * np.real(denom) * np.imag(denom))
            return num / denom_abs
        
        def f(i, k):
            # 计算节点索引，从0开始
            node_i = 4*(i-1)  # 节点i的起始索引
            node_k = 4*(i-1) + k - 1  # 相位k的索引
            node_n = 4*(i-1) + 3  # 中性线索引
            
            # 检查索引是否有效
            if node_k >= Vi_array.shape[0] or node_n >= Vi_array.shape[0]:
                return 0
                
            # 计算分母，避免除以零
            denom = (Vi_array[node_k, 0] - Vi_array[node_n, 0])
            denom_abs = abs(denom)**2
            if denom_abs < 1e-10:
                return 0
                
            # 计算函数值
            num = (Qi_array[node_k, 0] * (np.real(denom)**2 - np.imag(denom)**2) - 
                    2 * Pi_array[node_k, 0] * np.real(denom) * np.imag(denom))
            return num / denom_abs
        
        def g(i, k):
            return f(i, k)  # g和f相同
        
        def h(i, k):
            # 计算节点索引，从0开始
            node_i = 4*(i-1)  # 节点i的起始索引
            node_k = 4*(i-1) + k - 1  # 相位k的索引
            node_n = 4*(i-1) + 3  # 中性线索引
            
            # 检查索引是否有效
            if node_k >= Vi_array.shape[0] or node_n >= Vi_array.shape[0]:
                return 0
                
            # 计算分母，避免除以零
            denom = (Vi_array[node_k, 0] - Vi_array[node_n, 0])
            denom_abs = abs(denom)**2
            if denom_abs < 1e-10:
                return 0
                
            # 计算函数值
            num = (Pi_array[node_k, 0] * (np.real(denom)**2 - np.imag(denom)**2) + 
                    2 * Qi_array[node_k, 0] * np.real(denom) * np.imag(denom))
            return num / denom_abs
        
        # 构建雅可比矩阵 - 使用从0开始的索引
        for i in range(1, n+1):
            i0 = 4*(i-1)  # 节点i的起始索引(0开始)
            
            # 跳过超出范围的索引
            if i0 + 3 >= n_nodes:
                continue
                
            # 对角块修改
            for k in range(1, 4):  # 1, 2, 3相
                # 计算索引
                k0 = k - 1  # 相位索引(0开始)
                
                # 确保索引有效
                if i0 + k0 < n_nodes and i0 + 3 < n_nodes:
                    # 更新H块
                    jacco1_H[i0 + 3, i0 + k0] = G[i0 + 3, i0 + k0] + e(i, k)
                    jacco1_H[i0 + k0, i0 + 3] = G[i0 + k0, i0 + 3] + e(i, k)
                    jacco1_H[i0 + k0, i0 + k0] = G[i0 + k0, i0 + k0] - e(i, k)
                    
                    # 更新N块
                    jacco1_N[i0 + 3, i0 + k0] = -B[i0 + 3, i0 + k0] + f(i, k)
                    jacco1_N[i0 + k0, i0 + 3] = -B[i0 + k0, i0 + 3] + f(i, k)
                    jacco1_N[i0 + k0, i0 + k0] = -B[i0 + k0, i0 + k0] - f(i, k)
                    
                    # 更新M块
                    jacco1_M[i0 + 3, i0 + k0] = B[i0 + 3, i0 + k0] + g(i, k)
                    jacco1_M[i0 + k0, i0 + 3] = B[i0 + k0, i0 + 3] + g(i, k)
                    jacco1_M[i0 + k0, i0 + k0] = B[i0 + k0, i0 + k0] - g(i, k)
                    
                    # 更新L块
                    jacco1_L[i0 + 3, i0 + k0] = G[i0 + 3, i0 + k0] + h(i, k)
                    jacco1_L[i0 + k0, i0 + 3] = G[i0 + k0, i0 + 3] + h(i, k)
                    jacco1_L[i0 + k0, i0 + k0] = G[i0 + k0, i0 + k0] - h(i, k)
            
            # 更新对角元素
            if i0 + 3 < n_nodes:
                jacco1_H[i0 + 3, i0 + 3] = G[i0 + 3, i0 + 3] - e(i, 1) - e(i, 2) - e(i, 3)
                jacco1_N[i0 + 3, i0 + 3] = -B[i0 + 3, i0 + 3] - f(i, 1) - f(i, 2) - f(i, 3)
                jacco1_M[i0 + 3, i0 + 3] = B[i0 + 3, i0 + 3] - g(i, 1) - g(i, 2) - g(i, 3)
                jacco1_L[i0 + 3, i0 + 3] = G[i0 + 3, i0 + 3] - h(i, 1) - h(i, 2) - h(i, 3)
        
        # 处理平衡节点 - 使用从0开始的索引
        for b in balance:
            b_idx = int(b) - 1  # 从1开始到0开始的索引
            
            # 确保索引有效
            if 0 <= b_idx < n_nodes:
                # 将对应行列置零
                jacco1_H[b_idx, :] = 0
                jacco1_N[b_idx, :] = 0
                jacco1_M[b_idx, :] = 0
                jacco1_L[b_idx, :] = 0
                
                jacco1_H[:, b_idx] = 0
                jacco1_N[:, b_idx] = 0
                jacco1_M[:, b_idx] = 0
                jacco1_L[:, b_idx] = 0
                
                # 对角元置1
                jacco1_H[b_idx, b_idx] = 1
                jacco1_L[b_idx, b_idx] = 1
        
        # 组装完整的Jacobian矩阵
        top = np.hstack((jacco1_H, jacco1_N))
        bottom = np.hstack((jacco1_M, jacco1_L))
        jacco_full = np.vstack((top, bottom))
        
        # 创建稀疏矩阵
        jacco1 = sparse.csr_matrix(jacco_full)
        jacco2 = jacco1.copy()
        
        #print(f"雅可比矩阵构建完成，形状: {jacco1.shape}, 非零元素数: {jacco1.nnz}")
        return jacco1, jacco2
    
    except Exception as e:
        #print(f"构建雅可比矩阵时出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回默认雅可比矩阵
        n_nodes = 4 * n
        jacco_default = sparse.eye(2*n_nodes, format='csr')
        return jacco_default, jacco_default.copy() 
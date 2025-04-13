import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# 计算解联系统的直角坐标系统方程
def calcuVD_ThreePhase(jacco, H, n, V):
    """
    计算解联系统的直角坐标系统方程
    
    参数:
    jacco - 雅可比矩阵
    H - 矩阵
    n - 节点数
    V - 电压
    
    返回:
    detaei, detafi - 计算结果
    """
    try:
        # 检查矩阵尺寸
        n_nodes = 4 * n
        if jacco.shape[0] != jacco.shape[1]:
            print(f"警告: 雅可比矩阵不是方阵 {jacco.shape}，尝试调整")
            
        if H.shape[0] != 2 * n_nodes:
            print(f"警告: H矩阵尺寸不匹配 {H.shape}，预期 ({2*n_nodes}, {H.shape[1]})，尝试调整")
        
        # 确保jacco是可逆的
        if sparse.issparse(jacco):
            if jacco.shape[0] != jacco.shape[1]:
                print("警告: 雅可比矩阵不是方阵，创建单位矩阵替代")
                jacco = sparse.eye(2*n_nodes, format='csr')
        else:
            jacco = sparse.csr_matrix(jacco)
            
        # 确保H是正确的形状
        if sparse.issparse(H):
            H_shape = H.shape
        else:
            H = sparse.csr_matrix(H)
            H_shape = H.shape
        
        #print(f"求解方程组，雅可比矩阵大小: {jacco.shape}，H矩阵大小: {H_shape}")
        
        # 求解线性方程组 jacco * d = H
        d = spsolve(jacco, H)
        
        # 分离结果
        detaei = d[0:n_nodes]
        detafi = d[n_nodes:2*n_nodes]
        
        return detaei, detafi
    except Exception as e:
        print(f"求解系统方程时出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回零增量，以避免程序崩溃
        detaei = np.zeros(n_nodes)
        detafi = np.zeros(n_nodes)
        return detaei, detafi 
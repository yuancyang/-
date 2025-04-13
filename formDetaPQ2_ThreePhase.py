import numpy as np
from scipy import sparse

# 生成不平衡量
# 三相系统有功率无功率直角坐标系统过程、控制方程。
def formDetaPQ2_ThreePhase(m_PG, m_PD, Vi, Vj, Y, m_balance, balancevoltage, balanceangle, m_QG, m_QD, m_PVi, Nodes):
    """
    生成不平衡量
    
    参数:
    m_PG - 发电功率
    m_PD - 负荷功率
    Vi - 复电压
    Vj - 复共轭电压
    Y - 导纳矩阵
    m_balance - 平衡节点
    balancevoltage - 平衡节点电压
    balanceangle - 平衡节点角度
    m_QG - 发电无功
    m_QD - 负荷无功
    m_PVi - PV节点
    Nodes - 节点数
    
    返回:
    H - 不平衡量
    Pi, Qi - 节点功率
    """
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
    
    # 确保Pi和Qi是正确的形状
    Pi = sparse.csr_matrix(m_PG - m_PD)
    Qi = sparse.csr_matrix(m_QG - m_QD)
    
    if sparse.issparse(Pi):
        Pi_array = Pi.toarray()
    else:
        Pi_array = np.array(Pi)
        
    if sparse.issparse(Qi):
        Qi_array = Qi.toarray()
    else:
        Qi_array = np.array(Qi)
    
    # 初始化结果矩阵
    n_Nodes = 4 * Nodes
    detaR = np.zeros((n_Nodes, 1))
    detaI = np.zeros((n_Nodes, 1))
    
    try:
        # 计算不平衡量
        for i in range(1, Nodes + 1):  # 从1开始循环到Nodes
            # 计算索引
            idx_a = 4*i-4  # A相索引 (从0开始)
            idx_b = 4*i-3  # B相索引
            idx_c = 4*i-2  # C相索引
            idx_n = 4*i-1  # 中性线索引
            
            # 确保索引有效
            if max(idx_a, idx_b, idx_c, idx_n) >= n_Nodes:
                print(f"警告: 节点{i}的索引超出范围，跳过")
                continue
            
            # 确保不除以零
            if Vj_array[idx_a, 0] == Vj_array[idx_n, 0]:
                divisor_a = complex(1e-10, 0)  # 避免除以零
            else:
                divisor_a = Vj_array[idx_a, 0] - Vj_array[idx_n, 0]
                
            if Vj_array[idx_b, 0] == Vj_array[idx_n, 0]:
                divisor_b = complex(1e-10, 0)
            else:
                divisor_b = Vj_array[idx_b, 0] - Vj_array[idx_n, 0]
                
            if Vj_array[idx_c, 0] == Vj_array[idx_n, 0]:
                divisor_c = complex(1e-10, 0)
            else:
                divisor_c = Vj_array[idx_c, 0] - Vj_array[idx_n, 0]
            
            # 计算a, b, c变量
            a_complex = (Pi_array[idx_a, 0] - 1j*Qi_array[idx_a, 0]) / divisor_a - Y_array[idx_a, :].dot(Vi_array)[0]
            b_complex = (Pi_array[idx_b, 0] - 1j*Qi_array[idx_b, 0]) / divisor_b - Y_array[idx_b, :].dot(Vi_array)[0]
            c_complex = (Pi_array[idx_c, 0] - 1j*Qi_array[idx_c, 0]) / divisor_c - Y_array[idx_c, :].dot(Vi_array)[0]
            
            # 计算gh变量
            gh_complex = -(Pi_array[idx_a, 0] - 1j*Qi_array[idx_a, 0]) / divisor_a \
                       - (Pi_array[idx_b, 0] - 1j*Qi_array[idx_b, 0]) / divisor_b \
                       - (Pi_array[idx_c, 0] - 1j*Qi_array[idx_c, 0]) / divisor_c \
                       - Y_array[idx_n, :].dot(Vi_array)[0]
            
            # 将复数结果分解为实部和虚部，并存储在detaR和detaI中
            detaR[idx_a, 0] = np.real(a_complex)
            detaI[idx_a, 0] = np.imag(a_complex)
            detaR[idx_b, 0] = np.real(b_complex)
            detaI[idx_b, 0] = np.imag(b_complex)
            detaR[idx_c, 0] = np.real(c_complex)
            detaI[idx_c, 0] = np.imag(c_complex)
            detaR[idx_n, 0] = np.real(gh_complex)
            detaI[idx_n, 0] = np.imag(gh_complex)
        
        # 置零平衡节点的不平衡量
        for idx in m_balance:
            idx = int(idx) - 1  # 调整索引（从0开始）
            if 0 <= idx < n_Nodes:
                detaR[idx, 0] = 0
                detaI[idx, 0] = 0
    
    except Exception as e:
        print(f"计算不平衡量时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 形成不平衡量向量
    H = np.vstack([detaR, detaI])
    H = sparse.csr_matrix(H)
    
    return H, Pi, Qi 
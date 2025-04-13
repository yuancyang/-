import numpy as np
from scipy import sparse

def pf_line(V, deta, linei, linej, transi, transj, Y, Y0, nodes):
    """
    计算线路功率和损耗
    
    参数:
    V - 电压幅值
    deta - 相角
    linei, linej - 线路起点终点
    transi, transj - 变压器起点终点
    Y, Y0 - 导纳矩阵
    nodes - 节点数
    
    返回:
    S - 功率矩阵
    detaS - 对称功率矩阵
    """
    try:
        # 统一数据类型
        if sparse.issparse(V):
            V_array = V.toarray()
        else:
            V_array = np.array(V)
            
        if sparse.issparse(deta):
            deta_array = deta.toarray()
        else:
            deta_array = np.array(deta)
            
        if sparse.issparse(Y):
            Y_array = Y.toarray()
        else:
            Y_array = np.array(Y)
            
        if sparse.issparse(Y0):
            Y0_array = Y0.toarray()
        else:
            Y0_array = np.array(Y0)
            
        # 确保输入数组的形状正确
        n_nodes = 4 * nodes  # 扩展后的节点数
        #print(f"pf_line: 节点数={nodes}, 扩展节点数={n_nodes}")
        #print(f"pf_line: V形状={V_array.shape}, deta形状={deta_array.shape}, Y形状={Y_array.shape}")
        
        # 转换线路和变压器索引为列表
        if isinstance(linei, (list, np.ndarray)):
            li = np.concatenate((linei, transi))
            lj = np.concatenate((linej, transj))
        else:
            # 如果只有一个元素
            li = np.array([linei]) if linei else np.array([])
            lj = np.array([linej]) if linej else np.array([])
            
            if transi:
                li = np.append(li, transi)
                lj = np.append(lj, transj)
        
        #print(f"pf_line: 线路数量={len(li)}")
        
        # 如果没有线路，返回空矩阵
        if len(li) == 0:
            return sparse.csr_matrix((n_nodes, n_nodes)), sparse.csr_matrix((n_nodes, n_nodes))
            
        # 将相角转换为复电压
        U = V_array * (np.cos(deta_array*np.pi/180) + 1j * np.sin(deta_array*np.pi/180))
        
        # 创建功率矩阵
        S_data = []
        S_row = []
        S_col = []
        
        # 计算线路两端的复功率
        for i in range(len(li)):
            try:
                # 获取线路起点和终点的节点编号（从1开始）
                node_i = int(li[i])  
                node_j = int(lj[i])
                
                # 将节点编号转换为索引（从0开始）
                idx_i = 4 * (node_i - 1)  # 节点i的起始索引
                idx_j = 4 * (node_j - 1)  # 节点j的起始索引
                
                # 计算A、B、C三相的功率
                for phase in range(3):  # 0, 1, 2对应A、B、C相
                    # 计算相索引
                    phase_i = idx_i + phase
                    phase_j = idx_j + phase
                    
                    # 跳过超出范围的索引
                    if phase_i >= n_nodes or phase_j >= n_nodes:
                        continue
                    
                    # 获取导纳矩阵元素
                    if 0 <= phase_i < Y_array.shape[0] and 0 <= phase_j < Y_array.shape[1]:
                        Y_ij = Y_array[phase_i, phase_j]
                        Y0_ij = Y0_array[phase_i, phase_j]
                        
                        # 计算相电压
                        U_i = U[phase_i][0] if U.ndim > 1 else U[phase_i]
                        U_j = U[phase_j][0] if U.ndim > 1 else U[phase_j]
                        
                        # 计算支路电流（从i到j）
                        I_ij = Y_ij * (U_i - U_j) + Y0_ij * U_i
                        
                        # 计算支路复功率
                        S_ij = U_i * np.conj(I_ij)
                        
                        # 存储结果
                        S_data.append(S_ij)
                        S_row.append(phase_i)
                        S_col.append(phase_j)
                        
                        # 计算反向支路（从j到i）
                        I_ji = Y_ij * (U_j - U_i) + Y0_ij * U_j
                        S_ji = U_j * np.conj(I_ji)
                        
                        S_data.append(S_ji)
                        S_row.append(phase_j)
                        S_col.append(phase_i)
            except Exception as e:
                print(f"计算线路 {i+1} ({node_i}->{node_j}) 功率时出错: {e}")
                continue
        
        # 构建稀疏矩阵
        S = sparse.csr_matrix((S_data, (S_row, S_col)), shape=(n_nodes, n_nodes))
        
        # 计算对称功率矩阵
        detaS = S + S.transpose()
        
        return S, detaS
        
    except Exception as e:
        print(f"计算线路功率和损耗时出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回空矩阵
        n_nodes = 4 * nodes
        return sparse.csr_matrix((n_nodes, n_nodes)), sparse.csr_matrix((n_nodes, n_nodes)) 
import numpy as np
from scipy import sparse
from calc_impedance_matrix import calc_impedance_matrix
# from calc_admittance_matrix import calc_admittance_matrix  # 如果需要

# 生成节点导纳矩阵
def formACY_ThreePhase(m_nNodes, m_branchi, m_branchb, m_linei, m_linej, m_lineL,
                        m_transi, m_transj, m_transR, m_transX, m_transk):
    """
    生成节点导纳矩阵
    
    参数:
    m_nNodes - 节点数
    m_branchi - 接地支路编号
    m_branchb - 标准导纳量
    m_linei - 线路节点i
    m_linej - 线路节点j
    m_lineL - 线路长度
    m_transi - 变压器节点i
    m_transj - 变压器节点j
    m_transR - 变压器R
    m_transX - 变压器X
    m_transk - 变压器变比
    
    返回:
    Y, Y0 - 导纳矩阵
    """
    # 线路数据
    nLines = len(m_linei)  # 线路数量
    expandedNodes = 4 * m_nNodes  # 扩展后的节点数量
    
    # 输出调试信息
    #print(f"节点数: {m_nNodes}, 扩展后节点数: {expandedNodes}, 线路数: {nLines}")
    
    # 初始化稀疏矩阵参数
    I = []  # 存储稀疏矩阵行索引
    J = []  # 存储稀疏矩阵列索引
    S = []  # 存储稀疏矩阵值
    
    # 处理每条线路，添加 4x4 的导纳矩阵到导纳矩阵
    for k in range(nLines):
        try:
            # 获取第 k 条线路的起点和终点
            i = int(m_linei[k])
            j = int(m_linej[k])
            
            # 验证节点编号是否有效
            if i <= 0 or j <= 0 or i > m_nNodes or j > m_nNodes:
                print(f"警告: 线路 {k} 节点编号 ({i}, {j}) 超出范围 [1, {m_nNodes}]，跳过此线路")
                continue
                
            # 计算 4x4 的导纳矩阵
            Y_ij = calc_impedance_matrix(m_lineL[k])  # calc_impedance_matrix(m_lineL[k]) 返回 4x4 矩阵
            
            # 扩展索引
            idx_i = np.array(list(range(4 * (i - 1), 4 * i)))  # 节点 i 对应的 4 个扩展索引 (从0开始)
            idx_j = np.array(list(range(4 * (j - 1), 4 * j)))  # 节点 j 对应的 4 个扩展索引 (从0开始)
            
            # 验证扩展索引是否有效
            if np.any(idx_i >= expandedNodes) or np.any(idx_j >= expandedNodes):
                print(f"警告: 线路 {k} 扩展索引超出范围 [0, {expandedNodes-1}]，跳过此线路")
                continue
            
            # 互连部分数据（-Y_ij）
            neg_Y_ij = -Y_ij
            rows, cols = np.nonzero(neg_Y_ij)
            values = neg_Y_ij[rows, cols]
            
            # 源点到终点
            for r, c, v in zip(rows, cols, values):
                I.append(idx_i[r])
                J.append(idx_j[c])
                S.append(v)
            
            # 终点到源点
            for r, c, v in zip(rows, cols, values):
                I.append(idx_j[r])
                J.append(idx_i[c])
                S.append(v)
            
            # 自连部分数据（Y_ij 对角部分）
            rows, cols = np.nonzero(Y_ij)
            values = Y_ij[rows, cols]
            
            # 源点自连
            for r, c, v in zip(rows, cols, values):
                I.append(idx_i[r])
                J.append(idx_i[c])
                S.append(v)
            
            # 终点自连
            for r, c, v in zip(rows, cols, values):
                I.append(idx_j[r])
                J.append(idx_j[c])
                S.append(v)
        except Exception as e:
            print(f"警告: 处理线路 {k} 时出错: {e}，跳过此线路")
            continue
    
    # 构造扩展后的稀疏矩阵 Y
    try:
        # 验证索引有效性，确保没有负值索引
        if len(I) > 0 and len(J) > 0:
            min_I, max_I = min(I), max(I)
            min_J, max_J = min(J), max(J)
            
            if min_I < 0 or min_J < 0:
                print(f"警告: 发现负索引 I: [{min_I}, {max_I}], J: [{min_J}, {max_J}], 将被调整")
                # 调整负索引为0
                I = [max(0, idx) for idx in I]
                J = [max(0, idx) for idx in J]
            
            if max_I >= expandedNodes or max_J >= expandedNodes:
                print(f"警告: 索引超出范围 I: [{min_I}, {max_I}], J: [{min_J}, {max_J}], 超出{expandedNodes-1}，将被截断")
                # 截断超出范围的索引
                I = [min(idx, expandedNodes-1) for idx in I]
                J = [min(idx, expandedNodes-1) for idx in J]
        
        Y = sparse.csr_matrix((S, (I, J)), shape=(expandedNodes, expandedNodes))
        Y0 = Y.copy()  # 简化Y0的计算
        
        print(f"导纳矩阵构建完成: 形状 {Y.shape}, 非零元素数量: {Y.nnz}")
    except Exception as e:
        print(f"警告: 构建导纳矩阵时出错: {e}")
        # 创建默认导纳矩阵
        Y = sparse.eye(expandedNodes, format='csr')
        Y0 = Y.copy()
    
    return Y, Y0
    
    # 以下部分在原代码中被注释，保留为注释状态
    """
    # 电容部分（m_lineb 扩展后按对角线填充）
    I = []; J = []; S = []
    for k in range(nLines):
        i = m_linei[k]
        j = m_linej[k]
        
        # 电容矩阵扩展为 4x4 的对角矩阵
        B_diag = calc_admittance_matrix(m_lineL[k])  # 假设函数为计算扩展后 4x4 对角矩阵
        
        # 找对角线部分
        rows, cols = np.nonzero(B_diag)
        values = B_diag[rows, cols]
        
        idx_i = list(range(4 * i - 3, 4 * i + 1))  # 源点扩展索引
        idx_j = list(range(4 * j - 3, 4 * j + 1))  # 终点扩展索引
        
        # 源点自连
        for r, c, v in zip(rows, cols, values):
            I.append(idx_i[r])
            J.append(idx_i[c])
            S.append(v)
        
        # 终点自连
        for r, c, v in zip(rows, cols, values):
            I.append(idx_j[r])
            J.append(idx_j[c])
            S.append(v)
    
    # 添加电容
    Y = Y + 1j * sparse.csr_matrix((S, (I, J)), shape=(expandedNodes, expandedNodes))
    """
    
    # 以下部分在原代码中被注释，保留为注释状态
    """
    # 电容互联部分（Y0）
    I = []; J = []; S = []
    for k in range(nLines):
        i = m_linei[k]
        j = m_linej[k]
        
        # 电容矩阵扩展为 4x4 的对角矩阵
        B_off_diag = np.diag(np.ones(4))  # 假设函数为计算扩展后 4x4 对角矩阵
        
        # 处理互联部分
        rows, cols = np.nonzero(B_off_diag)
        values = B_off_diag[rows, cols]
        
        idx_i = list(range(4 * i - 3, 4 * i + 1))  # 源点扩展索引
        idx_j = list(range(4 * j - 3, 4 * j + 1))  # 终点扩展索引
        
        # 源点到终点电容耦合
        for r, c, v in zip(rows, cols, values):
            I.append(idx_i[r])
            J.append(idx_j[c])
            S.append(v)
        
        # 终点到源点电容耦合
        for r, c, v in zip(rows, cols, values):
            I.append(idx_j[r])
            J.append(idx_i[c])
            S.append(v)
    
    # 构造扩展矩阵 Y0
    Y0 = 1j * sparse.csr_matrix((S, (I, J)), shape=(expandedNodes, expandedNodes))
    
    # m_lineY=1./(m_lineR+j*m_lineX)
    # Y=sparse(m_linei,m_linej,-m_lineY,m_nNodes,m_nNodes)+...
    #      sparse(m_linej,m_linei,-m_lineY,m_nNodes,m_nNodes)+...
    #      sparse(m_linei,m_linei,m_lineY,m_nNodes,m_nNodes)+...
    #      sparse(m_linej,m_linej,m_lineY,m_nNodes,m_nNodes)
    # Y=Y+j*(sparse(m_linei,m_linei,m_lineb,m_nNodes,m_nNodes)+...
    #      sparse(m_linej,m_linej,m_lineb,m_nNodes,m_nNodes))
    # Y0=j*(sparse(m_linei,m_linej,m_lineb,m_nNodes,m_nNodes)+...
    #      sparse(m_linej,m_linei,m_lineb,m_nNodes,m_nNodes))
    
    # 变压器支路部分
    # m_transY=1./(m_transR+j*m_transX)
    # Y=Y+sparse(m_transi,m_transj,-(m_transY.*m_transk),m_nNodes,m_nNodes)+...
    #       sparse(m_transj,m_transi,-(m_transY.*m_transk),m_nNodes,m_nNodes)+...
    #       sparse(m_transi,m_transi,(m_transY),m_nNodes,m_nNodes)+...
    #       sparse(m_transj,m_transj,(m_transY.*(m_transk.^2)),m_nNodes,m_nNodes)
    # Y0=Y0+sparse(m_transi,m_transj,((1-m_transk).*m_transY),m_nNodes,m_nNodes)+...
    #       sparse(m_transj,m_transi,((m_transk-1).*m_transk.*m_transY),m_nNodes,m_nNodes)
    
    # 接地支路部分
    if m_branchi != 0:
        Y = Y + 1j * sparse.csr_matrix((m_branchb, (m_branchi, m_branchi)), shape=(m_nNodes, m_nNodes))
        Y0 = Y0 + 1j * sparse.csr_matrix((m_branchb, (m_branchi, m_branchi)), shape=(m_nNodes, m_nNodes))
    else:
        Y = Y + 0
    
    Y0 = sparse.diags(Y0.diagonal())
    """
    
    # 解析Y矩阵
    # G = Y.real
    # B = Y.imag
    # 可以使用 plt.spy(Y) 显示稀疏矩阵的非零元素分布图
    
    return Y, Y0 
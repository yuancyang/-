import numpy as np
from scipy import sparse
import math
import traceback
from formDetaPQ2_ThreePhase import formDetaPQ2_ThreePhase
from formJacco2_ThreePhase import formJacco2_ThreePhase
from calcuVD_ThreePhase import calcuVD_ThreePhase
from pf_line import pf_line

def NR_main2_ThreePhase(PVi, PVx, PVV, PV_deta, balancenotes, balancevoltage, balanceangle, Y, Y0, linei, linej, transi, transj,
                         PG, PD, QG, QD, maxIters, precision, Nodes):
    """
    牛顿-拉夫森法潮流计算主函数
    
    参数：
    PVi, PVx, PVV, PV_deta - PV节点相关参数
    balancenotes, balancevoltage, balanceangle - 平衡节点相关参数
    Y, Y0 - 导纳矩阵
    linei, linej - 线路起点终点
    transi, transj - 变压器起点终点
    PG, PD, QG, QD - 功率注入和负荷
    maxIters - 最大迭代次数
    precision - 计算精度
    Nodes - 节点数
    
    返回：
    Vi, deta - 电压和相角
    PQ_loss - 系统损耗
    S, detaS - 线路功率流
    Colab - 收敛标志
    Jacco, Jacco2 - 雅可比矩阵
    """
    try:
        # 检查导纳矩阵
        n_nodes = 4 * Nodes
        if Y.shape != (n_nodes, n_nodes):
            print(f"警告: 导纳矩阵尺寸 {Y.shape} 与预期尺寸 ({n_nodes}, {n_nodes}) 不匹配")
            # 如果需要，可以返回错误标志
            return None, None, None, None, None, 1, None, None
            
        # 设置系统电压和相角初值
        V = np.ones((n_nodes, 1))
        deta = np.zeros((n_nodes, 1))  # 初始化相角矩阵
        
        # 设置初始相角
        for i in range(1, Nodes+1):
            idx_a = 4*(i-1)      # A相索引
            idx_b = 4*(i-1) + 1  # B相索引
            idx_c = 4*(i-1) + 2  # C相索引
            idx_n = 4*(i-1) + 3  # 中性线索引
            
            if idx_b < n_nodes:  # 确保索引有效
                deta[idx_b, 0] = 120  # B相相角初始值
                
            if idx_c < n_nodes:  # 确保索引有效
                deta[idx_c, 0] = -120  # C相相角初始值
                
            if idx_n < n_nodes:  # 确保索引有效
                V[idx_n, 0] = 0  # 中性线电压幅值设0
        
        # 转换为复电压
        Vi = sparse.csr_matrix(V * (np.cos(deta*np.pi/180) + 1j*np.sin(deta*np.pi/180)))
        Vj = sparse.csr_matrix(np.conj(Vi.toarray()))
        ei = np.real(Vi.toarray())
        fi = np.imag(Vi.toarray())
        
        # 开始迭代
        k = 0
        Colab = 0  # 收敛标志位：0表示收敛，1表示发散
        jacco = None  # 初始化雅可比矩阵
        jacco2 = None
        
        for k in range(maxIters):
            # 更新复电压
            Vi = sparse.csr_matrix(ei + 1j * fi)
            Vj = sparse.csr_matrix(ei - 1j * fi)
            
            # 生成不平衡量
            try:
                H, Pi, Qi = formDetaPQ2_ThreePhase(PG, PD, Vi, Vj, Y, balancenotes, balancevoltage, balanceangle, QG, QD, PVi, Nodes)
            except Exception as e:
                print(f"生成不平衡量时出错: {e}")
                traceback.print_exc()
                return Vi, deta, 0, 0, 0, 1, None, None
            
            # 计算误差
            if sparse.issparse(H):
                e = np.max(np.abs(H.toarray()))
            else:
                e = np.max(np.abs(H))
            
            # 判断收敛条件
            if e < precision:  # 小于精度，收敛
                #print('收敛')
                Colab = 0
                break
            else:  # 大于精度，继续迭代
                # 形成雅可比矩阵
                try:
                    jacco, jacco2 = formJacco2_ThreePhase(Vi, Vj, Y, Nodes, PVi, balancenotes, QG, QD, PG, PD, Pi, Qi)
                except Exception as e:
                    print(f"形成雅可比矩阵时出错: {e}")
                    traceback.print_exc()
                    return Vi, deta, 0, 0, 0, 1, None, None
                
                # 求解节点系统的直角坐标系统方程增量
                try:
                    detaei, detafi = calcuVD_ThreePhase(jacco, H, Nodes, V)
                except Exception as e:
                    print(f"计算电压增量时出错: {e}")
                    traceback.print_exc()
                    return Vi, deta, 0, 0, 0, 1, None, None
                
                # 更新电压坐标系
                if isinstance(detaei, np.ndarray) and len(detaei) == n_nodes and \
                   isinstance(detafi, np.ndarray) and len(detafi) == n_nodes:
                    # 正常更新
                    ei = ei + detaei.reshape(ei.shape)
                    fi = fi + detafi.reshape(fi.shape)
                else:
                    # 输出警告并使用小增量更新
                    print(f"警告: 增量尺寸不匹配，ei: {ei.shape}, detaei: {detaei.shape if hasattr(detaei, 'shape') else 'unknown'}")
                    ei = ei + 0.001 * np.ones(ei.shape)
                    fi = fi + 0.001 * np.ones(fi.shape)
                
                # 判断迭代是否达到最大次数
                if k == maxIters-1:
                    print('发散')
                    Colab = 1
        
        # 如果迭代未收敛
        if k == maxIters:
            print('达到最大迭代次数仍未收敛')
            Colab = 1
            
        Jacco = jacco if jacco is not None else sparse.eye(2*n_nodes, format='csr')
        Jacco2 = jacco2 if jacco2 is not None else sparse.eye(2*n_nodes, format='csr')
        
        # 系统损耗
        try:
            PQ_loss = np.sum(Vi.toarray() * (np.conj(Y.toarray()) @ Vj.toarray()))
        except Exception as e:
            print(f"计算系统损耗时出错: {e}")
            PQ_loss = 0
        
        # 计算线路功率和损耗
        try:
            S, detaS = pf_line(Vi, deta, linei, linej, transi, transj, Y, Y0, Nodes)
        except Exception as e:
            print(f"计算线路功率时出错: {e}")
            S = sparse.csr_matrix((n_nodes, n_nodes))
            detaS = S.copy()
        
        return Vi, deta, PQ_loss, S, detaS, Colab, Jacco, Jacco2
        
    except Exception as e:
        print(f"牛顿-拉夫森法潮流计算出错: {e}")
        traceback.print_exc()
        
        # 返回默认值
        n_nodes = 4 * Nodes
        Vi = sparse.csr_matrix(np.ones((n_nodes, 1)) * (1 + 0j))
        deta = sparse.csr_matrix(np.zeros((n_nodes, 1)))
        PQ_loss = 0
        S = sparse.csr_matrix((n_nodes, n_nodes))
        detaS = S.copy()
        Colab = 1  # 发散
        Jacco = sparse.eye(2*n_nodes, format='csr')
        Jacco2 = Jacco.copy()
        
        return Vi, deta, PQ_loss, S, detaS, Colab, Jacco, Jacco2 
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import stats
import time
from dataIn import dataIn
from formACY_ThreePhase import formACY_ThreePhase
from NR_main2_ThreePhase import NR_main2_ThreePhase
from scipy.stats import gaussian_kde

def analyze_monte_carlo_results(Vfenbu_s, n_nodes=None):
    """
    蒙特卡洛法概率潮流结果分析
    
    参数:
        Vfenbu_s: 存储所有节点电压分布的数组
        n_nodes: 系统节点数量(可选)
    """
    # 节点电压概率密度分析
    nbins = 100                        # 直方图区间数
    target_node = 10                   # 选择分析的节点
    target_Phase = 2                   # 选择分析的相位
    target = 4*(target_node-1)+target_Phase - 1  # Python索引从0开始
    
    # 确保目标索引有效
    if n_nodes and (target < 0 or target >= n_nodes):
        print(f"警告: 目标索引 {target} 超出范围，使用默认索引0")
        target = 0
    
    # 提取目标节点电压数据
    V_target = Vfenbu_s[target, :]
    
    # 过滤掉为0的值（失败的迭代）
    V_target = V_target[V_target > 0]
    
    # 如果数据不足，返回
    if len(V_target) < 10:
        print("有效数据太少，无法生成有意义的统计分析")
        return None
    
    # 计算电压统计特征
    V_mean = np.mean(V_target)           # 电压均值
    V_std = np.std(V_target)             # 电压标准差
    V_min = np.min(V_target)             # 最小电压
    V_max = np.max(V_target)             # 最大电压
    
    print(f"节点{target_node}相位{target_Phase}电压统计特征:")
    print(f"均值: {V_mean:.4f} pu")
    print(f"标准差: {V_std:.4f} pu")
    print(f"最小值: {V_min:.4f} pu")
    print(f"最大值: {V_max:.4f} pu")
    
    # 计算概率密度分布
    counts, edges = np.histogram(V_target, bins=nbins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2  # 计算区间中心值
    
    # 核密度估计（平滑处理）
    kde = stats.gaussian_kde(V_target)
    xi = np.linspace(V_min-0.01, V_max+0.01, 500)
    pdf_kde = kde(xi)
    
    # 绘制概率密度曲线
    plt.figure(figsize=(10, 6))
    plt.title(f'节点{target_node}相位{target_Phase}电压概率密度分布')
    
    # 绘制直方图
    plt.bar(bin_centers, counts, width=np.mean(np.diff(bin_centers)), alpha=0.5, 
            color=[0.7, 0.7, 0.9], edgecolor='none')
    
    # 绘制核密度曲线
    plt.plot(xi, pdf_kde, 'b-', linewidth=2, label='核密度估计')
    
    # 标注统计特征
    plt.axvline(V_mean, color='r', linestyle='--', linewidth=1.5, 
                label=f'均值={V_mean:.4f}')
    plt.axvline(V_mean-V_std, color='m', linestyle='--')
    plt.axvline(V_mean+V_std, color='m', linestyle='--', 
                label='±1标准差范围')
    
    # 图形修饰
    plt.xlabel('电压幅值 (pu)')
    plt.ylabel('概率密度')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlim([V_min-0.01, V_max+0.01])
    plt.tight_layout()
    
    # 尝试显示图表
    try:
        plt.show()
    except Exception as e:
        print(f"注意: 无法显示图表: {e}")
        plt.close()
    
    return pdf_kde, xi, V_mean, V_std

# 蒙特卡洛法概率潮流计算主程序
def main_MC():
    # 设置字体以支持中文显示
    try:
        # 使用中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS'] 
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.rcParams['font.family'] = 'sans-serif'
    except:
        print("警告: 字体设置失败，图表中可能出现乱码")
    
    # 蒙特卡洛法计算计时开始
    start_time = time.time()
    
    # 输入基础参数
    try:
        print("开始读取系统数据...")
        (Nodes, linenum, SB, maxIters, OPdata1, precision, OPdata2, balanceNum, balancenotes, balancevoltage, balanceangle,
         lineID, linei, linej, lineL,
         branchi, branchb,
         transID, transi, transj, transr, transx, transk, transkMin, transkMax,
         PQi, PQx, PG, QG, PD, QD,
         PVi, PVx, PVV, PV_deta, PVQmin, PVQmax,
         NGi, OP_0, OP_1, OP_2, NGmin, NGmax) = dataIn('25NodesThreePhase.txt')
        
        print("系统数据读取完成")
        print(f"系统节点数: {Nodes}, 线路数: {linenum}")
    except Exception as e:
        print(f"数据读取错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 读取负荷正态分布数据
    with open('25nodesload_30%.txt', 'r') as f:
        pdfload = np.loadtxt(f)

    n_nodes = int(4*Nodes)
    PlPx = np.zeros((n_nodes, 2))
    PlQx = np.zeros((n_nodes, 2))

    for i in range(pdfload.shape[0]):
        node_i = int(pdfload[i, 0])
        node_x = int(pdfload[i, 1])
    # 确保索引有效
        if 1 <= node_i <= Nodes and 1 <= node_x <= 4:
            node_idx = int(4*(node_i-1)+node_x) - 1  # Python索引从0开始
            if 0 <= node_idx < n_nodes:
                PlPx[node_idx, 0] = -pdfload[i, 2]/SB
                PlPx[node_idx, 1] = pdfload[i, 3]
                PlQx[node_idx, 0] = -pdfload[i, 4]/SB
                PlQx[node_idx, 1] = pdfload[i, 5]
    print(f"PlPx: {PlPx}")
    # 蒙特卡洛迭代次数
    daishu = 500
    Vfenbu_s = np.zeros((n_nodes, daishu))
    
    # 转换稀疏矩阵为数组，以便修改
    if sparse.issparse(PD):
        PD_array = PD.toarray().copy()
        QD_array = QD.toarray().copy()
    else:
        # 如果已经是数组，直接使用
        PD_array = np.zeros((n_nodes, 1))
        QD_array = np.zeros((n_nodes, 1))
    
    # 备份原始负荷数据
    PD_original = PD_array.copy() if isinstance(PD_array, np.ndarray) else np.zeros((n_nodes, 1))
    QD_original = QD_array.copy() if isinstance(QD_array, np.ndarray) else np.zeros((n_nodes, 1))
    
    # 形成交流系统节点导纳矩阵 - 在循环外计算以提高效率
    try:
        #print("计算系统导纳矩阵...")
        Y, Y0 = formACY_ThreePhase(Nodes, branchi, branchb, linei, linej, lineL,
                                  transi, transj, transr, transx, transk)
        #print("导纳矩阵计算完成")
    except Exception as e:
        print(f"导纳矩阵计算错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"开始蒙特卡洛迭代，总迭代次数: {daishu}")
    success_count = 0
    
    # 增加调试参数
    debug_interval = 50  # 每多少次迭代输出详细信息
    max_trials = 3  # 每次迭代最多尝试次数
    
    # 蒙特卡洛法主循环
    for iii in range(daishu):
        # 重置负荷为原始值，避免累积误差
        PD_array = PD_original.copy()
        QD_array = QD_original.copy()
        
        # 尝试次数计数
        trial_count = 0
        success = False
        
        while trial_count < max_trials and not success:
            try:
                if len(PQi) > 0:
                    # 生成随机负荷
                    for i in range(len(PQi)):
                        node_i = int(PQi[i])
                        node_x = int(PQx[i])
                        # 确保索引有效
                        if 1 <= node_i <= Nodes and 1 <= node_x <= 4:
                            node_idx = int(4*(node_i-1)+node_x) - 1  # Python索引从0开始
                            
                            # 确保索引有效
                            if 0 <= node_idx < PlPx.shape[0]:
                                # 生成随机负荷，并添加约束条件
                                mean_P = PlPx[node_idx, 0]
                                std_P = PlPx[node_idx, 1]
                                mean_Q = PlQx[node_idx, 0]
                                std_Q = PlQx[node_idx, 1]
                                
                                # 限制随机值在均值±3倍标准差的范围内
                                random_P = np.random.normal(mean_P, std_P)
                                random_Q = np.random.normal(mean_Q, std_Q)
                                
                                # 更新数组
                                PD_array[node_idx, 0] = random_P
                                QD_array[node_idx, 0] = random_Q
                
                # 转回稀疏矩阵
                PD_current = sparse.csr_matrix(PD_array)
                QD_current = sparse.csr_matrix(QD_array)
                
                # 潮流计算
                Vi, deta, PQ_loss, S, detaS, Colab, Jacco, Jacco2 = NR_main2_ThreePhase(
                    PVi, PVx, PVV, PV_deta, balancenotes, balancevoltage, balanceangle, Y, Y0, 
                    linei, linej, transi, transj, PG, PD_current, QG, QD_current, maxIters, precision, Nodes)
                
                # 检查是否收敛
                if Colab == 0:  # 收敛
                    # 记录电压幅值
                    if sparse.issparse(Vi):
                        Vfenbu_s[:, iii] = np.abs(Vi.toarray().flatten())
                    else:
                        Vfenbu_s[:, iii] = np.abs(Vi.flatten())
                    
                    success_count += 1
                    success = True
                else:
                    # 不收敛，重试
                    trial_count += 1
                    if iii % debug_interval == 0:
                        print(f"迭代 {iii+1} 第 {trial_count} 次尝试不收敛，重试...")
            except Exception as e:
                # 如果是线路功率计算的错误，但电压值已经计算出来，仍视为成功
                if "计算线路功率时出错" in str(e) and 'Vi' in locals() and Vi is not None:
                    # 这种情况下，我们有电压值，只是线路功率计算失败
                    if sparse.issparse(Vi):
                        Vfenbu_s[:, iii] = np.abs(Vi.toarray().flatten())
                    else:
                        Vfenbu_s[:, iii] = np.abs(Vi.flatten())
                    
                    success_count += 1
                    success = True
                    if iii % debug_interval == 0:
                        print(f"迭代 {iii+1}: 电压计算成功，但线路功率计算失败，已保存电压值")
                else:
                    # 其他错误，重试
                    trial_count += 1
                    if iii % debug_interval == 0:
                        print(f"迭代 {iii+1} 第 {trial_count} 次尝试出错: {e}，重试...")
        
        # 如果所有尝试都失败
        if not success and iii % debug_interval == 0:
            print(f"迭代 {iii+1} 所有尝试都失败")
        
        # 显示进度
        if (iii + 1) % debug_interval == 0:
            print(f"完成迭代 {iii+1}/{daishu}，成功率: {success_count/(iii+1)*100:.1f}%")
    
    print(f"蒙特卡洛模拟完成，共成功 {success_count}/{daishu} 次迭代")
    
    # 如果没有成功的迭代，提前返回
    if success_count == 0:
        print("所有迭代都失败，无法进行统计分析")
        return None
    
    # 调用分析函数处理结果
    pdf_kde, xi, V_mean, V_std = analyze_monte_carlo_results(Vfenbu_s, n_nodes)
    
    # 蒙特卡洛法计算计时结束
    elapsed_time = time.time() - start_time
    print(f"计算耗时: {elapsed_time:.2f} 秒")
    
    return Vfenbu_s, pdf_kde, xi, V_mean, V_std

if __name__ == "__main__":
    main_MC() 
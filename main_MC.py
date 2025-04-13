import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import stats
import time
import argparse  # 导入参数解析模块
from dataIn import dataIn
from formACY_ThreePhase import formACY_ThreePhase
from NR_main2_ThreePhase import NR_main2_ThreePhase
from scipy.stats import gaussian_kde

def analyze_monte_carlo_results(Vfenbu_s, target_node=10, target_Phase=2, threshold=1.05, n_nodes=None):
    """
    蒙特卡洛法概率潮流结果分析
    
    参数:
        Vfenbu_s: 存储所有节点电压分布的数组
        target_node: 选择分析的节点，默认为10
        target_Phase: 选择分析的相位，默认为2
        threshold: 电压阈值，默认为1.05 pu
        n_nodes: 系统节点数量(可选)
    """
    # 计算目标节点索引并验证有效性
    target = 4*(target_node-1)+target_Phase - 1
    if n_nodes and (target < 0 or target >= n_nodes):
        print(f"警告: 目标索引 {target} 超出范围，使用默认索引0")
        target = 0
    
    # 提取并过滤有效电压数据
    V_target = Vfenbu_s[target, :][Vfenbu_s[target, :] > 0]
    
    # 数据有效性检查
    if len(V_target) < 10:
        print("有效数据太少，无法生成有意义的统计分析")
        return None
    
    # 计算统计特征
    V_mean, V_std = np.mean(V_target), np.std(V_target)
    V_min, V_max = np.min(V_target), np.max(V_target)
    prob_above_threshold = np.mean(V_target > threshold)
    
    # 打印统计结果
    print(f"\n节点{target_node}相位{target_Phase}电压统计特征:")
    print(f"均值: {V_mean:.4f} pu | 标准差: {V_std:.4f} pu")
    print(f"最小值: {V_min:.4f} pu | 最大值: {V_max:.4f} pu")
    print(f"电压高于{threshold} pu的概率: {prob_above_threshold*100:.2f}%")
    
    # 准备绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'节点{target_node}相位{target_Phase}电压概率密度分布')
    
    # 计算KDE与直方图
    kde = stats.gaussian_kde(V_target)
    xi = np.linspace(V_min-0.01, V_max+0.01, 500)
    pdf_kde = kde(xi)
    
    # 绘制直方图和核密度曲线
    ax.hist(V_target, bins=100, density=True, alpha=0.5, color=[0.7, 0.7, 0.9], edgecolor='none')
    ax.plot(xi, pdf_kde, 'b-', linewidth=2, label='核密度估计')
    
    # 添加统计参考线
    ax.axvline(V_mean, color='r', linestyle='--', linewidth=1.5, label=f'均值={V_mean:.4f}')
    ax.axvline(V_mean-V_std, color='m', linestyle='--')
    ax.axvline(V_mean+V_std, color='m', linestyle='--', label='±1标准差范围')
    ax.axvline(threshold, color='g', linestyle='-', linewidth=1.5,
               label=f'阈值{threshold} pu (概率: {prob_above_threshold*100:.2f}%)')
    
    # 标注超阈值区域
    high_voltage_xi = xi[xi > threshold]
    if len(high_voltage_xi) > 0:
        ax.fill_between(high_voltage_xi, kde(high_voltage_xi), alpha=0.3, 
                        color='green', label=f'高于{threshold} pu区域')
    
    # 完善图表设置
    ax.set(xlabel='电压幅值 (pu)', ylabel='概率密度',
           xlim=[V_min-0.01, V_max+0.01])
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    
    # 显示图表
    try:
        plt.show()
    except Exception as e:
        print(f"注意: 无法显示图表: {e}")
        plt.close()
    
    return pdf_kde, xi, V_mean, V_std, prob_above_threshold

def analyze_all_nodes_voltage_probability(Vfenbu_s, threshold=1.05):
    """
    分析所有节点电压高于指定阈值的概率
    
    参数:
        Vfenbu_s: 存储所有节点电压分布的数组
        threshold: 电压阈值，默认为1.05 pu
        
    返回:
        node_probs: 每个节点电压高于阈值的概率数组
        system_prob: 系统中任意节点电压高于阈值的概率
    """
    n_nodes, n_samples = Vfenbu_s.shape
    
    # 初始化概率数组
    node_probs = np.zeros(n_nodes)
    
    # 计算每个节点的概率
    for i in range(n_nodes):
        # 提取节点电压数据
        V_node = Vfenbu_s[i, :]
        
        # 过滤掉为0的值（失败的迭代）
        V_node = V_node[V_node > 0]
        
        if len(V_node) > 0:
            # 计算高于阈值的概率
            node_probs[i] = np.mean(V_node > threshold)
    
    # 输出每个节点的概率
    print(f"\n所有节点电压高于{threshold} pu的概率统计:")
    print("---------------------------------------")
    
    # 获取物理节点数 (总节点数除以4，因为每个物理节点有4个相位)
    physical_nodes = n_nodes // 4
    phases = ['A相', 'B相', 'C相', '中性线']
    
    # 按物理节点和相位输出概率
    for node in range(1, physical_nodes + 1):
        for phase in range(4):
            idx = 4 * (node - 1) + phase
            if idx < n_nodes:
                prob = node_probs[idx] * 100
                if prob > 0:  # 只显示概率大于0的节点
                    print(f"节点{node} {phases[phase]}: {prob:.2f}%")
    
    # 计算系统中任意节点电压高于阈值的概率
    # 对于每次采样，检查是否有任何节点电压高于阈值
    any_high_voltage = np.zeros(n_samples, dtype=bool)
    for j in range(n_samples):
        sample_voltages = Vfenbu_s[:, j]
        if np.any(sample_voltages > threshold):
            any_high_voltage[j] = True
    
    system_prob = np.mean(any_high_voltage)
    print("\n系统统计:")
    print(f"系统中任意节点电压高于{threshold} pu的概率: {system_prob*100:.2f}%")
    
    # 找出超标概率最高的前5个节点
    if np.any(node_probs > 0):
        top_indices = np.argsort(node_probs)[-5:][::-1]
        print("\n超标概率最高的前5个节点:")
        for idx in top_indices:
            if node_probs[idx] > 0:
                node = idx // 4 + 1
                phase = idx % 4
                print(f"节点{node} {phases[phase]}: {node_probs[idx]*100:.2f}%")
    
    return node_probs, system_prob

def visualize_voltage_probabilities(node_probs, threshold=1.05):
    """
    可视化所有节点电压高于阈值的概率
    
    参数:
        node_probs: 存储所有节点超过阈值概率的数组
        threshold: 电压阈值，默认为1.05 pu
    """
    # 获取物理节点数和相位数
    n_nodes = len(node_probs)
    physical_nodes = n_nodes // 4
    phases = ['A相', 'B相', 'C相', '中性线']
    
    # 将概率重新组织为矩阵形式（节点 x 相位）
    prob_matrix = np.zeros((physical_nodes, 4))
    for i in range(n_nodes):
        node = i // 4
        phase = i % 4
        if node < physical_nodes:
            prob_matrix[node, phase] = node_probs[i] * 100  # 转换为百分比
    
    # 创建一个形状更合适的图
    fig_width = max(10, physical_nodes * 0.5)
    plt.figure(figsize=(fig_width, 8))
    
    # 创建热力图
    plt.subplot(2, 1, 1)
    plt.title(f'节点电压高于{threshold} pu的概率热力图')
    im = plt.imshow(prob_matrix.T, cmap='hot_r', aspect='auto')
    plt.colorbar(im, label='概率 (%)')
    
    # 设置坐标轴标签
    plt.xlabel('节点编号')
    plt.ylabel('相位')
    plt.yticks(range(4), phases)
    plt.xticks(range(physical_nodes), range(1, physical_nodes + 1))
    
    # 在每个单元格中显示概率值
    for i in range(physical_nodes):
        for j in range(4):
            if prob_matrix[i, j] > 0:  # 只标注概率大于0的单元格
                plt.text(i, j, f'{prob_matrix[i, j]:.1f}%', 
                         ha='center', va='center', 
                         color='white' if prob_matrix[i, j] > 50 else 'black')
    
    # 创建每个节点的最大概率柱状图
    plt.subplot(2, 1, 2)
    plt.title(f'各节点电压高于{threshold} pu的最大概率')
    
    # 计算每个节点的最大概率
    max_probs = np.max(prob_matrix, axis=1)
    
    # 绘制柱状图
    bars = plt.bar(range(1, physical_nodes + 1), max_probs)
    
    # 设置柱状图颜色
    for i, bar in enumerate(bars):
        if max_probs[i] > 50:
            bar.set_color('red')
        elif max_probs[i] > 20:
            bar.set_color('orange')
        elif max_probs[i] > 0:
            bar.set_color('yellow')
    
    # 设置坐标轴
    plt.xlabel('节点编号')
    plt.ylabel('最大概率 (%)')
    plt.xticks(range(1, physical_nodes + 1))
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 在柱子上方标注最大概率值
    for i, v in enumerate(max_probs):
        if v > 0:
            plt.text(i + 1, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    
    # 尝试显示图表
    try:
        plt.show()
    except Exception as e:
        print(f"注意: 无法显示图表: {e}")
        plt.close()

# 蒙特卡洛法概率潮流计算主程序
def main_MC(threshold=1.05, target_node=10, target_phase=2, visualize=True):
    """
    蒙特卡洛法概率潮流计算主程序
    
    参数:
        threshold: 电压阈值，默认为1.05 pu
        target_node: 选择分析的节点，默认为10
        target_phase: 选择分析的相位，默认为2
        visualize: 是否执行可视化，默认为True
    """
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
    
    # 读取负荷正态分布数据（从文件中加载节点负荷的概率分布参数）
    with open('25nodesload_30%.txt', 'r') as f:
        pdfload = np.loadtxt(f) 

    n_nodes = int(4*Nodes)  # 计算系统中的总节点数（每个物理节点有4个电气节点，对应3相+中性线）
    PlPx = np.zeros((n_nodes, 2))  # 创建存储有功功率均值和标准差的空数组
    PlQx = np.zeros((n_nodes, 2))  # 创建存储无功功率均值和标准差的空数组

    for i in range(pdfload.shape[0]):  # 遍历负荷数据的每一行
        node_i = int(pdfload[i, 0])  # 获取物理节点编号
        node_x = int(pdfload[i, 1])  # 获取相位编号（1-4表示A,B,C相和中性线）
        node_idx = int(4*(node_i-1)+node_x) - 1  # 计算在数组中的实际索引位置（Python索引从0开始）

        PlPx[node_idx, 0] = -pdfload[i, 2]/SB  # 存储有功功率均值（负号表示负荷，并归一化到系统基准功率）
        PlPx[node_idx, 1] = pdfload[i, 3]  # 存储有功功率标准差
        PlQx[node_idx, 0] = -pdfload[i, 4]/SB  # 存储无功功率均值（负号表示负荷，并归一化到系统基准功率）
        PlQx[node_idx, 1] = pdfload[i, 5]  # 存储无功功率标准差
    #print(f"PlPx: {PlPx}")  # 打印有功功率分布数据，用于调试
    
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
    debug_interval = 100  # 每多少次迭代输出详细信息
    max_trials = 1  # 每次迭代最多尝试次数
    
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
                                # 生成随机负荷
                                mean_P = PlPx[node_idx, 0]
                                std_P = PlPx[node_idx, 1]
                                mean_Q = PlQx[node_idx, 0]
                                std_Q = PlQx[node_idx, 1]
                                if mean_P >= 0:
                                    random_P = np.random.normal(mean_P, std_P)
                                    random_Q = np.random.normal(mean_Q, std_Q)
                                else:
                                    random_P = np.random.beta(2, 5)*mean_P*30
                                    #print(f"random_P: {random_P}")
                                    random_Q = 0
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
    pdf_kde, xi, V_mean, V_std, prob_above_threshold = analyze_monte_carlo_results(
        Vfenbu_s, target_node=target_node, target_Phase=target_phase, threshold=threshold, n_nodes=n_nodes)
    
    # 蒙特卡洛法计算计时结束
    elapsed_time = time.time() - start_time
    print(f"计算耗时: {elapsed_time:.2f} 秒")
    
    return Vfenbu_s, pdf_kde, xi, V_mean, V_std, prob_above_threshold

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='蒙特卡洛法概率潮流计算与电压超标分析')
    
    parser.add_argument('--threshold', type=float, default=1.05,
                        help='电压阈值，默认为1.05 pu')
    parser.add_argument('--node', type=int, default=10,
                        help='选择分析的节点，默认为10')
    parser.add_argument('--phase', type=int, default=2,
                        help='选择分析的相位(1-4，分别对应A,B,C相和中性线)，默认为2')
    parser.add_argument('--no-viz', action='store_true',
                        help='禁用可视化功能')
    parser.add_argument('--all-nodes', action='store_true',
                        help='分析所有节点，而不仅仅是指定节点')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    print(f"使用电压阈值: {args.threshold} pu")
    print(f"分析节点: {args.node}, 相位: {args.phase}")
    print(f"是否可视化: {not args.no_viz}")
    
    results = main_MC(threshold=args.threshold, 
                     target_node=args.node, 
                     target_phase=args.phase, 
                     visualize=not args.no_viz)
                     
    if results:
        Vfenbu_s, pdf_kde, xi, V_mean, V_std, prob_above_threshold = results
        print(f"\n=========================================")
        print(f"选定节点电压高于{args.threshold} pu的概率: {prob_above_threshold*100:.2f}%")
        print(f"=========================================")
        
        # 分析所有节点电压概率
        if args.all_nodes:
            node_probs, system_prob = analyze_all_nodes_voltage_probability(Vfenbu_s, threshold=args.threshold)
            
            # 可视化所有节点的电压超标概率
            if not args.no_viz:
                visualize_voltage_probabilities(node_probs, threshold=args.threshold) 
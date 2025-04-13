import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_monte_carlo_results(Vfenbu_s):
    """
    蒙特卡洛法概率潮流结果分析
    
    参数:
        Vfenbu_s: 存储所有节点电压分布的数组
    """
    # 节点电压概率密度分析
    nbins = 100                        # 直方图区间数
    target_node = 10                   # 选择分析的节点
    target_Phase = 2
    target = 4*(target_node-1)+target_Phase

    # 提取目标节点电压数据
    V_target = Vfenbu_s[target, :]

    # 计算电压统计特征
    V_mean = np.mean(V_target)         # 电压均值
    V_std = np.std(V_target)           # 电压标准差
    V_min = np.min(V_target)           # 最小电压
    V_max = np.max(V_target)           # 最大电压

    # 计算概率密度分布
    counts, edges = np.histogram(V_target, bins=nbins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2  # 计算区间中心值

    # 核密度估计（平滑处理）
    kde = stats.gaussian_kde(V_target)
    xi = np.linspace(V_min - 0.01, V_max + 0.01, 500)
    pdf_kde = kde(xi)

    # 绘制概率密度曲线
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.bar(bin_centers, counts, width=np.mean(np.diff(bin_centers)),
            color=[0.7, 0.7, 0.9], alpha=0.5, edgecolor='none')

    # 绘制核密度曲线
    plt.plot(xi, pdf_kde, 'b-', linewidth=2, label='核密度估计')

    # 标注统计特征
    plt.axvline(x=V_mean, color='r', linestyle='--', linewidth=1.5,
                label=f'均值={V_mean:.4f}')
    plt.axvline(x=V_mean-V_std, color='m', linestyle='--',
                label='±1标准差范围')
    plt.axvline(x=V_mean+V_std, color='m', linestyle='--')

    # 图形修饰
    plt.title(f'节点{target_node}电压概率密度分布')
    plt.xlabel('电压幅值 (pu)')
    plt.ylabel('概率密度')
    plt.legend(loc='northwest')
    plt.grid(True)
    plt.xlim([V_min-0.01, V_max+0.01])
    
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 假设Vfenbu_s是来自蒙特卡洛模拟的电压分布数据
    # 这里创建一个示例数据, 实际使用时替换为真实数据
    np.random.seed(42)  # 设置随机种子以确保可重复性
    num_nodes = 20      # 总节点数
    num_phases = 4      # 每个节点的相数
    num_samples = 1000  # 蒙特卡洛模拟次数
    
    # 创建模拟的电压分布数据 (正态分布, 均值1.0, 标准差0.05)
    Vfenbu_s = np.random.normal(1.0, 0.05, (num_nodes * num_phases, num_samples))
    
    # 分析结果
    analyze_monte_carlo_results(Vfenbu_s) 
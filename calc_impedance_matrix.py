import numpy as np

def calc_impedance_matrix(length):
    """
    计算三相四线制架空线的导纳矩阵
    
    参数说明:
      length    - 线路长度 (km)
      r         - 单位电阻 (Ω/km)
      x_self    - 单位自感抗 (Ω/km)
      d_ij      - 相间距离 (m)，数组 [d_AB, d_BC, d_CA]
      d_neutral - 相对中性线的距离 (m)，数组 [d_A_N, d_B_N, d_C_N]
    
    返回值:
      Y_matrix  - 三相四线制导纳矩阵 (单位: Ω)
    """
    # 创建阻抗矩阵 zmatrix - 与MATLAB代码完全一致的表示方式
    zmatrix = length*(1/0.00529)*np.array([
        [0.01273+0.002933j, 0, 0, 0],
        [0, 0.01273+0.002933j, 0, 0],
        [0, 0, 0.01273+0.002933j, 0],
        [0, 0, 0, 0.01273+0.002933j]
    ])
    
    # 计算导纳矩阵 Y_matrix，使用伪逆以增强数值稳定性
    try:
        Y_matrix = np.linalg.inv(zmatrix)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异或接近奇异，使用伪逆
        print(f"警告: 线路长度 {length} 导致阻抗矩阵接近奇异，使用伪逆代替")
        Y_matrix = np.linalg.pinv(zmatrix)
    
    return Y_matrix 
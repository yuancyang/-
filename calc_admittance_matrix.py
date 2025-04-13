import numpy as np

def calc_admittance_matrix(length):
    """
    计算三相四线制架空线的对地导纳矩阵（对角形式）
    
    参数说明:
      length    - 线路长度 (km)
      c_self    - 单位自电纳 (μS/km)
      e_gnd     - 对地距离 (m)，数组 [e_A, e_B, e_C, e_N]
    
    返回值:
      y_matrix  - 三相四线制对地导纳对角矩阵 (单位: S)
    """
    # 创建阻抗矩阵 zmatrix - 与MATLAB代码完全一致的表示方式
    zmatrix = length*(1/0.00529)*np.array([
        [-2498.9j, 0, 0, 0],
        [0, -2498.9j, 0, 0],
        [0, 0, -2498.9j, 0],
        [0, 0, 0, -2498.9j]
    ])
    
    # 计算导纳矩阵 y_matrix
    y_matrix = np.linalg.inv(zmatrix)
    
    return y_matrix 
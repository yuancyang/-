import numpy as np
from scipy import sparse

def dataIn(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = []
    max_cols = 0
    for line in lines:
        row = [float(x) for x in line.strip().split() if x]
        if row:
            data.append(row)
            max_cols = max(max_cols, len(row))
    
    A = np.zeros((len(data), max_cols))
    for i, row in enumerate(data):
        A[i, :len(row)] = row
    
    Nodes = int(A[0, 0])
    linenum = int(A[0, 1])
    SB = A[0, 2]
    maxIters = int(A[0, 3])
    OPdata1 = A[0, 4] if A.shape[1] > 4 else 0
    
    precision = A[1, 0]
    OPdata2 = A[1, 1] if A.shape[1] > 1 else 0
    
    balanceNum = int(A[2, 0])
    balancenotes = A[3, 0:balanceNum]
    balancevoltage = A[4, 0:balanceNum]
    balanceangle = A[4, balanceNum:2*balanceNum]
    
    t = np.where(A[:, 0] == 0)[0]
    
    lineID = A[t[0]+1:t[1], 0]
    linei = A[t[0]+1:t[1], 1]
    linej = A[t[0]+1:t[1], 2]
    lineL = A[t[0]+1:t[1], 3]
    
    branchi = A[t[1]+1:t[2], 0]
    branchb = A[t[1]+1:t[2], 1] if A[t[1]+1:t[2]].shape[1] > 1 else np.array([])
    
    if t[2]+1 < t[3]:
        trans_data = A[t[2]+1:t[3]]
        if trans_data.size > 0:
            transID = trans_data[:, 0]
            transi = trans_data[:, 1]
            transj = trans_data[:, 2]
            transr = trans_data[:, 3]
            transx = trans_data[:, 4]
            transk = trans_data[:, 5]
            
            if trans_data.shape[1] > 6:
                transkMin = trans_data[:, 6]
                transkMax = trans_data[:, 7] if trans_data.shape[1] > 7 else np.array([])
            else:
                transkMin = np.array([])
                transkMax = np.array([])
        else:
            transID = transi = transj = transr = transx = transk = transkMin = transkMax = np.array([])
    else:
        transID = transi = transj = transr = transx = transk = transkMin = transkMax = np.array([])
    
    if t[3]+1 < t[4]:
        pq_data = A[t[3]+1:t[4]]
        if pq_data.size > 0:
            PQi = pq_data[:, 0]
            PQx = pq_data[:, 1]
            n = 4*(PQi-1)+PQx
            n = n.astype(int)
            
            PG = pq_data[:, 2] / SB
            QG = pq_data[:, 3] / SB
            PD = pq_data[:, 4] / SB
            QD = pq_data[:, 5] / SB
            
            PG_sparse = sparse.csr_matrix((PG, (n, np.zeros_like(n))), shape=(4*Nodes, 1))
            QG_sparse = sparse.csr_matrix((QG, (n, np.zeros_like(n))), shape=(4*Nodes, 1))
            PD_sparse = sparse.csr_matrix((PD, (n, np.zeros_like(n))), shape=(4*Nodes, 1))
            QD_sparse = sparse.csr_matrix((QD, (n, np.zeros_like(n))), shape=(4*Nodes, 1))
        else:
            PQi = PQx = np.array([])
            PG_sparse = QG_sparse = PD_sparse = QD_sparse = sparse.csr_matrix((4*Nodes, 1))
    else:
        PQi = PQx = np.array([])
        PG_sparse = QG_sparse = PD_sparse = QD_sparse = sparse.csr_matrix((4*Nodes, 1))
    
    if t[4]+1 < t[5]:
        pv_data = A[t[4]+1:t[5]]
        if pv_data.size > 0:
            PVi = pv_data[:, 0]
            PVx = pv_data[:, 1]
            PVV = pv_data[:, 2]
            PV_deta = pv_data[:, 3]
            PVQmin = pv_data[:, 2] / SB
            PVQmax = pv_data[:, 3] / SB
        else:
            PVi = PVx = PVV = PV_deta = PVQmin = PVQmax = np.array([])
    else:
        PVi = PVx = PVV = PV_deta = PVQmin = PVQmax = np.array([])
    
    if len(t) > 5 and t[5]+1 < len(A):
        if len(t) > 6:
            ng_data = A[t[5]+1:t[6]]
        else:
            ng_data = A[t[5]+1:]
            
        if ng_data.size > 0:
            NGi = ng_data[:, 0]
            OP_0 = ng_data[:, 1]
            OP_1 = ng_data[:, 2]
            OP_2 = ng_data[:, 3]
            NGmin = ng_data[:, 4] / SB
            NGmax = ng_data[:, 5] / SB
        else:
            NGi = OP_0 = OP_1 = OP_2 = NGmin = NGmax = np.array([])
    else:
        NGi = OP_0 = OP_1 = OP_2 = NGmin = NGmax = np.array([])
    
    return (Nodes, linenum, SB, maxIters, OPdata1, precision, OPdata2, balanceNum, 
            balancenotes, balancevoltage, balanceangle,
            lineID, linei, linej, lineL,
            branchi, branchb,
            transID, transi, transj, transr, transx, transk, transkMin, transkMax,
            PQi, PQx, PG_sparse, QG_sparse, PD_sparse, QD_sparse,
            PVi, PVx, PVV, PV_deta, PVQmin, PVQmax,
            NGi, OP_0, OP_1, OP_2, NGmin, NGmax) 
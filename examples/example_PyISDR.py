%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
np.random.seed(12)


data = loadmat('S1_EEG.mat')
G = data["G"]
M= data["M"]
SC = data["SC"]

import time
ts = time.time()
G = data["G"]
M= data["M"]
SC = data["SC"]

m_p = 2
A = np.zeros((G.shape[1], G.shape[1]*m_p), dtype=np.float64)
A[:, (m_p-1)*G.shape[1]:] = np.eye(G.shape[1])
Q = np.array([i for i in range(G.shape[1])])
before = G.shape[1]*m_p


a_scale = 1
g_scale = G.max()
m_scale = M.max()
for p in range(10):
    print(f"############### Iteration {p+1} ################")
    lambda_v = 0.01*Compute_alpha_max(np.dot(G, A), M, m_p)
    print(lambda_v)
    
    cls = reconstruct_activity(nbr_sources=G.shape[1], nbr_sensors=G.shape[0], m_p=m_p, lambda_v=lambda_v, n_t=M.shape[1], nbr_iterations=1000, tol=1e-3)    
    Jsdict, R, mse_list, gap = cls.run(M[:, m_p:], G, A)
    te = time.time()
    active_list = np.where(np.linalg.norm(Jsdict, axis=1) > 0)[0]
    Jsdictv2 = Jsdict[active_list, :]
     
    Gap_difference = np.abs(gap[-1] - gap[-2])
        
    print(f"Execution of eisdr is {(te-ts)/60:.3f} min | # of active sources {len(active_list)} | Gap difference {Gap_difference}")

    Q = Q[active_list]
    Sct = SC[active_list, :]
    Sct = Sct[:, active_list]
    Sct = Sct.astype(int)
    Gx = G[:, active_list]
    

    
    tsa = time.time()
    A=Astep(Gx, M, Sct, Jsdictv2, m_p)
    tea = time.time()
    print("Execution of Astep in {} min".format((tea-tsa)/60))

    import seaborn as sns

    # for i in range(A.shape[0]):
    #     A[i, :] /= max(np.abs(A[i, :]))
    # a[a==0] =None
    a = A.copy()
    a[a==0] = None

    plt.figure(figsize=(20, 4))
    plt.subplot(1,5,1)
    plt.title("Gap")
    plt.plot(np.array(gap));
    plt.subplot(1,5,2)
    plt.title("MSE")
    plt.plot(np.array(mse_list));
    plt.subplot(1,5,3)
    plt.title("Activation")
    plt.plot(data['time'][0][:Jsdictv2.shape[1]], Jsdictv2.T);
    plt.subplot(1,5,4)
    xticklabels=list(active_list)+list(active_list)
    yticklabels=active_list
    sns.heatmap(a)
    plt.xticks(rotation=45) 
    plt.subplot(1,5,5)
    n = np.where(Q == 42)[0]
    if len(n):
        plt.plot(Jsdictv2[n[0], :]);


        plt.title("Fusiform area")
    plt.tight_layout()
    plt.show()
    G = Gx
    a_scale = A.max()
    if before == A.shape[0]:
        break
    before = A.shape[0]




#     matrix = np.zeros((A.shape[1], A.shape[1]))
#     for i in range(m_p):
#         matrix[:A.shape[0], i*A.shape[0]: (i+1)*A.shape[0]] = A[:, (m_p-i-1)*A.shape[0]:(m_p-i)*A.shape[0]]
#     matrix[:A.shape[0], :] = A
#     for i in range(m_p-1):
#         matrix[A.shape[0]*(i+1):(i+2)*A.shape[0], A.shape[0]*i:(i+1)*A.shape[0]] = np.eye(A.shape[0])
#     eigenvalues, eigenvectors = np.linalg.eig(matrix)

#     spectral_radius = np.max(np.abs(eigenvalues))

#     # Normalize the matrix to ensure eigenvalues are smaller than 1 in absolute value
#     if spectral_radius > 1:
#         scaling_factor = 0.95 / spectral_radius  # Choose a scaling factor
#         A = A * scaling_factor
    # print("scaling_factor: ", scaling_factor, "spectral_radius: ", spectral_radius)

import numpy as np
from LS_forward_model import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# load and get data
data = np.load('Fdata.npz', allow_pickle=True)
Headers = list(data.keys())
for i in range(0, len(Headers)):
    H = Headers[i]
    str2 = H + ' = data[' + '"' + H + '"' + ']'
    exec(str2)


EC = RL_lin.T
ECf = np.fliplr(EC)
EC = np.vstack([EC, ECf])


layer_depths = np.array(
    [0, 0.2, 0.43, 0.7, 1.01, 1.37, 1.79, 2.27, 2.83, 3.47, 4.22, 5.08, 6.])
coiloffset = [1.48, 2.82, 4.49]
# coiloffset = [0.32, 0.71, 1.18]

ECa_hcp_ls = np.empty((40000, 3))
ECa_vcp_ls = np.empty((40000, 3))

for idx_off, offset in enumerate(coiloffset):

    for idx in range(0, 40000):
        # for idx in range(0, 20):

        hcp, vcp = computeForwardResponse(
            layer_depths, EC[idx], offset, 2 * np.pi * 10000)

        ECa_hcp_ls[idx, idx_off] = hcp
        ECa_vcp_ls[idx, idx_off] = vcp

np.save('./CNN_6ECa_more_models/RL_lin.npy', EC)
np.save('./CNN_6ECa_more_models/ECa_hcp.npy', ECa_hcp_ls)
np.save('./CNN_6ECa_more_models/ECa_vcp.npy', ECa_vcp_ls)

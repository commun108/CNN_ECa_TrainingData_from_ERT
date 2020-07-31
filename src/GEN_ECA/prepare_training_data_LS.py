import numpy as np
from LS_forward_model import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets


def regression_analysis(ax, X, y):
    """."""

    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    # Predict data of estimated models
    line_X = np.arange(0, 0.1, 0.01)[:, np.newaxis]
    line_y = lr.predict(line_X)
    line_y_diff = lr.predict(X)

    stdev_lr = np.sqrt(sum((line_y_diff - y)**2) / (len(y) - 2))

    # p = 0.997
    gaussian_critical_value = 2.74
    # p = 0.95
    gaussian_critical_value = 1.664

    lw = 1
    ax.plot(line_X, line_y, color='navy', linewidth=lw,
            label=r'R$^2$: {}'.format(round(lr.score(X, y), 2)))
    ax.plot(line_X, line_y - gaussian_critical_value * stdev_lr,
            color='navy', linewidth=lw, linestyle='--')
    ax.plot(line_X, line_y + gaussian_critical_value * stdev_lr,
            color='navy', linewidth=lw, linestyle='--')

    ax.legend(loc='best')
    ax.grid(True, linestyle='dotted')

    return lr


# load and get data
data = np.load('Fdata.npz', allow_pickle=True)
Headers = list(data.keys())
for i in range(0, len(Headers)):
    H = Headers[i]
    str2 = H + ' = data[' + '"' + H + '"' + ']'
    exec(str2)

# make input and output
ECa_hcp = ECa_hcp.T
ECa_vcp = ECa_vcp.T
EC = RL_lin.T
ECa = np.append(ECa_vcp, ECa_hcp, axis=1)

layer_depths = np.array(
    [0, 0.2, 0.43, 0.7, 1.01, 1.37, 1.79, 2.27, 2.83, 3.47, 4.22, 5.08, 6.])
coiloffset = [1.48, 2.82, 4.49]
coiloffset = [0.32, 0.71, 1.18]

ECa_hcp_ls = np.empty((20000, 3))
ECa_vcp_ls = np.empty((20000, 3))

for idx_off, offset in enumerate(coiloffset):

    for idx in range(0, 20000):
        # for idx in range(0, 20):

        hcp, vcp = computeForwardResponse(
            layer_depths, EC[idx], offset, 2 * np.pi * 10000)

        ECa_hcp_ls[idx, idx_off] = hcp
        ECa_vcp_ls[idx, idx_off] = vcp


sns.set(color_codes=True)

fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all')


for idx, ax in enumerate(axes[:, 0]):
    X = ECa_hcp[:, idx].reshape(-1, 1)
    y = ECa_hcp_ls[:, idx].reshape(-1, 1)
    ax.plot(X, y, 'o', markersize=0.3)
    lr = regression_analysis(ax, X, y)
    ax.set_ylabel("ECa [S/m]\nLS forward")


for idx, ax in enumerate(axes[:, 1]):
    X = ECa_vcp[:, idx].reshape(-1, 1)
    y = ECa_vcp_ls[:, idx].reshape(-1, 1)
    ax.plot(X, y, 'co', markersize=0.3)
    lr = regression_analysis(ax, X, y)
    ax.set_ylabel("Coil offset: {} m".format(coiloffset[idx]))
    ax.yaxis.set_label_position("right")

axes[0, 0].set_xlim([0, 0.1])
axes[0, 0].set_ylim([0, 0.1])
axes[0, 0].set_title('HCP')
axes[0, 1].set_title('VCP')
axes[-1, 0].set_xlabel("ECa [S/m]\nFull EM forward")
axes[-1, 1].set_xlabel("ECa [S/m]\nFull EM forward")

plt.show()

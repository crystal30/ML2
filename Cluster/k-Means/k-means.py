# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans


def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d


if __name__ == "__main__":
    N = 400
    centers = 4
    # 产生400个样本数据，每个样本两维特征，共有4个中心点，即4个簇。
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    # 产生4个簇的方差不同
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1,2.5,0.5,2), random_state=2)
    # 产生4个簇的样本数量不同
    data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)
    # 将样本数据旋转
    m = np.array(((1, 1), (1, 3)))
    data_r = data.dot(m)

    cls = KMeans(n_clusters=4, init='k-means++')
    y_hat = cls.fit_predict(data)
    y2_hat = cls.fit_predict(data2)
    y3_hat = cls.fit_predict(data3)
    y_r_hat = cls.fit_predict(data_r)

    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(9, 10), facecolor='w')
    plt.subplot(421)
    plt.title('original data')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(422)
    plt.title('KMeans++ cluster')
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(423)
    plt.title('rotation data')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data_r, axis=0)
    x1_max, x2_max = np.max(data_r, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(424)
    plt.title('rotation data KMeans++ cluster')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y_r_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(425)
    plt.title('data with unequal variances')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data2, axis=0)
    x1_max, x2_max = np.max(data2, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(426)
    plt.title('data with unequal variances and KMeans++ cluster')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(427)
    plt.title('data with unequal num')
    plt.scatter(data3[:, 0], data3[:, 1], s=30, c=y3, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data3, axis=0)
    x1_max, x2_max = np.max(data3, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(428)
    plt.title('data with unequal num and KMeans++ cluster')
    plt.scatter(data3[:, 0], data3[:, 1], c=y3_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.tight_layout(2, rect=(0, 0, 1, 0.97))
    plt.suptitle('the influence of data distribution on KMeans++', fontsize=18)
    # https://github.com/matplotlib/matplotlib/issues/829
    # plt.subplots_adjust(top=0.92)
    plt.show()

        



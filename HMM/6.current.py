# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings


def extend(a, b):
    return 1.05*a-0.05*b, 1.05*b-0.05*a


def data_clean(show):
    y = data['Current']
    idx_time = (data['Time'] > 6.76) & (data['Time'] < 6.91)
    idx_current = (y > 9.31) & (y < 9.33)
    idx = idx_time & idx_current
    data['Current'][idx] += (y[y > 9.34].median() - y[idx].median())

    idx_current = (y > 9.01) & (y < 9.04)
    idx = idx_time & idx_current
    data['Current'][idx] += (y[y < 9.1].median() - y[idx].median())

    if show:
        plt.figure(figsize=(13,6))
        plt.plot(data['Time'], y, 'r.-', lw=0.2)
        plt.ylim(extend(y.min(), y.max()))
        plt.grid(b=True, ls=':')
        plt.xlabel('time', fontsize=14)  # 时间
        plt.ylabel('current intensity', fontsize=14) # 电流强度
        plt.title('changes in the original current', fontsize=16) # 原始电流的变化情况
        plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")   # hmmlearn(0.2.0) < sklearn(0.18)
    pd.set_option('display.width', 100)
    np.set_printoptions(edgeitems=100)

    n_components = 3
    data = pd.read_excel(io='./data/Current.xls', sheet_name='Sheet1', header=0)
    # data['Current'] = MinMaxScaler().fit_transform(data['Current'])
    data['Current'] *= 1e6

    # 去除明显的异常值
    data_clean(False)

    x = np.array(data['Time']).reshape(-1, 1)
    y = np.array(data['Current']).reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=10)
    model.fit(y)
    components = model.predict_proba(y)
    components_state = model.predict(y)
    components_pd = pd.DataFrame(components, columns=np.arange(n_components), index=data.index)
    data = pd.concat((data, components_pd), axis=1)
    print('data = \n', data)

    plt.figure(num=1, facecolor='w', figsize=(8, 9))
    plt.subplot(n_components+1, 1, 1)
    plt.plot(x, y, 'r.-', lw=0.2)
    plt.ylim(extend(y.min(), y.max()))
    plt.grid(b=True, ls=':')
    plt.xlabel('time', fontsize=14)
    plt.ylabel('current intensity', fontsize=14) # 电流强度
    plt.title('changes in the original current', fontsize=16) # 原始电流的变化情况
    for component in np.arange(n_components):
        plt.subplot(n_components+1, 1, component+2)
        plt.plot(x, data[component], 'r.')
        plt.ylim((-0.1, 1.1))
        plt.grid(b=True, ls=':')
        plt.ylabel('component probability', fontsize=14)
        plt.xlabel('time', fontsize=14)
        plt.title('probability distribution of component %d' % (component+1), fontsize=16) # 第%d组份的概率分布
    plt.suptitle('The relationship between original current/composition and time', fontsize=18)  # 原始电流/组份与时间的变化关系
    plt.tight_layout(pad=1, rect=(0, 0, 1, 0.96))
    plt.savefig('1.png')

    plt.figure(num=2, facecolor='w', figsize=(8, 8))
    for component in np.arange(n_components):
        plt.subplot(n_components, 1, component + 1)
        plt.plot(y, data[component], 'r.')
        plt.xlim(extend(y.min(), y.max()))
        plt.ylim((-0.1, 1.1))
        plt.grid(b=True, ls=':')
        plt.xlabel('original current intensity', fontsize=14) # 原始电流强度
        plt.ylabel('component probability', fontsize=14) # 组份概率
        plt.title('probability distribution of component %d' % (component + 1), fontsize=16) # 第%d组份的概率分布
    plt.suptitle('probability distribution of components', fontsize=18) # 各组份的概率分布
    plt.tight_layout(pad=1, rect=(0, 0, 1, 0.96))
    plt.savefig('2.png')

    y_new = np.zeros_like(data['Current'])
    for component in np.arange(n_components):
        idx = components_state == component
        y_new[idx] = np.median(y[idx])
    print('整流后均值：', np.mean(y_new))
    data['New'] = y_new
    data.to_excel('./data/new_current.xls', index=False)

    plt.figure(num=3, facecolor='w', figsize=(8, 8))
    plt.subplot(211)
    plt.plot(x, y_new, 'r.-', lw=0.2)
    plt.ylim(extend(y_new.min(), y_new.max()))
    plt.grid(b=True, ls=':')
    plt.xlabel('time', fontsize=14) # 时间
    plt.ylabel('current intensity', fontsize=14) # 电流强度
    plt.title('current data after shaping', fontsize=18) # 整形后的电流数据
    plt.subplot(212)
    plt.plot(data['Current'], data['New'], 'r.')
    plt.ylim(extend(data['New'].min(), data['New'].max()))
    plt.grid(b=True, ls=':')
    plt.xlabel('original current intensity', fontsize=14) # 原始电流强度
    plt.ylabel('corrected current intensity', fontsize=14) # 修正后的电流强度
    plt.title('current relationship before and after shaping', fontsize=18) # 整形前后的电流关系
    plt.tight_layout(pad=1, rect=(0, 0, 1, 1))
    plt.savefig('3.png')

    plt.show()

# SVD分解
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def restore(sigma, u, v, K):
    m = u.shape[0]
    n = v.shape[0]
    svd_result = np.zeros((m, n))
    for k in range(K + 1):
        for i in range(m):
            svd_result[i] += sigma[k] * u[i][k] * v[k]

    svd_result[svd_result < 0] = 0
    svd_result[svd_result > 255] = 255
    return np.rint(svd_result).astype('uint8')


if __name__ == '__main__':
    palace = Image.open("palace.jpeg", "r")
    print(palace)

    output_path = r'./SVD'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    palace_array = np.array(palace)

    K = 50
    u_r, sigma_r, v_r = np.linalg.svd(palace_array[:, :, 0])
    u_g, sigma_g, v_g = np.linalg.svd(palace_array[:, :, 1])
    u_b, sigma_b, v_b = np.linalg.svd(palace_array[:, :, 2])
    plt.figure(figsize=(10, 10))

for k in range(1, K + 1):
    R = restore(sigma_r, u_r, v_r, k)
    G = restore(sigma_g, u_g, v_g, k)
    B = restore(sigma_b, u_b, v_g, k)
    I = np.stack((R, G, B), axis=2)
    Image.fromarray(I).save('%s/svd_%d.png' % (output_path, k))
    if k <= 12:
        plt.subplot(3, 4, k)
        plt.imshow(I)
        plt.axis('off')
        plt.title('svd num: %d' % k)

plt.suptitle('SVD and image decomposition', fontsize=20)



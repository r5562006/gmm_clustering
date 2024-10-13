# gmm_clustering.py
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 生成隨機數據
data = np.random.rand(100, 2)

# 應用 GMM 聚類
gmm = GaussianMixture(n_components=3)
gmm.fit(data)
labels = gmm.predict(data)

# 可視化結果
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.show()
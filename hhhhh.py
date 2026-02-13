import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set()  # use seaborn style
data = np.random.rand(10, 2)
dendrogram(linkage(data, method='ward'))
plt.show()
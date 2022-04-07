# 余弦相似度
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

df1 = pd.read_csv("./cars_vector.csv")
df1.columns=["id","space","power","control","oil","comfortable","appearance","inner","value"]
X = df1[["space","power","control","oil","comfortable","appearance","inner","value"]]
X = np.array(X)
print(X)

A = np.array(
    [[0, 1, 0, 0, 0, 1, 1, 1],
     [0, 0, 1, 0, 0, 1 ,1, 1]])


cosine_dis = cosine_similarity(A, X)
print('cosine_dis:\n', cosine_dis)

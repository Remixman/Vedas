import numpy as np
from sklearn.decomposition import PCA

# Read sparce matrix file
f = open('./tools/tmp-so-vec.txt', 'r')
row, col = f.readline().split()
X = np.zeros((int(row), int(col)))
r = 0
while True:
  line = f.readline()
  if not line:
    break
  for idx_str in line.split():
    X[r][int(idx_str)] = 1
  r += 1
f.close()

# print(X)

# Dimensionality Reduction with PCA
pca = PCA(n_components=1)
pca.fit(X)
Z = pca.transform(X)
# print(pca.components_)


# TODO: if value is equal, use heuristic to decide distance ???

id_val_pair = list()
for r in range(int(row)):
  id_val_pair.append([Z[r][0], r+1])
id_val_pair.sort()


f2 = open('./tools/reassigned-so-id.txt', 'w')
for new_id, p in enumerate(id_val_pair, start=1):
  f2.write(str(p[1]) + ' ' + str(new_id) + '\n')
f2.close()
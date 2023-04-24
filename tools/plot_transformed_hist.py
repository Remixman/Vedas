import matplotlib.pyplot as plt

z = []
f = open('./id-transformed.txt', 'r')
lines = f.readlines()
for line in lines:
  zval = line.split()[0]
  z.append(float(zval) * 1e20)

print(z[:10])
f.close()

plt.style.use('ggplot')
plt.hist(z, bins=100, density=True)
plt.savefig('transformed-histogram.png')

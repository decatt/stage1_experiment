import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure()

data = [1]
y = np.array(data)
x = np.array(range(len(data)))

plt.scatter(x, y, color='red', marker='+')
plt.show()

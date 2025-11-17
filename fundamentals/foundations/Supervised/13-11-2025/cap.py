import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
print(digits.DESCR)
print(digits.data)
print(digits.target)

# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()

samples = digits.data


num_clusters = list(range(1,15))
inertias = []
for k in num_clusters:
  model = KMeans(n_clusters=k)
  model.fit(samples)
  inertias.append(model.inertia_)

plt.plot(num_clusters, inertias, '-o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.show()

model = KMeans(n_clusters=10, random_state=42)
model.fit(samples)
fig = plt.figure(figsize=(8,3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')
for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.06,6.40,6.86,6.86,2.13,0.00,0.00,0.00,5.79,6.40,6.02,7.62,2.90,0.00,0.00,0.00,0.84,3.58,7.55,5.87,0.61,0.00,3.51,6.33,7.62,7.62,5.18,0.23,0.00,0.00,7.62,7.62,7.62,7.62,6.86,5.95,0.00,0.00,2.44,3.05,3.05,3.58,4.80,4.80,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.14,4.19,4.50,0.23,0.00,0.00,0.00,1.60,7.40,7.40,7.55,3.96,0.00,0.00,0.92,6.86,6.86,2.13,7.01,4.88,0.00,0.00,3.51,7.62,7.62,7.62,7.40,1.37,0.00,0.00,0.30,2.21,2.29,2.21,0.76,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.15,1.91,2.06,0.00,0.00,0.00,0.00,3.35,6.71,7.62,7.62,1.37,0.00,0.00,0.00,3.36,6.18,7.62,7.17,0.69,0.00,0.00,0.00,2.90,7.24,7.09,1.60,0.00,0.00,0.00,4.19,7.62,5.57,0.92,0.53,0.00,0.00,0.00,7.62,7.55,6.48,7.47,7.62,2.44,0.00,0.00,5.34,5.34,4.96,3.89,3.05,0.46,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.53,2.82,0.92,0.00,0.00,0.00,0.92,4.80,7.47,7.62,3.28,0.00,0.00,1.52,7.47,7.40,4.27,1.30,0.00,0.00,0.00,0.69,6.48,5.72,0.23,0.00,0.00,0.00,0.00,0.00,7.32,7.62,7.24,5.03,0.00,0.00,0.00,0.00,5.11,7.62,7.55,4.58,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
])
new_labels = model.predict(new_samples)
print(new_labels)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')

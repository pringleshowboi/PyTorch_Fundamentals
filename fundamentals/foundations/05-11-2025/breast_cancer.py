from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

print(training_data)
print(training_labels)

from sklearn.neighbors import KNeighborsClassifier

for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors=k)
  classifier.fit(training_data, training_labels)
  print(classifier.score(validation_data, validation_labels))

import matplotlib.pyplot as plt
k_list = range(1,101)
accuracies = []

for i in k_list:
  classifier = KNeighborsClassifier(n_neighbors=i)
  classifier.fit(training_data, training_labels)
  print(classifier.score(validation_data, validation_labels))
  accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.show()
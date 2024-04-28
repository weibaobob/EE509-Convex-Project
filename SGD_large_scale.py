import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

# Create a large-scale synthetic dataset
X, y = make_classification(n_samples=1000000, n_features=20, n_informative=10, n_redundant=0, n_classes=2, random_state=42)

# Initialize SGD classifier
sgd_clf = SGDClassifier(max_iter=1000, tol=None, random_state=42)  

# Initialize GD classifier using LogisticRegression
gd_clf = LogisticRegression(solver='lbfgs', max_iter=1000, tol=1e-4, random_state=42)  

# Keeping track of accuracies and running times
sgd_accuracies = []
gd_accuracies = []
sgd_times = []
gd_times = []

# Define the number of iterations for the demonstration
num_iterations = 30
batch_size = 10000

# Loop over the number of iterations
for i in range(num_iterations):
    # Timing and updating the SGD classifier with a mini-batch
    start_time = time.time()
    sgd_clf.partial_fit(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size], classes=np.unique(y))
    sgd_time = time.time() - start_time
    sgd_times.append(sgd_time)
    
    sgd_accuracy = accuracy_score(y[:batch_size*(i+1)], sgd_clf.predict(X[:batch_size*(i+1)]))
    sgd_accuracies.append(sgd_accuracy)

    # Timing and fitting the GD classifier with the data available up to the current iteration
    start_time = time.time()
    gd_clf.fit(X[:batch_size*(i+1)], y[:batch_size*(i+1)])
    gd_time = time.time() - start_time
    gd_times.append(gd_time)
    
    gd_accuracy = accuracy_score(y[:batch_size*(i+1)], gd_clf.predict(X[:batch_size*(i+1)]))
    gd_accuracies.append(gd_accuracy)

    # Print result for each iteration including running times
    print(f"Iteration {i+1} - SGD Accuracy: {sgd_accuracy}, Time: {sgd_time}s, GD Accuracy: {gd_accuracy}, Time: {gd_time}s")

# Plot the accuracies for SGD and GD
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), sgd_accuracies, label='SGD')
plt.plot(range(1, num_iterations + 1), gd_accuracies, label='GD')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('SGD vs GD Accuracy Over Iterations')
plt.legend()    
plt.show()

# Plot the running times for SGD and GD
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), sgd_times, label='SGD Time')
plt.plot(range(1, num_iterations + 1), gd_times, label='GD Time')
plt.xlabel('Iterations')
plt.ylabel('Time (seconds)')
plt.title('SGD vs GD Running Time Over Iterations')
plt.legend()
plt.show()

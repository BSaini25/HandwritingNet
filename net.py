from emnist import extract_training_samples
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

print("Imported the EMNIST libraries we need!")

# X is images and y is labels
X, y = extract_training_samples("letters")

# Normalizing pixel values to be between 0 and 1
X = X / 255

# Using the first 60,000 for training and next 10,000 for testing
X_train, X_test = X[:60000], X[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# Recording the number of samples in each dataset and number of pixels in each image
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
print("Extracted our samples and divided our training and testing data sets")

img_index = 1200
img = X_train[img_index]
print("Image Label: " + str(chr(y_train[img_index]+96)))
#plt.imshow(img.reshape((28,28)))

# Creating MLP with 1 hidden layer with 50 neurons and sets it to run 20 times
mlp1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
print("Created our first MLP network")

# Testing MLP
mlp1.fit(X_train, y_train)
print("Training set score: %f" % mlp1.score(X_train, y_train))
print("Test set score: %f" % mlp1.score(X_test, y_test))

# Initializing a list with all the predicted values from the training set
y_pred = mlp1.predict(X_test)

# Visualizing the errors between predictions and actual labels using a confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm)
plt.show()

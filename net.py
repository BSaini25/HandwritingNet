from emnist import extract_training_samples
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import os
import cv2
import numpy

print("Imported EMNIST libraries.")

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
print("Extracted samples and divided training and testing data sets.")

"""
img_index = 1200
img = X_train[img_index]
print("Image Label: " + str(chr(y_train[img_index] + 96)))
#plt.imshow(img.reshape((28, 28)))
"""

"""
# Creating MLP with 1 hidden layer with 50 neurons and sets it to run 20 times
mlp1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
print("Created MLP network")

# Testing MLP
print("Training MLP network.")
mlp1.fit(X_train, y_train)
print("Training set score: %f" % mlp1.score(X_train, y_train))
print("Test set score: %f" % mlp1.score(X_test, y_test))

# Initializing a list with all the predicted values from the training set
y_pred_1 = mlp1.predict(X_test)

# Visualizing the errors between predictions and actual labels using a confusion matrix
cm = confusion_matrix(y_test, y_pred_1)
#plt.matshow(cm)
#plt.show()

user_input = input("Would you like to see the errors? (y/n) ")
while (user_input.lower() == "y"):
    # Checking how many times a letter was predicted wrong
    actual_letter = input("What is the letter you want to see errors for? ")
    predicted_letter = input("What do you think the network classified it as? ")

    mistakes = []
    for i in range(len(y_test)):
        if (y_test[i] == (ord(actual_letter) - 96) and y_pred_1[i] == (ord(predicted_letter) - 96)):
            mistakes.append(i)
    print("There were " + str(len(mistakes)) + " times that the letter " + actual_letter + " was predicted to be the letter " + predicted_letter + ".")

    # Asking user which error they want to see
    err_num = input("Which error do you want to see? (1, 2, etc) ")
    if (int(err_num) <= len(mistakes)):
        img = X_test[mistakes[int(err_num) - 1]]
        plt.imshow(img.reshape((28, 28)))
        plt.show()
    else:
        print("Couldn't show error because there weren't enough errors.")
    user_input = input("Would you like to see the errors? (y/n) ")
"""

# Creating a second MLP with 5 hidden layer with 100 neurons each and sets it to run 50 times
mlp2 = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,), max_iter=50, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
print("Training MLP network.")
mlp2.fit(X_train, y_train)
print("Training set score: %f" % mlp2.score(X_train, y_train))
print("Test set score: %f" % mlp2.score(X_test, y_test))

# Initializing a list with all the predicted values from the training set
y_pred_2 = mlp2.predict(X_test)

# Visualizing the errors between predictions and actual labels using a confusion matrix
cm2 = confusion_matrix(y_test, y_pred_2)
plt.matshow(cm2)
#plt.show()

# Putting all the data in the "files" variable
path, dirs, files = next(os.walk("letters_2\\"))
files.sort()

# Processing all the scanned images and adding them to the handwritten_text
handwritten_text = []
for i in range(len(files)):
  img = cv2.imread("letters_2\\" + files[i], cv2.IMREAD_GRAYSCALE)
  handwritten_text.append(img)
print("Imported the scanned images.")
#plt.imshow(handwritten_text[5])
#plt.show()

# Processing the images the same way EMNIST did
processed_text = []
for img in handwritten_text:
    # Applying a Gaussian Blur
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # Extracting the Region of Interest and centering in square
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)

    if (w > 0 and h > 0):
        if w > h:
            y = y - (w - h) // 2
            img = img[y:y + w, x:x + w]
        else:
            x = x - (h - w) // 2
            img = img[y:y + h, x:x + h]
    # Resizing the image to be 28x28
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_CUBIC)

    # Normalizing the pixels and reshaping before adding to the new array
    img = img / 255
    img = img.reshape((28, 28))
    processed_text.append(img)

typed_text = ""
for letter in processed_text:
    # Resizing the image to be 28x28
    letter = cv2.resize(letter, (28, 28), interpolation = cv2.INTER_CUBIC)

    # Checking if the image is blank
    total_pixel_value = 0
    for j in range(28):
        for k in range(28):
            total_pixel_value += letter[j, k]
    if total_pixel_value < 20:
        typed_text = typed_text + " "
    else:
        single_item_array = (numpy.array(letter)).reshape(1, 784)
        prediction = mlp2.predict(single_item_array)
        typed_text = typed_text + str(chr(prediction[0] + 96))

print("Converted images to text")
print(typed_text)
#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Input data
X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
Y = np.array([60, 70, 80, 90, 100, 110, 120, 130, 140, 150])

# Calculate slope and intercept values
n = len(X)
b = ((n * np.sum(X*Y)) - (np.sum(X) * np.sum(Y))) / ((n * np.sum(X*2)) - (np.sum(X)*2))
a = (np.sum(Y) - b * np.sum(X)) / n

# Make a prediction for a given number of hours studied
hours_studied = 7
test_score = a + b * hours_studied
print(f"Predicted test score for {hours_studied} hours studied: {test_score}")

# Plot the data and regression line
plt.scatter(X, Y)
plt.plot(X, a + b*X, c='r')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.show()


# In[12]:


import numpy as np
from scipy.optimize import curve_fit

# Define the polynomial function
def func(x, a, b, c):
    return a + b*x + c*x**2

# Define the input and output variables
x_data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
y_data = np.array([60, 70, 80, 90, 100, 110, 120, 130, 140, 150])

# Estimate the coefficients
popt, pcov = curve_fit(func, x_data, y_data)

# Print the coefficients
print(popt)


# In[14]:


import numpy as np
from sklearn.linear_model import LinearRegression
# input data (hours studied)
X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).reshape((-1, 1))
# output data (test scores)
y = np.array([60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
# create a linear regression model and fit it to the data
model = LinearRegression().fit(X, y)
# print the coefficients of the linear regression model (slope and intercept)
print('Coefficients:', model.coef_, model.intercept_)
# predict the test scores for a new set of hours studied (e.g., 12 hours)
new_X = np.array([12]).reshape((-1, 1))
print('Predicted test score for 12 hours studied:', model.predict(new_X))


# In[15]:


import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
without_reshape = np.array([5, 15, 25, 35, 45, 55, 65])
print("Before Reshaping Numpy Array",without_reshape)
x=np.array([5, 15, 25, 35, 45, 55, 65]).reshape((-1, 1))
print("After Reshaping Numpy Array",x)
y = np.array([5, 20, 14, 32, 22, 38, 43])
print(y)
model = LinearRegression()
model.fit(x,y)   
r_sq = model.score(x, y)   
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(x) 
print('predicted response:', y_pred, sep='\n')

plt.plot(x,y)
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-5.0, 5.0, 0.1)

## You can adjust the slope and intercept to verify the changes in the graph
y = np.power(x, 2)
y_noise = 2 * np.random.normal(size = x.size)
ydata = y + y_noise
plt.plot(x, ydata, 'bo')
plt.plot(x, y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# In[17]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
print(x)
print(y)
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x, y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
#model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)

print("Predicted Classes" ,model.classes_)
print("Intercept ",model.intercept_)
print("Coefient Value ", model.coef_)
print("Predicted Values ",model.predict_proba(x))

plt.plot(model.predict_proba(x))
plt.show()


# In[18]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create an instance of the SVM classifier with a linear kernel
svm = SVC(kernel='linear')
# Train the classifier on the training data
svm.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = svm.predict(X_test)
# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load iris dataset
iris = load_iris()
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
# create KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
# fit the classifier to the training data
knn.fit(X_train, y_train)
# predict the classes of testing data
y_pred = knn.predict(X_test)
# evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[20]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load iris dataset
iris = load_iris()
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
# create Decision Tree classifier
clf = DecisionTreeClassifier()
# fit the classifier to the training data
clf.fit(X_train, y_train)
# predict the classes of testing data
y_pred = clf.predict(X_test)
# evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[22]:





# In[23]:





# In[24]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# load 20 newsgroups dataset
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
# extract features from text using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
y_train = newsgroups_train.target
# create Naive Bayes classifier
clf = MultinomialNB()
# fit the classifier to the training data
clf.fit(X_train, y_train)
# evaluate the classifier on test data
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
X_test = vectorizer.transform(newsgroups_test.data)
y_test = newsgroups_test.target
y_pred = clf.predict(X_test)
# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[25]:


from sklearn.cluster import KMeans
import numpy as np

# Generate some random data
X = np.random.rand(100, 2)

# Create a KMeans object and fit the data
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the centroids of each cluster
centroids = kmeans.cluster_centers_

# Print the results
print("Cluster labels:")
print(labels)
print("Centroids:")
print(centroids)


# In[26]:


from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Perform PCA on the training set
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
# Train a KNN classifier on the transformed training set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)
# Apply the same PCA transformation to the test set
X_test_pca = pca.transform(X_test)
# Predict the classes of the test set using the trained KNN classifier
y_pred = knn.predict(X_test_pca)
# Evaluate the accuracy of the classification
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[27]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Define the transaction data
transactions = [['bread', 'milk', 'vegetables'],
                ['bread', 'milk', 'cheese'],
                ['milk', 'cheese', 'eggs'],
                ['bread', 'cheese', 'eggs'],
                ['bread', 'milk', 'cheese', 'eggs', 'vegetables'],
                ['milk', 'eggs', 'vegetables'],
                ['bread', 'eggs', 'vegetables']]
# Transform the transaction data into a one-hot encoded boolean matrix
te = TransactionEncoder()
te_matrix = te.fit_transform(transactions)

# Convert the one-hot encoded matrix into a pandas DataFrame
df = pd.DataFrame(te_matrix, columns=te.columns_)
# Find frequent itemsets with minimum support of 0.4
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Generate association rules with minimum confidence of 0.7
association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Print the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(association_rules_df)


# In[28]:


pip install mlxtend


# In[29]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Define the transaction data
transactions = [['bread', 'milk', 'vegetables'],
                ['bread', 'milk', 'cheese'],
                ['milk', 'cheese', 'eggs'],
                ['bread', 'cheese', 'eggs'],
                ['bread', 'milk', 'cheese', 'eggs', 'vegetables'],
                ['milk', 'eggs', 'vegetables'],
                ['bread', 'eggs', 'vegetables']]
# Transform the transaction data into a one-hot encoded boolean matrix
te = TransactionEncoder()
te_matrix = te.fit_transform(transactions)

# Convert the one-hot encoded matrix into a pandas DataFrame
df = pd.DataFrame(te_matrix, columns=te.columns_)
# Find frequent itemsets with minimum support of 0.4
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Generate association rules with minimum confidence of 0.7
association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Print the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(association_rules_df)


# In[30]:


# Import the Sequential model and layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

batch_size = 20

# Training Augmentation configuration
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Testing Augmentation - Only Rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)
# Generates batches of Augmented Image data
train_generator = train_datagen.flow_from_directory('Covid19Train', target_size = (300, 300),
			batch_size = batch_size,
			class_mode = 'binary')
# Generator for validation data
validation_generator = test_datagen.flow_from_directory('Covid19Test',
                       target_size = (300, 300),
                       batch_size = batch_size,
                       class_mode = 'binary')
model.fit(train_generator,
                    	epochs = 3,
                    	validation_data = validation_generator, verbose = 1)
# Evaluating model performance on Testing data
loss, accuracy = model.evaluate(validation_generator)
print('\nAccuracy: ', accuracy, '\nLoss: ', loss)


# In[34]:


pip install keras


# In[35]:


# Import the Sequential model and layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

batch_size = 20

# Training Augmentation configuration
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Testing Augmentation - Only Rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)
# Generates batches of Augmented Image data
train_generator = train_datagen.flow_from_directory('Covid19Train', target_size = (300, 300),
			batch_size = batch_size,
			class_mode = 'binary')
# Generator for validation data
validation_generator = test_datagen.flow_from_directory('Covid19Test',
                       target_size = (300, 300),
                       batch_size = batch_size,
                       class_mode = 'binary')
model.fit(train_generator,
                    	epochs = 3,
                    	validation_data = validation_generator, verbose = 1)
# Evaluating model performance on Testing data
loss, accuracy = model.evaluate(validation_generator)
print('\nAccuracy: ', accuracy, '\nLoss: ', loss)


# In[39]:


pip install tensorflow


# In[41]:


# Import the Sequential model and layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

batch_size = 20

# Training Augmentation configuration
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Testing Augmentation - Only Rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)
# Generates batches of Augmented Image data
train_generator = train_datagen.flow_from_directory('Covid19Train', target_size = (300, 300),
			batch_size = batch_size,
			class_mode = 'binary')
# Generator for validation data
validation_generator = test_datagen.flow_from_directory('Covid19Test',
                       target_size = (300, 300),
                       batch_size = batch_size,
                       class_mode = 'binary')
model.fit(train_generator,
                    	epochs = 3,
                    	validation_data = validation_generator, verbose = 1)
# Evaluating model performance on Testing data
loss, accuracy = model.evaluate(validation_generator)
print('\nAccuracy: ', accuracy, '\nLoss: ', loss)


# In[ ]:





# In[ ]:





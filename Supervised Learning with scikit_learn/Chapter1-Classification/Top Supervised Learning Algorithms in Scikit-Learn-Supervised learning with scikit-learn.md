Supervised learning is a cornerstone of machine learning, enabling us to build models that classify data accurately and make predictions on new instances. Scikit-Learn is a popular machine learning library in Python that provides a variety of classification algorithms. Classification is a fundamental task in machine learning, where the goal is to assign a label or category to a given input based on its features.

In this guide, we explore key classification algorithms in Scikit-Learn, including

- Logistic Regression,
- SVM,
- Neural Networks,
- KNN
- Tree-based methods (DT,RF)

Whether you’re a beginner or looking to deepen your knowledge, this step-by-step tutorial will walk you through using Scikit-Learn in Python to create, train, and evaluate machine learning models effectively. Discover how to choose the best algorithm for your dataset and improve model accuracy with Scikit-Learn’s powerful tools


## 📑 Table of Contents  

- [Understanding Classification](#Understanding-Classification)  
- [Scikit-Learn Overview](#Scikit-Learn_Overview)  
- [Common steps of Scikit-Learn for classification](#Common_steps_of_Scikit-Learn_for_classification)  
- Logistic regression (LR)
- Artificial neural networks (ANN)
- Support Vector Machine (SVM)
- Naive Bayes (NB)
- KNN (K-Nearest Neighbors)
- Decision Tree Classifier
- RandomForestClassifier
- ExtraTreesClassifier
- Gradient Boosting Trees (GBT)

# **Understanding Classification** 

Def: Classification is a supervised learning technique in which we train a model on labeled data to make predictions on unseen instances. The labeled data consists of input variables (features) and output variables (labels or classes). The goal is to learn a mapping function that can accurately predict the class labels of new instances.

When it comes to classification, there are two main types: binary classification and multiclass classification. In binary classification, the target variable has only two classes, such as spam or not spam. In multiclass classification, the target variable can have more than two classes, like classifying images into different categories.

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised%20Learning%20with%20scikit_learn/Chapter1-Classification/33346d3c-afee-4069-a36e-9ccee57c783a_506x640.jpg"></a>
</p>

# **Scikit-Learn Overview**

Scikit-Learn is a powerful and easy-to-use library for machine learning in Python. It provides a wide range of algorithms and tools for various tasks, including classification. Some of the key features of Scikit-Learn include:

- Easy integration with other Python libraries like NumPy and Pandas.
- Consistent and intuitive API for all algorithms.
- Extensive documentation and community support.
- Efficient implementation of state-of-the-art algorithms.
- Cross-validation and model evaluation metrics.

# **Common steps of Scikit-Learn for classification**

To use Scikit-Learn for classification, you can follow these steps:
Import the necessary libraries.
- Load the data.
- Split the data into a training set and a test set.
- Choose a classification algorithm.
- Train the model on the training set.
- Evaluate the model on the test set.
- Make predictions on new data.

# **Scikit-Learn Installation**
 
From the notebook, you can ensure this via;


<pre> ```python -m pip install --upgrade scikit-learn==0.23.0 ``` </pre>

Alternatively, you could also run this from the command line inside of a virtual environment;


<pre> ```python -m pip install --upgrade scikit-learn==0.23.0 ``` </pre>

# **Load The Data**

Scikit-Learn Start with some data, you finally give it to the model and the model will learn from it then you will be able to prediction that is the general flow and check more specifically What is meant by giving data to the model. typically if we have a dataset that is useful for prediction then we split the data into two parts. One part is called X and the other part is called Y
X represents everything that is used for prediction and Y is the prediction in which I am interested.

The use case of this is house price prediction, Y contains the house prices and X is the information about the house, When you split data in this fashion the next thing you will do to pass it to the model. The model job to learn the pattern such that we can predict Y using X

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised%20Learning%20with%20scikit_learn/Chapter1-Classification/e8d6ef52-6b2b-4be2-88b9-9860266e9e3a_630x322.jpg"></a>
</p>


# **Logistic regression (LR)**

Def: The idea of logistic regression is to find the relationship between independent variables and the probability of dependent variables. Simply put, it is a classification algorithm used when the response variable is categorical — typically binary (e.g. 0 or 1).

Def: Logistic regression is a type of statistical model employed to examine the relationship between a binary outcome variable (e., yes/no or true/false) and one or more predictor variables. It estimates the probability of a binary outcome based on the values of predictor variables The model outputs a logistic function, transforming the input into a probability range between 0 and 1.

The accuracy of logistic regression with scikit-learn depends on a number of factors, including the quality of the training data, the choice of parameters, and the complexity of the model. In general, logistic regression can achieve good accuracy on simple classification problems. However, it may not be as accurate on complex problems.

**A Simple Example**

Suppose you have patient data and want to predict whether a person is likely to be diagnosed with diabetes. The output is binary: either diagnosed (1) or healthy (0). Similarly:

- Will it rain today? (Yes or No)
- Is this email spam? (Yes or No)

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised%20Learning%20with%20scikit_learn/Chapter1-Classification/5cbd4ed7-1d5a-47fd-9787-0e8d46cbf56c_630x379.jpg"></a>
</p>

  The logistic regression cost function in scikit-learn is the negative log-likelihood function. This function is minimized during the training process to find the parameters of the model that best fit the data.

Once we have extracted the features, we can build our logistic regression model. We will use the sci-kit-learn library in Python to implement logistic regression.

This involves splitting the data into training and testing sets, fitting the model on the training data, and evaluating its performance on the testing data. In scikit-learn, logistic regression is implemented in the LogisticRegression class.

To perform logistic regression with scikit-learn, you can follow these steps:

Import the LogisticRegression class from the sklearn.linear_model module.

- Create an instance of the LogisticRegression class and specify the parameters of the model. 
- The most important parameter is the solver parameter, which specifies the algorithm used to train the model.
-  Other parameters include the penalty parameter, which specifies the type of regularization to use,
-   and the C parameter, which controls the strength of the regularization.

- In this example, we are using the lbfgs solver, and the default value of C. The lbfgs solver is a relatively slow but accurate solver. The C parameter controls the strength of the regularization. A higher value of C will result in a less regularized model, which may be more accurate but also more prone to overfitting.
- Fit the model to the training data.
- Use the model to predict the labels of new data.

Here is an example of how to perform logistic regression with scikit-learn:
<pre> ```

import numpy as np
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.linear_model import LogisticRegression
# Load the iris dataset
iris = load_iris()
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)
clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='ovr')
# Train the model on the training set
clf.fit(X_train, y_train)
# Evaluate the model on the test set
score = clf.score(X_test, y_test)
print("Accuracy:", score)
# Make predictions on new data
X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = clf.predict(X_new)
print("Prediction:", prediction)
 ``` 
</pre>

**Notes**

- Multi-class Classification: If y contains multiple classes, LogisticRegression uses the one-vs-rest (OvR) or multinomial strategy depending on the solver.
- Solvers:
- liblinear: Good for small datasets; supports L1 and L2 regularization.
- lbfgs: Optimized for multiclass problems.
- saga: Handles large datasets; supports L1, L2, and elastic net regularization.
- Hyperparameter Tuning:
- penalty: Regularization type (l1, l2, elasticnet, none).
- C: Inverse of regularization strength (smaller values => stronger regularization)

  1. **Default Parameters and Regularization**

Scikit-learn’s LogisticRegression uses L2 regularization by default with a penalty coefficient (lambda) set to 1. This can be problematic if the data is not normalized. Without proper preprocessing, penalizing parameters may not yield sensible results, as the regularization assumes that features are on a similar scale. Users often overlook the need to standardize or normalize their data before applying the model, which can lead to suboptimal performance and misinterpretation of results

# **Artificial neural networks (ANN)**

Neural networks are a type of machine learning model that can learn complex relationships between input and output variables. They are often used for classification and regression tasks. In scikit-learn, neural networks are implemented in the MLPClassifier and MLPRegressor classes.

Artificial Neural Networks (ANNs) are a subset of machine learning models inspired by the structure and function of the human brain. They consist of interconnected nodes, or neurons, organized in layers.

Learning Process: ANNs learn by adjusting the weights of the connections between neurons based on the error of the output. This process is called training and involves techniques like backpropagation and gradient descent.

The accuracy of a neural network with scikit-learn depends on a number of factors, including the quality of the training data, the choice of parameters, and the complexity of the model. In general, neural networks can achieve good accuracy on complex problems. However, they may be more difficult to train than other machine learning models.
<pre>
import sklearn.neural_network 
# Create an instance of the MLPClassifier class
 neural_network = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu')
 # Fit the model to the training data 
neural_network.fit(X_train, y_train) 
# Predict the labels of new data 
y_pred = neural_network.predict(X_test)
Notes
 </pre>
**Note**
- hidden_layer_sizes: Specifies the number of neurons in each hidden layer (e.g., (100,) for one hidden layer with 100 neurons).
- activation: Sets the activation function (e.g., relu, tanh, logistic).
- solver: Optimization algorithm (e.g., adam, sgd, lbfgs).
- max_iter: Maximum number of iterations for training.
  
# **Support Vector Machine (SVM)**
Support Vector Machine (SVM) is one of the most popular supervised learning algorithms used for classification, regression, and outlier detection tasks. Known for its robustness in handling high-dimensional datasets, SVM is a go-to method for many machine learning practitioners.

support vector machines (SVMs) can be used in Scikit-Learn using the SVC class. This class implements a support vector machine classifier, which is a powerful tool for classification tasks. SVMs can be used for both binary and multiclass classification.

The accuracy of SVM with scikit-learn depends on a number of factors, including the quality of the training data, the choice of parameters, and the complexity of the model. In general, SVM can achieve good accuracy on a variety of classification and regression problems.

The SVM cost function in scikit-learn is the hinge loss function. This function is minimized during the training process to find the parameters of the model that best fit the data.
<pre>
import sklearn.svm 
# Create an instance of the SVC class 
svm = sklearn.svm.SVC(kernel='linear', C=1.0) 
# Fit the model to the training data 
svm.fit(X_train, y_train) 
# Predict the labels of new data 
y_pred = svm.predict(X_test)
 </pre>
 In this example, we are using the linear kernel and the default value of C. The linear kernel is a simple kernel that is suitable for linearly separable data. The C parameter controls the trade-off between the margin and the misclassification penalty. A higher value of C will result in a less regularized model, which may be more accurate but also more prone to overfitting.
 
 **Key Parameters in SVM**
- Kernel: Defines the type of hyperplane used for decision boundaries.
- 'linear': For linearly separable data.
- 'rbf' (default): Radial basis function for non-linear data.
- 'poly': Polynomial kernel.
- C (Regularization Parameter): Controls the trade-off between maximizing the margin and minimizing classification errors. Smaller values allow more misclassifications for a wider margin.
- Gamma: Influences the curvature of the decision boundary in non-linear kernels. Lower values mean broader influence, while higher values mean tighter influence.

  # **Naive Bayes (NB)**

  Naive Bayes is a supervised learning algorithm that is used for classification tasks. It is based on the Bayes theorem, which states that the probability of event A occurring, given that event B has already occurred, is equal to the probability of event A occurring times the probability of event B occurring given that event A has already occurred, divided by the probability of event B occurring.

In scikit-learn, there are three different implementations of the Naive Bayes classifier:
- GaussianNB : This classifier is used for data that is distributed normally.
- MultinomialNB : This classifier is used for data that is counts of occurrences.
- BernoulliNB : This classifier is used for data that is binary (0 or 1).

Here are some of the parameters that can be tuned for the Naive Bayes classifier:

- alpha: The smoothing parameter. This parameter controls how much smoothing is applied to the estimated probabilities.
- fit_prior: Whether to fit the prior probabilities for the classes. If this parameter is set to False, then the prior probabilities will be assumed to be equal.
<pre>
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
 </pre>
 # **KNN (K-Nearest Neighbors)**
 K-Nearest Neighbors is an algorithm for supervised learning. Where the data is ‘trained’ with data points corresponding to their classification. Once a point is to be predicted, it takes into account the ‘K’ nearest points to it to determine it’s classification. The k-nearest neighbors (KNN) algorithm is a non-parametric machine learning algorithm that can be used for both classification and regression tasks. It works by finding the k most similar instances in the training set to a new instance and then predicting the label of the new instance based on the labels of the k nearest neighbors.In scikit-learn, the KNeighborsClassifier class implements the k-nearest neighbors algorithm for classification tasks. The following code shows how to build a KNN classifier with 5 neighbors:

The n_neighbors parameter specifies the number of neighbors to use for the prediction. The default value is 5.

Other parameters that can be tuned for the KNN classifier include:
- metric: The distance metric to use for calculating the distance between neighbors. The default metric is the Euclidean distance.
- weights: The weight function to use for the prediction. The default weight function is uniform, which means that all neighbors are weighted equally.
- algorithm: The algorithm to use for finding the nearest neighbors. The default algorithm is brute-force search.
You can experiment with different values for these parameters to find the best model for your data.

Here is an example of how to use the KNeighborsClassifier class to classify the Iris dataset:
<pre>
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
 </pre>

 # **Decision Tree Classifier**

 The DecisionTreeClassifier class in scikit-learn is a supervised learning algorithm that can be used for classification tasks. It works by constructing a decision tree, which is a flowchart-like structure that maps features to labels.

The DecisionTreeClassifier class has the following parameters:

- criterion: The splitting criterion to use. The default criterion is "gini", which minimizes the Gini impurity. Other possible criteria include "entropy" and "crossentropy".
- max_depth: The maximum depth of the tree.
- min_samples_split: The minimum number of samples required to split a node.
- min_samples_leaf: The minimum number of samples required to be at a leaf node.
- min_impurity_decrease: The minimum decrease in impurity required to split a node.
- random_state: The random seed used for the tree construction.
<pre>
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
</pre> 

# RandomForestClassifier

An ensemble method that builds multiple decision trees and merges them together to improve accuracy and control overfitting. It is robust and effective for a variety of classification tasks.The RandomForestClassifier class in scikit-learn is an ensemble learning algorithm that can be used for both classification and regression tasks. It works by constructing a set of decision trees, each of which is trained on a random subset of the training data. The predictions of the individual trees are then combined to make a final prediction.

<pre>
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
 </pre> 

 The RandomForestClassifier class has the following parameters:

- n_estimators: The number of trees in the forest.
- max_depth: The maximum depth of each tree.
- min_samples_split: The minimum number of samples required to split a node.
- min_samples_leaf: The minimum number of samples required to be at a leaf node.
- min_impurity_decrease: The minimum decrease in impurity required to split a node.
- bootstrap: Whether to use bootstrap sampling.
- oob_score: Whether to calculate the out-of-bag score.
- random_state: The random seed used for the tree construction.

# ExtraTreesClassifier

ExtraTreesClassifier is an ensemble learning algorithm that combines the predictions of multiple decision trees to make more accurate predictions. It is similar to Random Forest, but it uses different randomization strategies to build the trees in the forest. One of the key features of ExtraTreesClassifier is that it splits each feature at a random split point when building a tree. This helps to reduce the bias of the model and makes it more robust to noise in the data. Another key feature of ExtraTreesClassifier is that it uses the entire training set to build each tree. This is in contrast to Random Forest, which randomly samples the training data to build each tree. This helps to improve the accuracy of the model, but it also increases the training time. ExtraTreesClassifier is a powerful machine-learning algorithm that can be used for a variety of classification tasks. It is relatively easy to use, and it is often quite effective.Here are some of the benefits of using ExtraTreesClassifier:

- It is a powerful and robust algorithm that can be used for a variety of classification tasks.
- It is relatively easy to use and tune.
- It is less prone to overfitting than other machine learning algorithms.
- It can be used to identify the most important features in the data.

ExtraTreesClassifier is a good choice for classification tasks where accuracy is important and where the data is noisy or has a large number of features. Here are some examples of tasks where ExtraTreesClassifier can be used:

- Image classification
- Text classification
- Fraud detection
- Medical diagnosis
- Customer segmentation

The n_estimators parameter specifies the number of trees in the forest. You can tune this parameter to improve the performance of the model. Once the model is trained, you can use it to make predictions on new data by calling the predict() method. The predict() method returns an array of predicted class labels.
This is a basic example of how to use ExtraTreesClassifier using scikit-learn. You can tune the model’s parameters to improve its performance, and you can also use it for other classification tasks, such as multiclass classification and regression.

Here are some additional tips for using ExtraTreesClassifier:

- Use the n_estimators parameter to control the number of trees in the forest. A higher value of n_estimators will generally improve the model's performance, but it will also increase the training time.
- Use the max_depth parameter to control the depth of each tree in the forest. A higher value of max_depth will generally improve the model's performance, but it will also increase the risk of overfitting.
- Use the min_samples_split parameter to control the minimum number of samples required to split a node in a tree. A higher value of min_samples_split will make the model more robust to noise, but it will also make it more difficult to learn complex relationships in the data.
- Use the min_samples_leaf parameter to control the minimum number of samples required in a leaf node of a tree. A higher value of min_samples_leaf will make the model more robust to noise, but it will also make it more difficult to learn complex relationships in the data.
- You can also use the feature_importances_ attribute of the ExtraTreesClassifier object to identify the most important features for the model. This can be useful for feature selection and for understanding how the model works.

Here is an example of how to build an ExtraTreesClassifier using scikit-learn:
<pre>
from sklearn.ensemble import ExtraTreesClassifier
# Create an instance of the ExtraTreesClassifier class
clf = ExtraTreesClassifier(n_estimators=100)
# Fit the model to the training data
clf.fit(X_train, y_train)
# Make predictions on the test data
y_pred = clf.predict(X_test)
# Make predictions on the test data
y_pred = clf.predict(X_test)
# Print the predicted class labels
print(y_pred)
  </pre> 
  
# Gradient Boosting Trees (GBT)
  
To perform classification using Gradient Boosting Trees (GBT) using scikit-learn, you can follow these steps:
<pre>
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
# Load the training data 
X_train = pd.read_csv("train_data.csv")
y_train = X_train["target"]
# Load the test data
 X_test = pd.read_csv("test_data.csv")
# Create a GradientBoostingClassifier object
 clf = GradientBoostingClassifier(n_estimators=100)
# Fit the model to the training data 
clf.fit(X_train, y_train)
# Make predictions on the test data 
y_pred = clf.predict(X_test)
# Evaluate the model's performance 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) 
print("Accuracy:", accuracy)
  </pre> 
Here are some additional tips for using GradientBoostingClassifier:

Use the n_estimators parameter to control the number of trees in the forest. A higher value of n_estimators will generally improve the model's performance, but it will also increase the training time.

Use the max_depth parameter to control the depth of each tree in the forest. A higher value of max_depth will generally improve the model's performance, but it will also increase the risk of overfitting.

Use the learning_rate parameter to control the amount of weight given to each tree in the forest. A higher value of learning_rate will cause the model to learn more quickly, but it will also make it more prone to overfitting.

Use the subsample parameter to control the fraction of the training data used to train each tree in the forest. A lower value of subsample will reduce the model's overfitting, but it will also increase the training time.

You can also use the feature_importances_ attribute of the GradientBoostingClassifier object to identify the most important features for the model. This can be useful for feature selection and for understanding how the model works.

Gradient Boosting Trees is a powerful machine learning algorithm that can be used to solve a wide variety of classification problems. It is relatively easy to use and tune, and it is often quite effective.
 
### References


1-[sklearn-cheat](https://github.com/thegeekyb0y/sklearn-cheat?tab=readme-ov-file#logisticregression)




<p align="center">
  <a href="#previous-section" style="text-decoration:none;">
    <button style="padding:20px 40px; font-size:24px; font-weight:bold; border-radius:12px; background-color:#007BFF; color:white; border:none; cursor:pointer;">
      ⬅️ Previous
    </button>
  </a>

  <a href="#next-section" style="text-decoration:none;">
    <button style="padding:20px 40px; font-size:24px; font-weight:bold; border-radius:12px; background-color:#28A745; color:white; border:none; cursor:pointer;">
      Next ➡️
    </button>
  </a>
</p>



# 📘 NLP Concepts Quiz  
Test your knowledge of **Natural Language Processing (NLP)** based on the provided materials.  
Choose the best answer for each question.  

---

### 1. What is the primary goal of Natural Language Processing (NLP)?  
- a) To work with numerical values and spreadsheets.  
- b) To make computers understand, interpret, and manipulate human language.  
- c) To create hand-coded rules for language analysis.  
- d) To model the hierarchical structure of computer code.  

---

### 2. NLP encompasses two key areas. What are they?  
- a) Data Labeling and Text Analysis.  
- b) Natural Language Understanding (NLU) and Natural Language Generation (NLG).  
- c) Rule-Based Approaches and Statistical Approaches.  
- d) Machine Learning and Linguistics.  

---

### 3. Which of the following is an example of Natural Language Generation (NLG)?  
- a) Recognizing the user's intent when they ask "What's the weather like?".  
- b) An AI writing assistant crafting a paragraph based on provided data.  
- c) Classifying a news article as "sports" or "politics".  
- d) Extracting a person's name from a document.  

---

### 4. The 'Deep Learning Era' in NLP began in which decade?  
- a) 1950s-1960s.  
- b) 1970s-1980s.  
- c) 1990s-2000s.  
- d) 2010s-present.  

---

### 5. The task of analyzing text to determine if the emotion is positive, negative, or neutral is called:  
- a) Named Entity Recognition.  
- b) Document Summarization.  
- c) Sentiment Analysis.  
- d) Machine Translation.  

---

### 6. True or False: The earliest work in NLP relied on machine learning and statistical models.  
- a) True  
- b) False  

---

### 7. Which NLP application is used to automatically translate text from one language to another, like Google Translate does?  
- a) Question Answering.  
- b) Machine Translation.  
- c) Document Clustering.  
- d) Keyword Extraction.  















































































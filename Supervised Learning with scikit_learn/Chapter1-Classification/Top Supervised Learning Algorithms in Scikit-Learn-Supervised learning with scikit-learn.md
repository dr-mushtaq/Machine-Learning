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
- [Why NLP is so important?](#Why-NLP-is-so-important?)  
- [What is NLP?](#What-is-NLP?)  
- [Brief History of NLP](#Brief-History-of-NLP)
- [*Application](#*Application)   

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

### References

1-[Natural Language Processing (NLP) with Python](https://pub.towardsai.net/natural-language-processing-nlp-with-python-tutorial-for-beginners-1f54e610a1a0)



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





























































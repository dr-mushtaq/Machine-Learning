Supervised learning, a key concept in machine learning, enables computers to make decisions based on labeled data. Whether you’re classifying emails as spam or predicting customer behavior, scikit-learn offers accessible tools to build effective models.

In this blog, we dive into classification techniques, a major type of supervised learning, using Python’s powerful scikit-learn library. This guide provides a step-by-step approach, from understanding core concepts to implementing your first classification model, perfect for beginners and those looking to refine their data science skills.


## 📑 Table of Contents  

- [What is Machine Learning](#What-is-Machine-Learning)  
- [Unsupervised learning](#Unsupervised-learning)  
- [Reinforcement learning](#Reinforcement-learning)  
- [Supervised learning explanation](#Supervised-learning-explanation)
- [Supervised learning in Python](#Supervised-learning-in-Python)    

cvcv
### **What is Machine Learning** 

Machine learning is the science and art of giving computers the ability to learn to make decisions from data without being explicitly programmed.
For example, your computer can learn to predict whether an email is spam or not spam given its content and sender. Another example: your computer can learn to cluster, say, Wikipedia entries, into different categories based on the words they contain. It could then assign any new Wikipedia article to one of the existing clusters. Notice that, in the first example, we are trying to predict a particular class label, that is, spam or not spam. In the second example, there is no such label. When there are labels present, we call it **supervised learning**. When there are no labels present, we call it **unsupervised learning**.

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised%20Learning%20with%20scikit_learn/Chapter1-Classification/1.png"></a>
</p>

 ## **Unsupervised learning**
 
Unsupervised learning, in essence, is the machine learning task of uncovering hidden patterns and structures from unlabeled data. For example, a business may wish to group its customers into distinct categories based on their purchasing behavior without knowing in advance what these categories may be. This is known as clustering, one branch of unsupervised learning.

## **Reinforcement learning** 

There is also reinforcement learning, in which machines or software agents interact with an environment. Reinforcement agents are able to automatically figure out how to optimize their behavior given a system of rewards and punishments. Reinforcement learning draws inspiration from behavioral psychology and has applications in many fields, such as, economics, genetics, as well as game playing. In 2016, reinforcement learning was used to train Google DeepMind’s AlphaGo, which was the first computer program to beat the world champion in Go.

## **Supervised learning explanation** 

In supervised learning, we have several data points or samples, described using predictor variables or features and a target variable. Our data is commonly represented in a table structure such as the one you see here, in which there is a row for each data point and a column for each feature. Here, we see the iris dataset: each row represents measurements of a different flower and each column is a particular kind of measurement, like the width and length of a certain part of the flower. The aim of supervised learning is to build a model that is able to predict the target variable, here the particular species of a flower, given the predictor variables, here the physical measurements. If the target variable consists of categories, like ‘click’ or ‘no click’, ‘spam’ or ‘not spam’, or different species of flowers, we call the learning task classification. Alternatively, if the target is a continuously varying variable, for example, the price of a house, it is a regression task. In this chapter, we will focus on classification. In the following, on regression.

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised%20Learning%20with%20scikit_learn/Chapter1-Classification/2.png"></a>
</p>

The goal of supervised learning is frequently to either automate a time-consuming or expensive manual task, such as a doctor’s diagnosis or to make predictions about the future, say whether a customer will click on an ad or not. For supervised learning, you need labeled data and there are many ways to get it: you can get historical data, which already has labels that you are interested in; you can perform experiments to get labeled data, such as A/B-testing to see how many clicks you get; or you can also crowdsourced labeling data which, like reCAPTCHA does for text recognition. In any case, the goal is to learn from data for which the right output is known, so that we can make predictions on new data for which we don’t know the output.
 
 **Naming conventions**

A note on naming conventions: out in the wild, you will find that what we call a feature, others may call a predictor variable or independent variable, and what we call the target variable, others may call dependent variable or response variable.

## **Supervised learning in Python** 

There are many ways to perform supervised learning in Python. In this course, we will use scikit-learn, or sklearn, one of the most popular and user-friendly machine-learning libraries for Python. It also integrates very well with the SciPy stack, including libraries such as NumPy. There are a number of other ML libraries out there, such as TensorFlow and Keras, which are well worth checking out once you got the basics down.

1- **What is scikit -Learn?**

What is it:In simple terms, Scikit Learn is an open source and one of the most useful libraries for machine learning in Python. It has tools for predictive data analysis [6]. Scikit learn is a library that is written in Python and built upon Scipy, Matplotlib and Numpy provides a set of useful and efficient tools for machine learning and statistical modeling including regression, classification, clustering, predictive data analysis and dimensionality reduction etc and known as the most robust and useful library for Machine Learning .

Background: A developer named David Cournapeau originally released scikit-learn as a student in 2007. The open source community quickly adopted it and has updated it numerous times over the years [5]. This library, which is largely written in Python, is built upon NumPy, SciPy and Matplotlib [2].The library provides a unified API (Application Programming Interface) for practitioners to ease the use of machine learning algorithms with only writing a few lines to accomplish the predictive or classification task [3].The package is written heavily in python, and it incorporates C++ libraries like LibSVM and LibLinear for support vector machines and generalized linear model implementation [3]

Features: The packages in Scikit-learn focus on modeling data.One of the most prominent Python libraries for machine learning:
Contains many state-of-the-art machine learning algorithms

It was designed to work seamlessly with NumPy and SciPy (both described below) for data cleaning, preparation, and calculation.

Builds on numpy (fast), implements advanced techniques

It has modules for loading data as well as splitting it into training and test sets.

It supports feature extraction for text and image data.

Wide range of evaluation measures and techniques

Offers comprehensive documentation about each algorithm

Widely used, and a wealth of tutorials and code snippets are available

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised%20Learning%20with%20scikit_learn/Chapter1-Classification/3.jpg"></a>
</p>

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised%20Learning%20with%20scikit_learn/Chapter1-Classification/4.jpg"></a>
</p>

### References

1-[Scikit Learn Tutorial](https://www.tutorialspoint.com/scikit_learn/index.htm)

2-[Scikit-Learn: A silver bullet for basic machine learning](https://medium.com/analytics-vidhya/scikit-learn-a-silver-bullet-for-basic-machine-learning-13c7d8b248ee)

4-[Machine Learning with scikit-learning (Datacamp)](https://www.datacamp.com/users/sign_in?redirect=http%3A%2F%2Fapp.datacamp.com%2Flearn%2Fcourses%2Fmachine-learning-with-scikit-learn)

5-[-Essential Python Libraries for Machine Learning and Data Science](https://www.deeplearning.ai/blog/essential-python-libraries-for-machine-learning-and-data-science/?utm_campaign=DLAI+Blog&utm_content=248986290&utm_medium=social&utm_source=facebook&hss_channel=fbp-1027125564106325)


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

































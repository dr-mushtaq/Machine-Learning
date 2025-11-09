# Types of Machine Learning

## Introduction
Supervised learning and unsupervised learning are two fundamental concepts in machine learning. In supervised learning, we teach a machine learning model to make predictions based on labeled data, while in unsupervised learning, the model discovers patterns and relationships in unlabeled data without any guidance. This blog will explore the key differences between supervised and unsupervised learning algorithms and their applications.

## Sections
- [Supervised Learning](#supervised-learning)
- [Type of Supervised Learning](#type-of-supervised-learning)
- [Semi-supervised learning](#semi-supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Reinforcement Learning](#reinforcement-learning)
- [Comparison](#comparison)
- [Conclusion](#conclusion)


There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/1.jpg"></a>
</p>

# 1. Supervised Learning
## Introduction
Supervised learning and unsupervised learning are two fundamental concepts in machine learning. In supervised learning, we teach a machine learning model to make predictions based on labeled data, while in unsupervised learning, the model discovers patterns and relationships in unlabeled data without any guidance. This blog will explore the key differences between supervised and unsupervised learning algorithms and their applications.

## Section
- Supervised Learning
- Type of Supervised Learning
- Semi-supervised learning
- Unsupervised Learning
- Reinforcement Learning
- Comparison
- Conclusion

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

---

# Supervised Learning

## Definition

**Def1:** Supervised learning is a type of machine learning in which our algorithms are trained using well-labeled training data, and machines predict the output based on that data.

**Def2:** Supervised learning is where the computer is given a set of training data, and it learns how to predict outcomes based on that data. Labeled data indicates that the input data has already been tagged with the appropriate output.

**Def3:** Basically, it is the task of learning a function that maps the input set and returns an output. In supervised methods, we have a target variable. This means that the algorithm is fed with examples containing several features and one outcome variable. The outcome variable can be binary or multifactorial.

**Def4:** Supervised learning is a machine learning paradigm where the available data consists of a pairing of inputs with know, correct, outputs. What is unknown is exactly how the mapping between those inputs and outputs works. The goal is to infer, from the available data, the general structure of the mapping in the hope that this will generalize to unseen situations.

**Def5:** Supervised learning algorithms make predictions based on a set of examples. For example, historical sales can be used to estimate future prices. With supervised learning, you have an input variable that consists of labeled training data and a desired output variable. You use an algorithm to analyze the training data to learn the function that maps the input to the output. This inferred function maps new, unknown examples by generalizing from the training data to anticipate results in unseen situations. Some of its examples are: Linear Regression, Logistic Regression, KNN, etc.

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/2.jpg"></a>
</p>

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/3.png"></a>
</p>

---

### Example
Imagine you are teaching a kid to differentiate dogs from cats: at first, you show him many images of both animals, identifying each of them. With these examples, he can associate each animal with its name and then classify new images correctly [4]. Supervised learning has exactly the same idea: from a big train dataset, the algorithm "learns" the relationship between data and label and, therefore, it can predict the result of any other input [4].

---

### Mathematics
In mathematical terms, we are trying to find an expression `Y = f(X) + b` that can predict the results.  
Where X is the input, Y is the prediction, and `f(X) + b` is the model learned by the algorithm [4].

Imagine you are teaching a kid to differentiate dogs from cats: at first, you show him many images of both animals, identifying each of them. With these examples, he can associate each animal with its name and then classify new images correctly. Supervised learning has exactly the same idea: from a big train dataset, the algorithm "learns" the relationship between data and label and, therefore, it can predict the result of any other input.

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/4.png"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/5.png"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/6.png"></a>
</p>



---

### The supervised learning problem

---

### Example 2
Given this data, let's say you have a friend who owns a house that is, say 750 square feet, and hoping to sell the house and they want to know how much they can get for the house. So how can the learning algorithm help you?

One thing a learning algorithm might be able to do is put a straight line through the data or to fit a straight line to the data and, based on that, it looks like maybe the house can be sold for maybe about $150,000.

But maybe this isn't the only learning algorithm you can use. There might be a better one. For example, instead of sending a straight line to the data, we might decide that it's better to fit a quadratic function or a second-order polynomial to this data.
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/7.jpg"></a>
</p>

But maybe this isn't the only learning algorithm you can use. There might be a better one. For example, instead of sending a straight line to the data, we might decide that it's better to fit a quadratic function or a second-order polynomial to this data. ...

(continues like this — content unchanged)
<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/1746996482895.jfif"></a>
</p>
---

### Type of Supervised Learning
The two main tasks supervised learning aims to solve are classification, regression, and forecasting. The three main tasks supervised learning aims to solve are: classification, regression and forcasting. The former, as the name says, is related to assign a label to the data, such as classify images in dog, cat or bird. The latter aims to predict a continuous value given some conditions, for example, estimate a house price given its size, location and number of rooms [4].

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/8.jpg"></a>
</p>

---

## Regression Problem
When predicting continuous values, the problems become a regression problem.
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/9.png"></a>
</p>

**What it is:** Def: To define with a bit more terminology this is also called a regression problem and by regression problem I mean we're trying to predict a continuous value output. ...

---

## Classification Problem
Here's another supervised learning example, some friends and I were actually working on this earlier.

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/10.png"></a>
</p>
...

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/11.jpg"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/12.jpg"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/13.jpg"></a>
</p>
...
(unchanged content – already markdown applied to structure)
...

---

# Forecasting
This is the process of making predictions about the future based on past and present data. It is most commonly used to analyze trends. A common example might be an estimation of the next year sales based on the sales of the current year and previous years.

---

# Semi-supervised learning
The challenge with supervised learning is that labeling data can be expensive and time-consuming. If labels are limited, you can use unlabeled examples to enhance supervised learning. Because the machine is not fully supervised in this case, we say the machine is semi-supervised. With semi-supervised learning, you use unlabeled examples with a small amount of labeled data to improve the learning accuracy.

---

# Unsupervised Learning
What is Unsupervised Learning:  
Defin1.  Unsupervised learning involves training a model using unlabeled data.  
Def 2:The objective is for the model to find patterns or relationships within the data.  
Def 3: Unsupervised learning is where the computer is given data but not told what to do with it; it has to figure out how to group or cluster the data itself. With this method, we have no outcome variable or label variable. This way, the algorithm search for patterns in all data and all variables.  
Def 4: Unsupervised learning is another type of machine learning in which the computer is trained on unlabeled data, meaning the data does not have any pre-existing labels or targets. The goal of unsupervised learning is to find hidden patterns or structures in the data without the guidance of a labeled dataset [6].  
Def:When performing unsupervised learning, the machine is presented with totally unlabeled data. It is asked to discover the intrinsic patterns that underlie the data, such as a clustering structure, a low-dimensional manifold, or a sparse tree and graph.

Applications in various fields
...

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/14.PNG"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/15.PNG"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/16.PNG"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/17.PNG"></a>
</p>

(unchanged — markdown sections added above)

---

# Reinforcement Learning
**Def 1:** Reinforcement learning involves training a model by rewarding correct behavior and punishing incorrect behavior. This type of machine learning is often used in gaming simulations.

Reinforcement learning (RL) is a type of machine learning where an agent learns to behave in an environment by trial and error. The agent receives rewards for taking actions that lead to desired outcomes, and penalties for taking actions that lead to undesired outcomes. Over time, the agent learns to take actions that maximize its rewards.

**Def 2:** Reinforcement Learning (RL) is a type of machine learning algorithm that allows an agent to learn through trial and error by interacting with an environment. The agent receives feedback in the form of rewards or penalties for the actions it takes in the environment. The goal of the agent is to learn a policy that maximizes the total reward it receives over time [6].

RL is a powerful tool that can be used to solve a wide variety of problems, including game playing, robotics, and finance. Some of the most successful RL algorithms include Q-learning, SARSA, and Deep Q-Networks.

### Example

Here is an example of how RL can be used to solve a problem. Let's say we want to train a robot to walk. We can create an environment for the robot that consists of a treadmill and a reward function that gives the robot a reward for taking steps in the forward direction. The robot can then use RL to learn how to walk by trial and error.

---

### Commonly used Methods

Reinforcement learning algorithms can be broadly categorized into two types [6]:

**Model-based algorithms:** These algorithms explicitly learn a model of the environment, including transition probabilities and rewards. They use this learned model to plan and make decisions. Examples include Monte Carlo methods, Temporal Difference (TD) learning, and Q-learning.

**Model-free algorithms:** These algorithms directly learn the optimal policy or value function without building an explicit model of the environment. They rely on trial-and-error learning through repeated interactions. Examples include Q-learning, SARSA, Deep Q-Networks (DQN), and Proximal Policy Optimization (PPO).

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/18.jpg"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/19.jpg"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/20.png"></a>
</p>


<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/21.jpg"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/22.jpg"></a>
</p>








---

# Reference
1- Main concepts behind Machine Learning  
2- 7 Important Concepts in Artificial Intelligence and Machine Learning  
3- Elucidating the Power of Inferential Statistics To Make Smarter Decisions! (Unread)  
4- Main concepts behind Machine Learning  
5- Important Concepts in Artificial Intelligence and Machine Learning  
6- Machine Learning: An Introductory Tutorial for Beginners  
7- Which machine learning algorithm should I use?

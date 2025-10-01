Artificial Intelligence (AI) has transformed from a visionary concept into an indispensable technology reshaping our world. From its philosophical roots and early computations to modern machine learning breakthroughs, AI now powers everyday tools—voice assistants, recommendation systems, and cutting-edge healthcare diagnostics. In this post, we’ll explore why humans created AI, trace its evolution from symbolic logic to neural networks, clarify what AI truly means today, and showcase real-world applications driving innovation across industries.
## 📑 Table of Contents  

- [Why we used AI](#Why-we-used-AI)  
- [Unsupervised learning](#Unsupervised-learning)  
- [Reinforcement learning](#Reinforcement-learning)  
- [Supervised learning explanation](#Supervised-learning-explanation)
- [Supervised learning in Python](#Supervised-learning-in-Python)    


## **Why we used AI** 

Artificial Intelligence is an exciting scientific discipline that studies how we can make computers exhibit intelligent behavior, e.g. do those things that human beings are good at doing.
Originally, computers were invented by **Charles Babbage** to operate on numbers following a well-defined procedure - an algorithm. Modern computers, even though significantly more advanced than the original model proposed in the 19th century, still follow the same idea of controlled computations. Thus it is possible to program a computer to do something if we know the exact sequence of steps that we need to do in order to achieve the goal.

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/dsh_age.png"></a>
</p>

✅ Defining the age of a person from his or her photograph is a task that cannot be explicitly programmed, because we do not know how we come up with a number inside our head when we do it.

There are some tasks, however, that we do not explicitly know how to solve. Consider determining the age of a person from his/her photograph. We somehow learn to do it, because we have seen many examples of people of different age, but we cannot explicitly explain how we do it, nor can we program the computer to do it. This is exactly the kind of task that are of interest to Artificial Intelligence (AI for short).

✅ Think of some tasks that you could offload to a computer that would benefit from AI. Consider the fields of finance, medicine, and the arts - how are these fields benefiting today from AI?

In traditional programming, when aiming to implement new functionalities or automate tasks, software development is typically required. This involves writing code containing a predetermined set of instructions, such as if-then-else statements, to direct the computer’s actions. Consequently, to accomplish a variety of tasks, a corresponding number of rules must be provided to the computer, posing a significant challenge. This limitation highlights that conventional programming approaches lack generalizability.

if you haven’t done it by yourself, requires laying out in excruciating detail every single step that you want the computer to do in order to achieve your goal. Now, if you want to do something that you don’t know how to do, then this is going to be a great challenge. Basically, regular programming is pretty limited and can’t make decisions on its own. That’s why we need generalized programming, which is more than just a programmer and can make decisions from our perspective.

So, basically, this Arthur Samuel had a challenge in 1956. He wanted to teach a computer to beat him at checkers. Like, how do you even do that? Well, he came up with a plan. He had the computer play against itself over and over again, like thousands of times, until it learned how to play checkers really well. And guess what? It actually worked! By 1962, the computer had even beaten the Connecticut state champion. Pretty impressive, right?

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/1_fzqAsLGKIt3jzstwBLGsJA%20(1).png"></a>
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

## Why we used AI
Artificial Intelligence is an exciting scientific discipline that studies how we can make computers exhibit intelligent behavior, e.g. do those things that human beings are good at doing.

Originally, computers were invented by Charles Babbage to operate on numbers following a well-defined procedure - an algorithm. Modern computers, even though significantly more advanced than the original model proposed in the 19th century, still follow the same idea of controlled computations. Thus it is possible to program a computer to do something if we know the exact sequence of steps that we need to do in order to achieve the goal. 

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


### Example 2
Given this data, let's say you have a friend who owns a house that is, say 750 square feet, and hoping to sell the house and they want to know how much they can get for the house. So how can the learning algorithm help you?

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/7.jpg"></a>
</p>

One thing a learning algorithm might be able to do is put a straight line through the data or to fit a straight line to the data and, based on that, it looks like maybe the house can be sold for maybe about $150,000.

But maybe this isn't the only learning algorithm you can use. There might be a better one. For example, instead of sending a straight line to the data, we might decide that it's better to fit a quadratic function or a second-order polynomial to this data, we might decide that it's better to fit a quadratic function or a second-order polynomial to this data. And if you do that, and make a prediction here, then it looks like, well, maybe we can sell the house for closer to $200,000.

One of the things we'll talk about later is how to choose and how to decide do you want to fit a straight line to the data or do you want to fit the quadratic function to the data and there's no fair picking whichever one gives your friend the better house to sell. But each of these would be a fine example of a learning algorithm.

So this is an example of a supervised learning algorithm. And the term supervised learning refers to the fact that we gave the algorithm a data set in which the "right answers" were given. That is, we gave it a data set of houses in which for every example in this data set, we told it what is the right price so what is the actual price that, that house sold for and the toss of the algorithm was to just produce more of these right answers such as for this new house, you know, that your friend may be trying to sell.

---

### Type of Supervised Learning
The two main tasks supervised learning aims to solve are classification, regression, and forecasting. The three main tasks supervised learning aims to solve are: classification, regression and forcasting. The former, as the name says, is related to assign a label to the data, such as classify images in dog, cat or bird. The latter aims to predict a continuous value given some conditions, for example, estimate a house price given its size, location and number of rooms [4].

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/1746996482895.jfif"></a>
</p>

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/8.jpg"></a>
</p>

---

## 1.1 Regression Problem
When predicting continuous values, the problems become a regression problem.
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/9.png"></a>
</p>

**What it is: Def**: To define with a bit more terminology this is also called a regression problem and by regression problem I mean we're trying to predict a continuous value output. Namely the price. So technically I guess prices can be rounded off to the nearest cent. So maybe prices are actually discrete values, but usually we think of the price of a house as a real number, as a scalar value, as a continuous value number and the term regression refers to the fact that we're trying to predict the sort of continuous values attribute.

---

## 1.2 Classification Problem
Here's another supervised learning example, some friends and I were actually working on this earlier. Let's see you want to look at medical records and try to predict of a breast cancer as malignant or benign. If someone discovers a breast tumor, a lump in their breast, When the data are being used to predict a categorical variable, supervised learning is also called classification. This is the case when assigning a label or indicator, either dog or cat to an image. When there are only two labels, this is called binary classification. When there are more than two categories, the problems are called multi-class classification.

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/10.png"></a>
</p>
a malignant tumor is a tumor that is harmful and dangerous and a benign tumor is a tumor that is harmless.
So obviously people care a lot about this. Let's see a collected data set and suppose in your data set you have on your horizontal axis the size of the tumor and on the vertical axis.

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/11.jpg"></a>
</p>
I'm going to plot one or zero, yes or no, whether or not these are examples of tumors we've seen before are malignant which is one or zero if not malignant or benign. So let's say our data set looks like this where we saw a tumor of this size that turned out to be benign. One of this size, one of this size. And so on. And sadly we also saw a few malignant tumors, one of that size, one of that size, one of that size... So on. So this example... I have five examples of benign tumors shown down here, and five examples of malignant tumors shown with a vertical axis value of one.

And let's say we have a friend who tragically has a breast tumor, and let's say her breast tumor size is maybe somewhere around this value. The machine learning question is, can you estimate what is the probability, what is the chance that a tumor is malignant versus benign?

To introduce a bit more terminology this is an example of a classification problem. The term classification refers to the fact that here we're trying to predict a discrete value output: zero or one, malignant or benign.

And it turns out that in classification problems sometimes you can have more than two values for the two possible values for the output. As a concrete example maybe there are three types of breast cancers and so you may try to predict the discrete value of zero, one, two, or three with zero being benign. Benign tumor, so no cancer. And one may mean, type one cancer, like, you have three types of cancer, whatever type one means. And two may mean a second type of cancer, a three may mean a third type of cancer. But this would also be a classification problem, because this other discrete value set of output corresponding to, you know, no cancer, or cancer type
one, or cancer type two, or cancer type three.


<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/12.jpg"></a>
</p>

In classification problems there is another way to plot this data. Let me show you what I mean. Let me use a slightly different set of symbols to plot this data. So if tumor size is going to be the attribute that I'm going to use to predict malignancy or benignness, I can also draw my data like this. I'm going to use different symbols to denote my benign and malignant, or my negative and positive examples.

So instead of drawing crosses, I'm now going to draw O's for the benign tumors. Like so. And I'm going to keep using X's to denote my malignant tumors.

Okay? I hope this is beginning to make sense. All I did was I took, you know, these, my data set on top and I just mapped it down. To this real line like so. And started to use different symbols, circles, and crosses, to denote malignant versus benign examples.

Now, in this example, we use only one feature or one attribute, mainly, the tumor size in order to predict whether the tumor is malignant or benign. In other machine learning problems when we have more than one feature, more than one attribute.

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/13.jpg"></a>
</p>
Here's an example. Let's say that instead of just knowing the tumor size, we know both the age of the patients and the tumor size. In that case maybe your data set will look like this where I may have a set of patients with those ages and that tumor size and they look like this. And a different set of patients, they look a little different, whose tumors turn out to be malignant, as denoted by the crosses.

So, let's say you have a friend who tragically has a tumor. And maybe, their tumor size and age falls around there. So given a data set like this, what the learning algorithm might do is throw the straight line through the data to try to separate out the malignant tumors from the benign ones and, so the learning algorithm may decide to throw the straight line like that to separate out the two classes of tumors. And. You know, with this, hopefully you can decide that your friend's tumor is
more likely to if it's over there, that hopefully your learning algorithm will say that your friend's tumor falls on this benign side and is therefore more likely to be benign than malignant.

In this example we had two features, namely, the age of the patient and the size of the tumor. In other machine learning problems we will often have more features, and my friends that work on this problem, they actually use other features like these, which is clump thickness, the clump thickness of the breast tumor. Uniformity of cell size of the tumor. Uniformity of cell shape of the tumor, and so on, and other features as well. And it turns out one of the interes-, most interesting learning algorithms that we'll see in this class is a learning algorithm that can deal with, not just two or three or five features, but an infinite number of features. On this slide, I've
listed a total of five different features. Right, two on the axes and three more up here.
But it turns out that for some learning problems, what you really want is not to use, like, three or five features. But instead, you want to use an infinite number of features, an infinite number of attributes, so that your learning algorithm has lots of attributes or features or cues with which to make those predictions. So how do you deal with an infinite number of features? How do you even store an infinite number of things on the computer when your computer is gonna run out of memory. It turns out that when we talk about an algorithm called the Support Vector Machine, there will be a neat mathematical trick that will allow a computer to deal with an infinite number of features. Imagine that I didn't just write down two features here and three features on the right. But, imagine that I wrote down an infinitely long list, I just kept writing more and more and more features. Like an infinitely long list of features. Turns out, we'll be able to come up with an algorithm that can deal with that.

So, just to recap. In this class we'll talk about supervised learning. And the idea is that, in supervised learning, in every example in our data set, we are told what is the "correct answer" that we would have quite liked the algorithms have predicted on that example. Such as the price of the house, or whether a tumor is malignant or benign. We also talked about the regression problem. And by regression, that means that our goal is to predict a continuous-valued output. And we talked about the classification problem, where the goal is to predict a discrete value output.
---

# 1.3 Forecasting
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

**Applications in various fields**
So we're given the data set and we're not told what to do with it and we're not told what each data point is. Instead we're just told, here is a data set. Can you find some structure in the data? Given this data set, an Unsupervised Learning algorithm might decide that the data lives in two different clusters. And so there's one cluster and there's a different cluster. And yes, the Supervised Learning algorithm may break these data into these two separate clusters. So this is called a clustering algorithm. And this turns out to be used in many places. 

...

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/14.PNG"></a>
</p>
One example where clustering is used is in Google News and if you have not seen this before, you can actually go to this URL news.google.com to take a look. What Google News does is everyday it goes and looks at tens of thousands or hundreds of thousands of new stories on the web and it groups them into cohesive news stories. For example, let's look here. The URLs here link to different news stories about the BP Oil Well story. So, let's click on one of these URL's and we'll click on one of these URL's. 

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/15.PNG"></a>
</p>

This second application is on social network analysis. So given knowledge about which friends you email the most or given your Facebook friends or your Google+ circles, can we automatically identify which are cohesive groups of friends, also which are groups of people that all know each other? Market segmentation. Many companies have huge databases of customer information. So, can you look at this customer data set and automatically discover market segments and automatically group your customers into different market segments so that you can automatically and more efficiently sell or market your different market segments together? Again, this is Unsupervised Learning because we have all this customer data, but we don't know in advance what are the market segments and for the customers in our data set, you know, we don't know in advance who is in market segment one, who is in market segment two, and so on. But we have to let the algorithm discover all this just from the data. Finally, it turns out that Unsupervised Learning is also used for surprisingly astronomical data analysis and these clustering algorithms gives surprisingly interesting useful theories of how galaxies are born. All of these are examples of clustering, which is just one type of Unsupervised Learning. Let me tell you about another one
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/16.PNG"></a>
</p>

I'm gonna tell you about the cocktail party problem. So, you've been to cocktail parties before, right? Well, you can imagine there's a party, room full of people, all sitting around, all talking at the same time and there are all these overlapping voices because everyone is talking at the same time, and it is almost hard to hear the person in front of you. So maybe at a\ cocktail party with two people, two people talking at the same

time, and it's a somewhat small cocktail party. And we're going to put two microphones in the room so there are microphones, and because these microphones are at two different distances from the speakers, each microphone records a different combination of these two speaker voices. Maybe speaker one is a little louder in microphone one and maybe speaker two is a

little bit louder on microphone 2 because the 2 microphones are at different positions relative to the 2 speakers, but each microphone would cause an overlapping combination of both speakers' voices. So here's an actual recording of two speakers recorded by a researcher.
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/17.PNG"></a>
</p>

Let me play for you the first, what the first microphone sounds like. One (uno), two (dos), three (tres), four (cuatro), five (cinco), six (seis), seven (siete), eight (ocho), nine (nueve), ten (y diez). All right, maybe not the most interesting cocktail party, there's two people counting from one to ten in two languages but you know. What you just heard was the first microphone recording, here's the second recording. Uno (one), dos (two), tres (three), cuatro (four), cinco (five), seis (six), siete (seven), ocho (eight), nueve (nine) y diez (ten). So we can do, is take these two microphone recorders and give them to an Unsupervised Learning algorithm called the cocktail party algorithm, and tell the algorithm - find structure in this data for you. And what the algorithm will do is listen to these

audio recordings and say, you know it sounds like the two audio recordings are being added together or that have being summed together to produce these recordings that we had. Moreover, what the cocktail party algorithm will do is separate out these two audio sources that were being added or being
summed together to form other recordings and, in fact, here's the first output of the cocktail party algorithm. One, two, three, four, five, six, seven, eight, nine, ten. So, I separated out the English voice in one of the recordings. And here's the second of it. Uno, dos, tres, quatro, cinco, seis, siete, ocho, nueve y diez. Not too bad, to give you one more example, here's another

recording of another similar situation, here's the first microphone :  One, two, three, four, five, six, seven, eight, nine, ten. OK so the poor guy's gone home from the cocktail party and he 's now sitting in a room by himself talking to his radio. Here's the second microphone recording. One, two, three, four, five, six, seven, eight, nine, ten. When you give these two microphone recordings to the same algorithm, what it does, is again say, you know, it sounds like there are two audio sources, and moreover, the album says, here is the first of the audio sources I found. One, two, three, four, five, six, seven, eight, nine, ten. So that wasn't perfect, it got the voice, but it also got a little bit of the music in there. Then here's the second output to the algorithm. Not too bad, in that second output it managed to get rid of the voice entirely. And just, you know, cleaned up the music, got rid of the counting from one to ten.

**Right programmingProgramming language for Machin Learning**
So you might look at an Unsupervised Learning algorithm like this and ask how complicated this is to implement this, right? It seems like in order to, you know, build this application, it seems like to do this audio processing you need to write a ton of code or maybe link into like a bunch of synthesizer Java libraries that process audio, seems like a really complicated program, to do this audio, separating out audio and so on. It turns out the algorithm, to do what you just heard, that can be done with one line of code 

It take researchers a long time to come up with this line of code. I'm not saying this is an easy problem, But it turns out that when you use the right programming environment, many learning algorithms can be really short programs. So this is also why in this class we're going to use the Octave programming environment. Octave, is free open source software, and using a tool like Octave or Matlab, many learning algorithms become just a few lines of code to implement. Later in this class, I'll just teach you a little bit about how to use Octave and you'll be implementing some of these algorithms in Octave. Or if you have Matlab you can use that too. It turns out the Silicon Valley, for a lot of machine learning algorithms, what we do is first prototype our software in Octave because software in Octave makes it incredibly fast to implement these learning algorithms. Here each of these functions like for example the SVD function that stands for singular value decomposition; but that turns out to be a linear algebra routine, that is just built into Octave. If you were trying to do this in C++ or Java, this would be many many lines of code

linking complex C++ or Java libraries. So, you can implement this stuff as C++ or Java or Python, it's just much more complicated to do so in those languages. What I've seen after having taught machine learning for almost a decade now, is that, you learn much faster if you use Octave as your programming environment, and if you use Octave as your learning tool and as your prototyping tool, it'll let you learn and prototype learning algorithms much more quickly. And in fact what many people will do to in the large Silicon Valley companies is in fact, use an algorithm like Octave to first prototype the learning algorithm, and only after you've gotten it to work, then you migrate it to C++ or Java or whatever. It turns out that by doing things this way, you can often get your algorithm to work much faster than if you were starting out in C++. So, I know that as an instructor, I get to say "trust me on this one" only a finite number of times, but for those of you who've never used these Octave type programming environments before, I am going to ask you to trust me on this one, and say that you, you will, I think your time, your development time is one of the most valuable resources. And having seen lots of people do this, I think you as a machine learning researcher, or machine learning developer will be much more productive if you learn to start in prototype, to start in Octave, in some other language.



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
Some commonly used methods include clustering and association. Unsupervised methods are particularly useful when the researcher does not know much about the data and wants to find patterns or associations among observations.[5]
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/18.jpg"></a>
</p>

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/18.jpg"></a>
</p>

Reinforcement learning algorithms can be broadly categorized into two types [6]:

**Model-based algorithms:** These algorithms explicitly learn a model of the environment, including transition probabilities and rewards. They use this learned model to plan and make decisions. Examples include Monte Carlo methods, Temporal Difference (TD) learning, and Q-learning.

**Model-free algorithms:** These algorithms directly learn the optimal policy or value function without building an explicit model of the environment. They rely on trial-and-error learning through repeated interactions. Examples include Q-learning, SARSA, Deep Q-Networks (DQN), and Proximal Policy Optimization (PPO).


<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/19.jpg"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/20.png"></a>
</p>

**Commonly used Methods**

Reinforcement learning algorithms can be broadly categorized into two types [6]:

**Model-based algorithms**: These algorithms explicitly learn a model of the environment, including transition probabilities and rewards. They use this learned model to plan and make decisions. Examples include Monte Carlo methods, Temporal Difference (TD) learning, and Q-learning.

**Model-free algorithms**: These algorithms directly learn the optimal policy or value function without building an explicit model of the environment. They rely on trial-and-error learning through repeated interactions. Examples include Q-learning, SARSA, Deep Q-Networks (DQN), and Proximal Policy Optimization (PPO).

<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/21.jpg"></a>
</p>
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/22.jpg"></a>
</p>

---

## References

1. [Main Concepts Behind Machine Learning](https://medium.com/neuronio/main-concepts-behind-machine-learning-22cd81d68a11)
2. [7 Important Concepts in Artificial Intelligence and Machine Learning](https://pub.towardsai.net/clarification-of-concepts-and-definitions-in-artificial-intelligence-and-machine-learning-b1a547e8cafe?gi=f4d15e605843)
3. [Elucidating the Power of Inferential Statistics To Make Smarter Decisions! (Unread)](https://pub.towardsai.net/elucidating-the-power-of-inferential-statistics-to-make-smarter-decisions-6e8d4b0643ef)
4. [Main Concepts Behind Machine Learning (Duplicate)](https://medium.com/neuronio/main-concepts-behind-machine-learning-22cd81d68a11)
5. [Important Concepts in Artificial Intelligence and Machine Learning (Duplicate)](https://pub.towardsai.net/clarification-of-concepts-and-definitions-in-artificial-intelligence-and-machine-learning-b1a547e8cafe)
6. [Machine Learning: An Introductory Tutorial for Beginners](https://arunp77.medium.com/machine-learning-an-introductory-tutorial-for-beginners-1957475e6c0)
7. [Which Machine Learning Algorithm Should I Use?](https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/)


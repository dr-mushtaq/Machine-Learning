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


 ## **Natural Intelligence (NI)**
 
NI is the intelligence naturally evolved in living beings in response to the level of complexity in problems, challenges  etc  Natural intelligence (NI) is the inherent ability of living organisms to adapt to and interact with their environment. It encompasses a wide range of cognitive capabilities, including learning, problem-solving, reasoning, and communication.

NI refers to the intelligence naturally evolved in living beings. It is the inherent ability of organisms—such as humans, animals, and birds—to adapt to their environment, solve problems, and make decisions in uncertain situations.

In simple terms, NI helps us figure out “what to do when we don’t know what to do.”

Helps in “What to do, when we do not know” 
Humans, animals, birds etc 
Thought, Vision, Speech, Hear, Feel
Key features of Natural Intelligence:
Cognition: Thinking, reasoning, and learning from experiences.

Perception: Vision, hearing, and touch.

Communication: Speech and interaction.

Adaptation: Adjusting behavior to survive and thrive.

Examples: Human problem-solving, birds migrating across continents, or animals using tools in the wild.

The human brain perceives things from the real world, processes the perceived information, makes rational decisions, and performs certain actions based on circumstances. This is what we called behaving intelligently. When we program a facsimile of the intelligent behavioral process to a machine, it is called artificial intelligence (AI).

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

## History of Artificial intelligence

Even though artificial intelligence, or AI, has only been around for less than a hundred years, the idea of machines that can think goes way back. Even in ancient Greece, people were talking about intelligent robots and artificial beings. The whole idea of AI really starts with asking if machines can think like humans.

**1955** 

In 1955, Allen Newell and Herbert A. Simon made the first computer program meant to act like a smart thinker. They called it the "Logic Theorist." This program tried to prove math ideas using logic symbols. It used a special way of searching for answers that imitated how humans solve problems. The Logic Theorist was like the first computer tool that could solve lots of different problems, not just one. It was a big deal in the world of smart computer programs.

 Alan Turing, was an eminent mathematician, who is famous for breaking the Nazi Enigma code during World War II. This code gave the Allied Powers the edge they needed to win the war — it also laid the foundation for the creation of the computer.

**1956**

It was the year when the term “Artificial Intelligence” was first coined as an academic field by American computer scientist John McCarthy at the Dartmouth Conference. This conference was attended by some of the leading researchers in the field of AI, including Marvin Minsky, Claude Shannon, and Nathaniel Rochester.At the conference, McCarthy gave a talk titled “The Limitations of and Prospects for Information Processing in Problem-Solving Machines.” In this talk, he proposed the creation of a new field of study called “Artificial Intelligence” that would focus on developing intelligent machines [7].

Dartmouth scientist, John McCarthy, expanded on Turing’s ideas and coined the term “artificial intelligence” in 1956. McCarthy assembled a team of computer scientists and mathematicians to investigate if robots could learn the same way that children do, through trial and error, to build formal reasoning. The team hoped to ascertain how they could make machines “use language, form abstractions and concepts, solve [the] kinds of problems now reserved for humans, and improve themselves.”

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/file-6016287eb5c25%20(1).jpg"></a>
</p>

 First, Alan Turing, a founding father of AI, came up with the question that “Can machines think like humans?”. Later, John McCarthy created the term “Artificial Intelligence” and invented the programming language LISP, played computer chess via the telegraph with opponents in Russia, and invented computer time-sharing. At that time, the computer was big enough to fill a room. But the concept of AI has created great hope and enthusiasm for the world of science and technology.

In recent years, AI has focused on tasks that only humans can do, such as image and voice recognition. Thus, the following problems that were previously unsolvable were now overcome with AI [3].
- Image recognition
- Object recognition
- Language-to-language translations
- Natural language comprehension
- Image and speech recognition
- Assistant assistants
- Driverless cars
While AI was seen in the 60s and 70s as a computer skill that could play chess and checkers, perform simple calculations, and solve mathematical problems, in the 80s and 90s it was seen as a risk assessment and decision-making ability, and in the 2000s, with the development of the computational potential of computers, it was understood that learning systems could be possible [3]

- Thought: Facebook uses an artificial neural network for facial recognition
- Speech and Hearing: Google Assistant, Siri
- Vision: Capturing Traffic Violations                              

**The child's brain**

A child's brain and senses perceive the facts of their surroundings and gradually learn the hidden patterns of life which help the child to craft logical rules to identify learned patterns. The learning process of the human brain makes humans the most sophisticated living creature of this world. Learning continuously by discovering hidden patterns and then innovating on those patterns enables us to make ourselves better and better throughout our lifetime. This learning capacity and evolving capability is related to a concept called brain plasticity. Superficially, we can draw some motivational similarities between the learning process of the human brain and the concepts of machine learning.

 ## **What is Artificial intelligence (AI)**

 There are many different definitions and versions of Artificial Intelligence, WHICH IS GENUINE, TRUE AND REAL.

**Def:** Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

Unlike NI, which naturally evolved, Artificial Intelligence (AI) is created by humans. AI refers to the simulation of human intelligence in machines that are designed to think, reason, and act in ways similar to people.

The concept of AI was formally introduced in 1956 at the Dartmouth Conference by John McCarthy, who defined it as “the science and engineering of making intelligent machines.”

**Multiple Definitions of AI**

Because AI is such a broad and evolving field, different organizations and researchers define it in slightly different ways:

**Def:** What is AI? AI is a technology that allows machines to simulate human behavior. It is a field of computer science that allows machines to execute complex tasks such as image recognition, decision-making, and conversing [4]. 

**Def:** The term Artificial intelligence (AI) was first coined a decade ago in the year 1956 by John McCarthy at the Dartmouth conference. He defined “Artificial intelligence as the science and engineering of making intelligent machines”. In a sense, artificial intelligence is a technique of getting a machine to work and behave like humans. AI works best by combining a large number of data sets with fast, repetitive, and intelligent algorithms.

**IBM**

Def : Artificial intelligence leverages computers and machines to mimic the problem-solving and decision-making capabilities of the human mind.

**Oracle**

Def: Artificial Intelligence refers to systems or machines that mimic human intelligence to perform tasks and can iteratively improve themselves based on the information they collect.

**Accenture**

Def :Artificial intelligence is a constellation of many different technologies working together to enable machines to sense, comprehend, act, and learn with human-like levels of intelligence

**SAS**

Artificial intelligence (AI) makes it possible for machines to learn from experience, adjust to new inputs and perform human-like tasks.

**Encyclopedia Britannica**

Artificial Intelligence is the ability of a digital computer or computer-controlled robot to perform tasks commonly associated with intelligent beings.
Stanford University
Artificial Intelligence is the science and engineering of making intelligent machines, especially intelligent computer programs.
**Amazon AWS**

Artificial Intelligence is the field of computer science dedicated to solving cognitive problems commonly associated with human intelligence, such as learning, problem solving, and pattern recognition.
**European Parliament**

AI is the ability of a machine to display human-like capabilities such as reasoning, learning, planning and creativity.
**Qualcomm**

AI is an umbrella term representing a range of techniques that allow machines to mimic or exceed human intelligence.

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/what%20is%20ai.png"></a>

**Background:**

This allows AI software to automatically learn from patterns or symbols in those big data sets. AI is the simulation of human intelligence by machines. It enables the machine to think like a human. It is a program that can learn, reach and sense the same as humans do or  Similar to the intelligence humans possess, artificial intelligence is the one donned by machines. The only difference remains the absence of emotionality and consciousness. The Capacity is given by humans to machines to memorize and learn from experience and to think and create, to speak, to judge, and make decisions. The brain is the most wonderful organ of the human body. The brain controls thought, memory, emotion, motor skills, vision, breathing, and touch. This complex structure of the brain became a source of inspiration for scientists and the concept of AI emerged. AI is the ability of a computer or robot to perform humanoid tasks [3].


 ## **Weak AI vs. Strong AI**

| **Weak AI** | **Strong AI** |
|--------------|---------------|
| Weak AI refers to AI systems that are designed and trained for a specific task or a narrow set of tasks. | Strong AI, or Artificial General Intelligence (AGI), refers to AI systems with human-level intelligence and understanding. |
| These AI systems are not generally intelligent; they excel in performing a predefined task but lack true understanding or consciousness. | These AI systems have the ability to perform any intellectual task that a human being can do, adapt to different domains, and possess a form of consciousness or self-awareness. |
| Examples of weak AI include virtual assistants like Siri or Alexa, recommendation algorithms used by streaming services, and chatbots that are designed for specific customer service tasks. | Achieving Strong AI is a long-term goal of AI research and would require the development of AI systems that can reason, learn, understand, and adapt across a wide range of tasks and contexts. |
| Weak AI is highly specialized and does not possess human-like cognitive abilities or general problem-solving capabilities beyond its narrow domain. | Strong AI is currently a theoretical concept, and no AI system has reached this level of general intelligence. |

## **How can an algorithm be defined?** 

An algorithm can be defined as a set of step-by-step instructions for solving a problem or accomplishing a task. It's essentially a roadmap for a computer or other system to follow to achieve a desired outcome.   
Here are some key characteristics of an algorithm:

- **Unambiguous:** The instructions must be clear and concise, leaving no room for interpretation.
- **Finite:** The algorithm must have a definite starting and ending point.
- **Effective:** The algorithm must be efficient and solve the problem within a reasonable amount of time and resources.
- **Generalizable:** The algorithm should be able to work for a variety of inputs within its intended scope.

**Example:**
A simple example of an algorithm is the recipe for baking a cake. It provides a clear sequence of steps, starting with preheating the oven and ending with checking if the cake is done. Each step is specific and unambiguous, and the recipe can be easily followed by anyone to achieve the desired outcome of a delicious cake.

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/408209845_7282561661763388_2901782783385024646_n.jpg"></a>
</p>

# 🤖 Why We Use Machine Learning: Moving Beyond Rule-Based Systems

## 🌍 From Hand-Written Rules to Intelligent Systems

Before machine learning became mainstream, programmers relied on manually written rules derived from experience and domain knowledge.

For example, if you were building a fraud detection system, you might write rules like:

- “If the transaction amount exceeds $10,000 and the location is X, flag it as suspicious.”

This traditional approach worked — but only for a while. As data became more complex and dynamic, these hard-coded rules started to break down.

Before machine learning became mainstream, programmers wrote rules derived from a function of their domain knowledge, observation of some hand-picked instances, and the business requirement to perform a particular task. But this legacy way of delivering business results suffered some evident constraints.

## ⚙️ The Limitations of Rule-Based Systems

Hand-written rules face several key challenges:

**Limited human capacity:**

As psychologist George A. Miller’s classic paper “The Magical Number Seven, Plus or Minus Two” suggests, humans can only process a limited number of variables at once. Beyond that, managing all possible edge cases becomes impossible.

**Data keeps changing:**

In our digital world, data is dynamic. A rule that worked yesterday may fail today as user behavior, trends, and threats evolve.

**Poor scalability:**

Updating a large set of rules is time-consuming and error-prone, especially if the original developer leaves and a new one inherits the system.

**Static intelligence:**

Once written, rules don’t learn from new data. They stay frozen in time — which is fatal in fast-changing domains like cybersecurity, healthcare, and finance.

## 💡 Why Machine Learning Wins

Machine learning (ML) flips the script.
Instead of telling the system what to do, we teach it how to learn.

ML algorithms automatically extract patterns from historical data and improve over time as new data arrives.

Returning to our fraud detection example:

- A traditional rule might check for a single suspicious pattern.

- A machine learning model can learn hundreds of subtle relationships — transaction timing, device type, spending frequency, or geographic anomalies — and detect previously unseen fraud patterns.

The key difference?

- Rule-based systems are explicitly programmed, while machine learning systems are trained.

## 🧠 Machine Learning: A System That Learns Like Humans

From a systems perspective, machine learning creates automated systems capable of discovering hidden patterns in data to make intelligent decisions.

This concept draws inspiration from how the human brain learns.
Just as humans adjust behavior based on experience, ML models improve their predictions by continuously learning from new examples.

# What is machine learning?

There are many different definitions and versions of Artificial Intelligence, Machine Learning , WHICH IS GENUINE, TRUE AND REAL

**Def**: Machine learning is a branch of artificial intelligence (AI) that focuses on developing algorithms and statistical models that enable computer systems to learn from data, identify patterns, and make predictions or decisions without being explicitly programmed. 

**Def**: Artificial intelligence is an umbrella term that contains many realms like machine learning, image processing, neural networks, cognitive science, and many more. AI is an umbrella discipline that covers everything related to making machines smarter. 

**Def:** The capability of Artificial Intelligence systems to learn by extracting patterns from data is known as Machine Learning is the idea to learn from examples and experiences without being explicitly programmed. Instead of writing code, you feed data to the generic algorithm and it builds logic based on the data given.

Although the terms can be confused, machine learning (ML) is an important subset of artificial intelligence. ML is concerned with using specialized algorithms to uncover meaningful information and find hidden patterns from perceived data to corroborate the rational decision-making process.

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/394577986_345138894702935_4471456550713557158_n.jpg"></a>
</p>


<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/ww.jpg "></a>
</p>

**Definition:** 1 Machine Learning (ML) is commonly used along with AI but it is a subset of AI. ML refers to an AI system that can self-learn based on the algorithm. Systems that get smarter and smarter over time without human intervention are ML Machine learning is the study of Computer Algorithms that allow computer programs to automatically improve through experience Or In ML machines can learn by themselves without being explicitly programmed.

**Definition  2**: Machine Learning (ML) is based on algorithms that can learn from data without relying on rule-based programming

**Definition:** 2  Machine learning is a branch of study that allows machines to learn patterns from data without the involvement of explicit programming. The learnings are based on their experiences without human interference. 

**Definition:** 3:  Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem. Instead of writing code, you feed data to the generic algorithm and it builds its own logic based on the data.

**MathWorks**

**Def:** "Machine Learning is an AI technique that teaches computers to learn from experience. Machine learning algorithms use computational methods to “learn” information directly from data without relying on a predetermined equation as a model. The algorithms adaptively improve their performance as the number of samples available for learning increases. Deep learning is a specialized form of machine learning".

Unlike conventional programming, in machine learning language, the computer uses a pre-written algorithm and learns how to solve the problem itself. It is a more sophisticated way of solving a problem. Machine learning language is beyond algorithmic solutions; instead, it trains a machine to solve different complex tasks by itself.

**Definition:** 4 Arthur Samuel Definition (1954): He defined machine learning as the field of study that gives computers the ability to learn without being explicitly programmed.



**Definition:** 5 Here's a slightly more recent definition by Tom Mitchell,   So Tom defines machine learning by saying that, a well-posed learning problem is defined as follows. He says, a computer program is said to learn from experience E, with respect to some task T, and some performance measure P, if its performance on T as measured by P improves with experience E.

For the checkers playing example the experience E, will be the experience of having the program play 10's of 1000's of games against itself. Task T will be the task of playing checkers. And the performance measure P will be the probability that it wins the next game of checkers against some new opponent.

Machine Learning is the study of computer algorithms that allow computer programs to automatically improve through experience

**Definition  6:** Machine learning (ML) in the field of the scientific study of algorithms and uses various statistical models. The computer systems use these statistical models to perform a specific task effectively. Here you don’t need to provide explicit instructions; instead, it relies on patterns and inference. As shown in the above image, machine learning is a subset of artificial intelligence. Machine learning language algorithms build a mathematical model depending on sample data. This data is known as “training data.” Using these data and algorithms prepares predictions or decisions. Here you don’t need to program to perform the task explicitly.

                                       

Def: Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem. Instead of writing code, you feed data to the generic algorithm and it builds its own logic based on the data [8].



For example, one kind of algorithm is a classification algorithm. It can put data into different groups. The same classification algorithm used to recognize handwritten numbers could also be used to classify emails into spam and not-spam without changing a line of code. It’s the same algorithm but it’s fed different training data so it comes up with different classification logic.

Example: Arthur Samuel wrote a checkers playing program. And the amazing thing about this checkers playing program was that Arthur Samuel himself wasn't a very good checkers player. But what he did was, he had to program for it to play 10's of 1000's of games against itself. And by watching what sorts of board positions tended to lead to wins, and what sort of board positions tended to lead to losses. The checkers playing program learns over time what are good board positions and what are bad board positions. And eventually, learn to play checkers better than Arthur Samuel himself was able to. This was a remarkable result.

Although Samuel himself turned out not to be a very good checkers player. But because the computer has the patience to play tens of thousands of games itself. No human has the patience to play that many games. By doing this the computer was able to get so much checkers-playing experience that it eventually became a better checkers player than Arthur Samuel himself. This is the somewhat informal definition and an older one.

                                        

  

Example

let us begin by considering a couple of examples from naturally occurring animal learning.

Bait Shyness — Rats Learning to Avoid Poisonous Baits: When rats encounter food items with novel look or smell, they will first eat very small amounts, and subsequent feeding will depend on the flavor of the food and its physiological effect. If the food produces an ill effect, the novel food will often be associated with the illness, and subsequently, the rats will not eat it. Clearly, there is a learning mechanism in play here.– the animal used past experience with some food to acquire expertise in detecting the safety of this food. If past experience with the food was negatively labeled, the animal predicts that it will also have a negative effect when encountered in the future.

Inspired by the preceding example of successful learning, let us demonstrate a typical machine learning task. Suppose we would like to program a machine that learns how to filter spam e-mails. A naive solution would be seemingly similar to the way rats learn how to avoid poisonous baits. The machine will simply memorize all previous e-mails that had been labeled as spam e-mails by the human user. When a new e-mail arrives, the machine will search for it in the set of previous spam e-mails. If it matches one of them, it will be trashed. Otherwise, it will be moved to the user’s inbox folder.

While the preceding “learning by memorization” approach is sometimes useful, it lacks an important aspect of learning systems — the ability to label unseen e-mail messages. A successful learner should be able to progress from individual examples to broader generalization. This is also referred to as inductive reasoning or inductive inference. In the bait shyness example presented previously, after the rats encounter an example of a certain type of food, they apply their attitude toward it on new, unseen examples of food of similar smell and taste. To achieve generalization in the spam filtering task, the learner can scan the previously seen e-mails, and extract a set of words whose appearance in an e-mail message is indicative of spam. Then, when a new e-mail arrives, the machine can check whether one of the suspicious words appears in it, and predict its label accordingly. Such a system would potentially be able correctly to predict the label of unseen e-mails.

2.3 - Benefits of Machine Learning

Powerful processing
Better Decision making and prediction
Quicker processing
accurate
affordable data management
Inexpensive
analyzing complex big data     
2.4- Application of Machine Learning Algorithms
The developed machine learning algorithms are used in various applications such as

 1- AlphaGo':- Apparently, the landmarks of publicity and publicity were AlphaGo's ingenious completion program that ended 2 500 years of humanity in May 2017 in the ancient board game GO using a machine learning algorithm called "reinforcing learning". 

 Web Search
Computational Biology 
 Finance
 E-Commerce
Space exploration
Robotics
Information extraction
Social Networks
Debugging
Data mining
Vision processing
Forecasting things like stock market trends, weather
Pattern recognition
Game
2.5- Difference between ML and Programming.

In programming, we can segregate the world into two broad categories – Conventional programming and Machine learning. Conventional or conventional programming has been around here for more than a century. The first computer programming was introduced in the mid-1800s. On the other side, machine learning programming is a new addition to this family, which has revolutionized the business for the last few decades. 

In conventional programming, programs are created manually by providing input data based on the programming logic, and the computer generates the output.

On the contrary, On the contrary, in machine learning programming, the input and output data are fed to the algorithm, creating the program.

Conventional Programming uses conventional procedural language. It could be an assembly language or a high-level language such as C, C++, Java, JavaScript, Python, etc.

Conventional programming is a manual process, which means the programmer creates the logic of the program. They need to code the rules and write lines of code manually. 

They provide the input data, and based on the program’s programming logic; it produces the desired output. 

The conventional programming approach is algorithm-dependent, and for a program, multiple algorithms can work. It is up to the programmer how he will design and develop the logic of the program.

Machine Learning is based on the idea that analytic systems can learn to identify patterns and make decisions with minimal human involvement using statistics, linear algebra, and numerical optimization.

Machine Learning as a way of writing programs whose business logic is generated from input data. We feed data to the algorithm and the result of the program execution will be the logic for processing new data. It is a new way of writing software, a step away from the traditional development process.

In conventional programming, a programmer needs to hard code the logic of the program.

 In machine learning, it depends a lot on the machine which learns from input data.

Conventional Programming is not a very advanced level, where decision-making is based on IF-ELSE conditions. Therefore, many solutions cannot be modeled with it.

On the contrary, machine learning programming solves the problem by modeling the data with train data and test data. Based on these data and statistical models, machine learning predicts the result.

Also, there is a significant difference between machine learning and conventional programming based on the number of input parameters that the model can process. In machine learning, to get an accurate prediction, it is required to feed thousands of parameters. Besides, it must be done with high accuracy, because every bit can affect the final result. However, in conventional programming, a programmer cannot build an algorithm following the same patterns.







2.6-- What is an ML model

The output of Machine learning algorithms is called the ML model or what was learned from machine learning algorithms save after running ML algorithms on training data  and solutions that rules, numbers, and any algorithms-specific data structure 

                                                    

2.7- What is an ML Algorithm?
An algorithm is a procedure that is run on data to create a machine model. These algorithms perform pattern recognition and learn from data. Or are fit on a dataset

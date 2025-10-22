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

- **Hand-written rules** are limited by the knowledge of what edge cases a programmer can cover. This concept is very well explained by one of the most highly cited papers in the world of psychology titled “The Magical Number Seven, Plus or Minus Two: Some Limits on Our Capacity for Processing Information.” Commonly cited as Miller’s law, the paper describes the limited amount of information an average brain can hold and how it becomes unmanageable with the increasing number of variables and dimensions.
- **Data is dynamic** by nature and has become more so over the last decade with the proliferation of technology in our day-to-day lives. The varying data patterns fed to static pre-written rules are of little help to the business in taking meaningful actions. That is where the pattern mining ability of machine learning algorithms is put to the best use
  
- Let’s take an example of a fraud detection case. The programmer would write rules that if the transaction amount is above $10K, transaction location is X, and it is made from a particular type, i.e., wire transfer, then it is flagged as a potentially fraudulent transaction.This could work like a charm for some time until the bad actors find an intelligent way to commit fraud. The conventional hard-coded rules are no more effective in detecting fraud. As their way of operation evolves, our fraud detection system needs to become too.
- Further, all the software development processes are highly collaborative?—?what if the developer who wrote the initial rules is no longer associated with the fraud detection project? And the new developer tasked with upgrading the logic has no understanding of the previous system and is skeptical about whether the recent changes will have backward compatibility. To summarize, updating a rule-based system is not only a cumbersome process but unscalable too.
That’s where machine learning algorithms come to our rescue. If the metrics are well-defined and well-aligned with the business objective, it continues to learn from the new training data and evolves into a sophisticated machine learning system.

Machine learning, from a systems perspective, is defined as the creation of automated systems that can learn hidden patterns from data to aid in making intelligent decisions.

This motivation is loosely inspired by how the human brain learns certain things based on the data it perceives from the outside world.


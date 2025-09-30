Supervised learning, a key concept in machine learning, enables computers to make decisions based on labeled data. Whether you’re classifying emails as spam or predicting customer behavior, scikit-learn offers accessible tools to build effective models.

In this blog, we dive into classification techniques, a major type of supervised learning, using Python’s powerful scikit-learn library. This guide provides a step-by-step approach, from understanding core concepts to implementing your first classification model, perfect for beginners and those looking to refine their data science skills.


## 📑 Table of Contents  

- [What is Machine Learning](#What-is-Machine-Learning)  
- [Unsupervised learning](#Unsupervised-learning)  
- [Reinforcement learning](#Reinforcement-learning)  
- [Supervised learning explanation](#Supervised-learning-explanation)
- [Real life Example?](#Real-life-Example)   
- [History of Computer Vision](#history-of-computer-vision)  

cvcv
### **What is Machine Learning** 

Machine learning is the science and art of giving computers the ability to learn to make decisions from data without being explicitly programmed.
For example, your computer can learn to predict whether an email is spam or not spam given its content and sender. Another example: your computer can learn to cluster, say, Wikipedia entries, into different categories based on the words they contain. It could then assign any new Wikipedia article to one of the existing clusters. Notice that, in the first example, we are trying to predict a particular class label, that is, spam or not spam. In the second example, there is no such label. When there are labels present, we call it **supervised learning**. When there are no labels present, we call it **unsupervised learning**.

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised%20Learning%20with%20scikit_learn/Chapter1-Classification/1.png"></a>
</p>

 ## **Unsupervised learning**
 
Unsupervised learning, in essence, is the machine learning task of uncovering hidden patterns and structures from unlabeled data. For example, a business may wish to group its customers into distinct categories based on their purchasing behavior without knowing in advance what these categories may be. This is known as clustering, one branch of unsupervised learning.

### **Reinforcement learning** 

There is also reinforcement learning, in which machines or software agents interact with an environment. Reinforcement agents are able to automatically figure out how to optimize their behavior given a system of rewards and punishments. Reinforcement learning draws inspiration from behavioral psychology and has applications in many fields, such as, economics, genetics, as well as game playing. In 2016, reinforcement learning was used to train Google DeepMind’s AlphaGo, which was the first computer program to beat the world champion in Go.

### **Supervised learning explanation** 

In supervised learning, we have several data points or samples, described using predictor variables or features and a target variable. Our data is commonly represented in a table structure such as the one you see here, in which there is a row for each data point and a column for each feature. Here, we see the iris dataset: each row represents measurements of a different flower and each column is a particular kind of measurement, like the width and length of a certain part of the flower. The aim of supervised learning is to build a model that is able to predict the target variable, here the particular species of a flower, given the predictor variables, here the physical measurements. If the target variable consists of categories, like ‘click’ or ‘no click’, ‘spam’ or ‘not spam’, or different species of flowers, we call the learning task classification. Alternatively, if the target is a continuously varying variable, for example, the price of a house, it is a regression task. In this chapter, we will focus on classification. In the following, on regression.

<p align="center">
<img src="https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised%20Learning%20with%20scikit_learn/Chapter1-Classification/2.png"></a>
</p>
 
Computer Vision is really about analyzing images and videos to extract knowledge from them. Mostly these images are of real scenes like that of a street image with cars and such where autonomous vehicle they'll have to navigate through or it could be other types of images like that of an X-ray inside a human head and we need to do image analysis to be able to extract things about of interest in medical applications.

Computer vision is the field of computer science that focuses on creating digital systems that can process, analyze, and make sense of visual data (images, videos) in the same way that humans do. Computer Vision uses convolutional neural networks to process visual data at the pixel level and deep learning recurrent neural networks to understand how one pixel relates to another [2] So essentially the goal is image and video understanding which means labeling interesting things in an image and also tracking them as they move.

Well, there's a couple of ways of thinking about it. I like this slide that I borrowed from Steve Seitz, where he talks about every picture tells a story. And one way of thinking about computer vision is the goal  is to interpret images. That is, say something about what's present in the scene or what's actually going on. So, what we're doing is, we're going to take images in, and what's going to come out is something that has some meaning to it. That is, we're going to extract, we're going to create some sort of interpretations, some sort of an understanding of what that image is representative of. This is different, many of you may have some exposure to image processing, which is the manipulation of images. That's images in and images out. And we'll talk a little bit about that because you use image processing for per, for computer vision. But fundamentally computer vision is about understanding something that's in the image.
9
###  **3-How does computer vision work**?
Computer vision technology tends to mimic the way the human brain works. But how does our brain solve visual object recognition? One of the popular hypothesis states that our brains rely on patterns to decode individual objects. This concept is used to create computer vision systems [5].Computer vision algorithms that we use today are based on pattern recognition. We train computers on a massive amount of visual data — computers process images, label objects on them, and find patterns in those objects. For example, if we send a million images of flowers, the computer will analyze them, identify patterns that are similar to all flowers and, at the end of this process, will create a model “flower.” As a result, the computer will be able to accurately detect whether a particular image is a flower every time we send them pictures.
Computer vision works in three basic steps:

1- **Acquiring an image**

Images, even large sets, can be acquired in real-time through video, photos or 3D technology for analysis.

2- **Processing the image**

Deep learning models automate much of this process, but the models are often trained by first being fed thousands of labeled or pre-identified images. Computer vision algorithms are based on pattern recognition. We train our model on a massive amount of visual(images) data. Our model processes the images with label and find patterns in those objects(images).

3- **Understanding the image**

The final step is the interpretative step, where an object is identified or classified.

###  Real life Example

For example, If we send a million pictures of vegetable images to a model to train, it will analyze them and create an Engine (Computer Vision Model) based on patterns that are similar to all vegetables. As a result, Our Model will be able to accurately detect whether a particular image is a Vegetables every time we send it .

<p align="center">
<img src="https://github.com/dr-mushtaq/Computer-Vision/blob/main/%F0%9F%93%9AChapter%201-Introduction/1_uhwJAFDBNBjTVmJ_6P5Zyg.png"></a>
</p>

### References

1-[What is Computer Vision? & Its Applications](https://medium.com/@draj0718/what-is-computer-vision-its-applications-826c0bbd772b)

2-[-Introduction of Computer Vision](https://auth.udacity.com/sign-in)

4-[How computer vision works](https://www.sas.com/en_us/insights/analytics/computer-vision.html#technical)

5-[Computer Vision 🤖 Fundamentals with OpenCV](https://medium.com/codex/computer-vision-fundamentals-with-opencv-9fc93b61e3e8)


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



























# Machine Learning

The goal of machine learning is to discover patterns and relationships in data and put those findings to use. This process of discovery is achieved through the use of modeling techniques that have been developed over the past 30 years in statistics, computer science and applied mathematics [1]. These different approaches can range from simple to extremely complex, but they all share a common goal: to estimate the functional relationship between the input characteristics and the target variable. 

Building a successful machine-learning model involves the following steps:

- Data Preparation
- Model Selection
- Model Training
- Validation
- Hyperparameter Tuning

---

## 1- Data Preparation or Preprocessing

A crucial component of machine learning is the availability of large and labeled datasets. These datasets serve as the training data for ML algorithms, allowing them to learn patterns and make accurate predictions. Datasets can vary based on the problem domain, such as images, text, audio, or a combination of different data types.

Data is a crucial component of deep learning. A dataset refers to a collection of examples or instances used to train, validate, and test deep learning models. Datasets can include images, text, audio, video, or any other type of information relevant to the task at hand.

The first step in building a machine learning model is to gather and prepare the data. This includes cleaning, transforming, and scaling the data. In data science, data cleaning is the process of identifying incorrect data and fixing the errors so the final dataset is ready to be used. Errors could include duplicate fields, incorrect formatting, incomplete fields, irrelevant or inaccurate data, and corrupted data.

### 1.1 Historical Data or Raw Data

As shown in the image above, the machine learning process usually begins with collecting historical data. Then, this data is prepared to fit into a machine-learning model[1].

### 1.2 Creating New Features

Creating new features is an important step in feature engineering that involves creating new variables or columns from existing data. This can help to capture complex relationships between the features and improve the accuracy of the models. Here are some techniques for creating new features[2]:

1. **Interaction features:** Interaction features are created by multiplying two or more existing features together.  
2. **Polynomial features:** Polynomial features are created by raising existing features to a higher power.  
3. **Binning:** Binning involves grouping continuous values into discrete categories.  

### 1.3 Encoding Categorical Variables

Encoding categorical variables is a crucial step in feature engineering that involves converting categorical variables into a numerical form that machine learning algorithms can understand. Here are some common techniques used for encoding categorical variables[2]:

- **One-Hot Encoding**
- **Label Encoding**
- **Ordinal Encoding**

### 1.4 Handling Imbalanced Data

Imbalanced data is a situation where the distribution of the target variable is not uniform, and one class is underrepresented compared to the other. Some techniques to handle imbalanced data are[2]:

1. Upsampling  
2. Downsampling  
3. SMOTE  
4. Class Weighting  
5. Anomaly Detection (Isolation Forest)  
6. Cost-Sensitive Learning  

### 1.5 Skewness and Kurtosis Handling

Skewness = asymmetry of distribution.  
Kurtosis = peakedness or flatness.

Techniques:

- Log Transformation  
- Square Root Transformation  
- Box-Cox Transformation  

### 1.6 Handling Rare Categories

Rare categories can cause problems in machine learning models, as they may not have enough representation. Techniques [2]:

- Grouping rare categories
- Replace rare categories with common category
- One-hot encoding with rare flag

### 1.7 Scaling and Normalization

| Technique | Description |
|----------|-------------|
| Standardization | mean=0, std=1 |
| Min-Max Scaling | values between 0 and 1 |
| Robust Scaling | uses median & IQR |
| Normalization | unit norm |

### 1.8 Data Splitting

Generally, a dataset should be split:

- **80% Training**
- **20% Testing**

#### Training Data
ML algorithms learn relationships from the training data.

#### Test Data
Used only after training to evaluate performance.

#### Validation Set
Used to evaluate hyperparameters.

---

## 2- Model Selection

After data preparation, the next step is to choose an appropriate ML algorithm.

Depends on:

- Size / quality / nature of data
- Available computation
- Urgency of task
- What you want to do with the data

---

## 3- Model Training
Once an algorithm has been selected, it needs to be trained using a subset of the data.

---

## 4- Validation
Validation involves testing the trained model on a separate dataset to evaluate its performance.

---

## 5- Hyperparameter Tuning
Hyperparameters are settings that govern how the ML algorithm works. Tuning them can improve performance.

---

## References

1. Machine Learning Process  
2. The Ultimate Guide to Machine Learning: Feature Engineering — Part -2  
3. How to Split the Dataset into Training and Test sets  
4. Which machine learning algorithm should I use?

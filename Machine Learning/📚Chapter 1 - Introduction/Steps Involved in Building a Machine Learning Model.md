# Steps involved in building Machine Learning Models

The goal of machine learning is to discover patterns and relationships in data and put those findings to use. This process of discovery is achieved through the use of modeling techniques that have been developed over the past 30 years in statistics, computer science and applied mathematics [1]. These different approaches can range from simple to extremely complex, but they all share a common goal: to estimate the functional relationship between the input characteristics and the target variable. Building a successful machine-learning model involves the following steps:

- Data Preparation  
- Model Selection  
- Model Training  
- Validation  
- Hyperparameter Tuning  

# 1- Data Preparation or Preprocessing
<p align="center">
<img src="https://github.com/aminasaeed223/Machine-Learning/blob/master/Machine%20Learning/%F0%9F%93%9AChapter%201%20-%20Introduction/1.jpg"></a>
</p>


A crucial component of machine learning is the availability of large and labeled datasets. These datasets serve as the training data for ML algorithms, allowing them to learn patterns and make accurate predictions. Datasets can vary based on the problem domain, such as images, text, audio, or a combination of different data types.

Data is a crucial component of deep learning. A dataset refers to a collection of examples or instances used to train, validate, and test deep learning models. Datasets can include images, text, audio, video, or any other type of information relevant to the task at hand.

The first step in building a machine learning model is to gather and prepare the data. This includes cleaning, transforming, and scaling the data. In data science, data cleaning is the process of identifying incorrect data and fixing the errors so the final dataset is ready to be used. Errors could include duplicate fields, incorrect formatting, incomplete fields, irrelevant or inaccurate data, and corrupted data.

---

## 1.1- Historical Data or Raw Data  
As shown in the image above, the machine learning process usually begins with collecting historical data. Then, this data is prepared to fit into a machine-learning model [1].

---

## 1.2- Creating New Features  
Creating new features is an important step in feature engineering that involves creating new variables or columns from existing data. This can help to capture complex relationships between the features and improve the accuracy of the models. Here are some techniques for creating new features [2]:

1. **Interaction features:**  
   Interaction features are created by multiplying two or more existing features together. This can help to capture the joint effects of the features and uncover new patterns in the data. For example, if we have two features, “age” and “income”, we can create a new interaction feature called “age_income” by multiplying these two features together.

2. **Polynomial features:**  
   Polynomial features are created by raising existing features to a higher power. This can help to capture non-linear relationships between the features and improve the accuracy of the models. For example, if we have a feature “age”, we can create a new polynomial feature called “age_squared” by squaring this feature.

3. **Binning:**  
   Binning involves grouping continuous values into discrete categories. This can help to capture non-linear relationships and reduce the impact of outliers in the data. For example, if we have a feature “age”, we can create a new binned feature called “age_group” by grouping the ages into different categories such as “0–18”, “18–25”, “25–35”, “35–50”, and “50+”.

---

## 1.3- Encoding Categorical Variables  
Encoding categorical variables is a crucial step in feature engineering that involves converting categorical variables into a numerical form that machine learning algorithms can understand. Here are some common techniques used for encoding categorical variables [2]:

### One-Hot Encoding  
One-hot encoding converts categorical variables into a set of binary features, where each feature corresponds to a unique category in the original variable. In this technique, a new binary column is created for each category, with 1 representing presence and 0 otherwise [2].

### Label Encoding  
Label encoding assigns a unique numerical value to each category. Labels are assigned based on the order of categories in the variable [2].

### Ordinal Encoding  
Ordinal encoding assigns a numerical value to each category based on their order or rank. Categories are arranged according to a specific criterion, and values are assigned based on their position [2].

---

## 1.4- Handling Imbalanced Data  
Imbalanced data occurs when one class is underrepresented. This can bias the model toward the majority class. Some techniques to handle imbalanced data include [2]:

1. **Upsampling:** Creating more samples for the minority class by resampling with replacement.  
2. **Downsampling:** Removing some samples from the majority class.  
3. **SMOTE:** Creating synthetic samples for the minority class.  
4. **Class Weighting:** Assigning weights to classes to balance their influence.  
5. **Anomaly Detection:** Identifying and removing outliers or rare events.  
6. **Cost-Sensitive Learning:** Assigning different costs to errors during training.

---

## 1.5- Skewness and Kurtosis Handling  
Skewness measures asymmetry; kurtosis measures the peakedness of data. Techniques include:

### For Skewness  
- Log transformation  
- Square root transformation  
- Box-Cox transformation  

### For Kurtosis  
- Log transformation  
- Square transformation  
- Box-Cox transformation  

---

## 1.6- Handling Rare Categories  
Rare categories may lack representation. Techniques include [2]:

- Grouping rare categories together  
- Replacing rare categories with the most common one  
- One-hot encoding with a rare-category flag  

---

## 1.7- Scaling and Normalization  
Scaling ensures features are on similar ranges. Common methods include [2]:

1. **Standardization:** Zero mean and unit variance.  
2. **Min-Max Scaling:** Scale values to 0–1 range.  
3. **Robust Scaling:** Uses median and IQR; good for outliers.  
4. **Normalization:** Scales each observation to unit norm.

---

## 1.8- Data Splitting  

A dataset should typically be split into **80% training** and **20% testing**.

### Training Data  
Training data is used by algorithms to learn relationships and patterns. More training data usually leads to better model performance.

### Test Data  
Used only after training to evaluate performance.

### Validation Set  
Used to tune hyperparameters.  
- Small models may not need a validation set.  
- Models with many hyperparameters require a larger validation split.

---

# 2- Model Selection  
Choosing the right algorithm depends on:

- Data size, quality, and type  
- Computation time  
- Urgency  
- Purpose of analysis  

---

# 3- Model Training  
After selecting an algorithm, it is trained using part of the dataset.

---

# 4- Validation  
Validation tests the trained model on separate data to measure performance.

---

# 5- Hyperparameter Tuning  
Hyperparameters are algorithm settings. Adjusting them can significantly improve model performance.

---

# References  
1. Machine Learning Process  
2. The Ultimate Guide to Machine Learning: Feature Engineering — Part -2  
3. How to Split the Dataset into Training and Test sets  
4. Which machine learning algorithm should I use?

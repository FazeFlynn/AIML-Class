# AIML Class Notes

---


# Structure of ML:

<div style="display: flex; justify-content: center;">
<table style="text-align: center;">
  <tr>
    <th colspan="2">Supervised Learning</th>
    <th>Unsupervised Learning</th>
  </tr>
  <tr>
    <td>Classification</td>
    <td>Regression</td>
    <td>PCA</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>Linear Regression</td>
    <td>K-mean Clustering</td>
  </tr>
  <tr>
    <td>Naive Bayes</td>
    <td>Ridge Regression</td>  
    <td>Hierarchical Clustering</td>
  </tr>
  <tr>
    <td>Linear Discriminant Analysis (LDA)</td>
    <td>Lasso Regression</td>  
    <td>DB Scan Clustering</td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center;">Decision Trees</td>
    <td> </td>
  </tr>
  <tr>
    <td colspan="2">Random Forest</td>
    <td> </td>
  </tr>
  <tr>
    <td colspan="2">Support Vector Machines (SVM)</td>
    <td> </td>
  </tr>
  <tr>
    <td colspan="2">K-Nearest Neighbors (KNN)</td>
    <td> </td>
  </tr>
  <tr>
    <td colspan="2">Gradient Boosting Algorithms</td>
    <td> </td>
  </tr>
  <tr>
    <td colspan="2">Neural Networks</td>
    <td> </td>
  </tr>
</table>
</div>



---

## 29 August 2024

### Machine Learning

**Prerequisites:**
- Statistics and Probability

**Types of Machine Learning:**
1. **Supervised Learning:** 
   - Dependent (Target) and Independent (Features) variables.
2. **Unsupervised Learning:** 
   - Clustering of data.
3. **Reinforcement Learning:** 
   - Can involve both supervised and unsupervised elements. Commonly used in robotics, where the model learns through feedback from its actions.

**Types of Variables:**
1. **Numerical (Quantitative)**
2. **Categorical**
   - Unsure distinction between continuous and categorical, quantitative and qualitative, or discrete (whole numbers only).

---

## 31 August 2024

### Supervised Machine Learning

**Divided into:**
- **Regression:** Used when the target variable is continuous.
- **Classification:** Used when the target variable has a limited number of categories.

**Data Types:**
- **Training Data**
- **Testing Data**

**Types of Categorical Data:**
1. **Nominal Data**
2. **Ordinal Data**

**Note:** Check the **target (dependent variable)** to decide whether to use Regression or Classification:
- Continuous target → **Regression**
- Categorical target → **Classification**

> If the number of categories < 30 → Classification, otherwise Regression (usually decided by domain experts in companies).

---

### K-Nearest Neighbors (KNN)

**Handling Imbalanced Datasets:**
1. **Undersampling**
2. **Oversampling**
3. **SMOTE:** Synthetic Minority Over-sampling Technique to balance data by generating new samples near existing ones.

**Disadvantages of KNN:**
1. Not suitable for large datasets due to high computational complexity in calculating distances.
2. Sensitive to outliers, requiring a clean dataset.
3. Imbalanced datasets can reduce the efficiency and accuracy of the KNN algorithm.

**Distance Metrics:**
- **Euclidean Distance:** Shortest distance (based on Pythagoras' theorem).
- **Manhattan Distance:** Sum of the absolute differences along each axis.

**Train-Test Split:**
- `x_train`, `y_train`, `x_test`, `y_test`

---

## Exam Topics - 17 September 2024

**Units 1 and 2:**
- **Algorithms:** Linear Regression, KNN
- **Accuracy Metrics:** R-squared (R²) for regression, precision/recall for classification
- **Supervised, Unsupervised, and Reinforcement Learning**
- **Measures of Central Tendency:** Mean, Median, Mode
- **Types of Variables:** Numerical and Categorical, Interval Ratios
- **Statistics in Machine Learning**

---

## 17 September 2024

### Precision & R-Squared (for Regression)

1. **R-Squared (R²):**
  - Formula: 

  $$
    R^2 = 1 - \left(\frac{\text{Sum of Squared Residuals}}{\text{Total Sum of Squares}}\right)
  $$

  - Problem: It doesn't account for which features are relevant.

2. **Adjusted R-Squared:**
  - Formula:

  $$
    \text{Adjusted } R^2 = 1 - \left(\frac{(1-R^2)(N-1)}{N-P-1}\right)
  $$

  - `N`: Number of data points
  - `P`: Number of independent features

---

### Recall

**Fitting Models:**
- **Overfitting:** High train accuracy, but low test accuracy.
- **Underfitting:** Insufficient data to train the model.
- **Best Fitting:** Good accuracy for both train and test datasets.

**Errors:**
- **Bias:** Training errors.
- **Variance:** Testing errors.

---

## 18 September 2024

### Classification

| y (Actual) | ŷ (Predicted)  | 
|------------|----------------|
| 0          | 1              |
| 1          | 1              |
| 0          | 0              |
| 1          | 1              |
| 1          | 1              |
| 0          | 1              |
| 1          | 0              |
|            |                |

### Confusion Matrix:

|               | Actual 1 | Actual 0 |
|---------------|----------|----------|
| **Pred 1**    | TP = 3   | FP = 2   |
| **Pred 0**    | FN = 1   | TN = 1   |

---

## 19 September 2024

### Performance Metrics for Imbalanced Data

1. **Precision:**
  - Formula: 

  $$
    \text{Precision} = \frac{TP}{TP + FP}
  $$

2. **Recall:**
  - Formula: 

  $$
    \text{Recall} = \frac{TP}{TP + FN}
  $$

  - Used when reducing **False Negatives (FN)** is important.

3. **F-Beta Score:**
   - Combines precision and recall.

---

## 21 September 2024

### Descriptive and Inferential Statistics

#### Measures of Central Tendency:
1. **Mean:** 
  - Formula: 

  $$
    \text{Mean} = \frac{\text{Sum of All Numbers}}{\text{Count of Numbers}}
  $$

2. **Median:**
   - Arrange data in ascending order and pick the middle value.
   - Use **Median** when there are outliers.
3. **Mode:**
   - The most frequent value. Use Mode for categorical data.

#### Measures of Variance:
- **Variance:** Describes the spread of data around the mean.
- **Standard Deviation:** The square root of variance.

---

### KNN for Classification and Regression

1. **KNN for Classification:**

```bash
            |
            |     *   *
            |   * *  *    0   *  0
    Graph   |     *    *  0  X
            |    0    0    0   0        
            |   0 0     0    0
            |       0      
            -------------------------
```


# `Decision Trees`

### `Sample PLay Tennis Dataset`

| Day  | Outlook  | Temperature | Humidity | Wind  | Play Tennis |
|------|----------|-------------|----------|-------|-------------|
| D1   | Sunny    | Hot         | High     | Weak  | No          |
| D2   | Sunny    | Hot         | High     | Strong| No          |
| D3   | Overcast | Hot         | High     | Weak  | Yes         |
| D4   | Rain     | Mild        | High     | Weak  | Yes         |
| D5   | Rain     | Cool        | Normal   | Weak  | Yes         |
| D6   | Rain     | Cool        | Normal   | Strong| No          |
| D7   | Overcast | Cool        | Normal   | Strong| Yes         |
| D8   | Sunny    | Mild        | High     | Weak  | No          |
| D9   | Sunny    | Cool        | Normal   | Weak  | Yes         |
| D10  | Rain     | Mild        | Normal   | Weak  | Yes         |
| D11  | Sunny    | Mild        | Normal   | Strong| Yes         |
| D12  | Overcast | Mild        | High     | Strong| Yes         |
| D13  | Overcast | Hot         | Normal   | Weak  | Yes         |
| D14  | Rain     | Mild        | High     | Strong| No          |


### `Decision Tree`:
**Definition**: A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks. It splits the data into subsets based on the feature values, using a tree-like structure, where each internal node represents a decision (based on a feature), each branch represents the outcome of a decision, and each leaf node represents a class label (for classification) or a continuous value (for regression).

> ![Decision tree Of the Datset](/images/decisiontrees.png)




### `Formulas Of Decision Trees`:

#### `Entropy (For Small datasets)`
> ![Entropy Formula](/images/btentropy.png)

- where:
   - H(S) is the entropy of the set (S),
   - p1 is the proportion of instances in class 1,
   - p2 is the proportion of instances in class 2.
   <!-- - in the case of pure split the will be 0
   - in the case of impure split the will be 1 -->

#### `Gini Impurity (For Large Datasets)`
> ![Ginny Formula](https://latex2png.com/pngs/6cb4ca476124cf03743e4651613e01c4.png)

- where:
   - Gini(S) is the Gini impurity of the set (S).
   - c is the number of classes.
   - pi is the proportion of instances belonging to class i.

#### `Gini Impurity` - `when there are only two classes (class 1 and class 2)`
> ![Ginny2 Formula](https://latex2png.com/pngs/f86cbe93d832c32c3a84538425c56e6f.png)

- where:
   - p1 is the proportion of instances in class 1,
   - p2 is the proportion of instances in class 2.


### `Gini Example`

> ![giniExample](/images/ginniExample.png)

`Solution`:
> ![Solution](https://latex2png.com/pngs/cda73feb60862c53948c9fa484c357c8.png)





<!-- ```
GI(C1) = 1 - |P1.pow2 + p2.pow2|
       => 1 - |(3/6).pow2 + (3/6)Pow2 |
       => 1- |1/4 + 1/4|
       => 0.5
``` -->


# `Example: Choosing the Root Node by Information Gain`

Let’s consider a small dataset to explain how to choose the root node using **Information Gain**. We'll use the **Play Tennis** dataset as an example. The dataset contains the following features:

- **Outlook**
- **Temperature**
- **Humidity**
- **Wind**

Our goal is to choose the root node by calculating the **Information Gain** for each feature and selecting the one with the highest value.

---

## Dataset

| Day  | Outlook   | Temperature | Humidity | Wind  | Play Tennis |
|------|-----------|-------------|----------|-------|-------------|
| D1   | Sunny     | Hot         | High     | Weak  | No          |
| D2   | Sunny     | Hot         | High     | Strong| No          |
| D3   | Overcast  | Hot         | High     | Weak  | Yes         |
| D4   | Rain      | Mild        | High     | Weak  | Yes         |
| D5   | Rain      | Cool        | Normal   | Weak  | Yes         |
| D6   | Rain      | Cool        | Normal   | Strong| No          |
| D7   | Overcast  | Cool        | Normal   | Strong| Yes         |
| D8   | Sunny     | Mild        | High     | Weak  | No          |
| D9   | Sunny     | Cool        | Normal   | Weak  | Yes         |
| D10  | Rain      | Mild        | Normal   | Weak  | Yes         |
| D11  | Sunny     | Mild        | Normal   | Strong| Yes         |
| D12  | Overcast  | Mild        | High     | Strong| Yes         |
| D13  | Overcast  | Hot         | Normal   | Weak  | Yes         |
| D14  | Rain      | Mild        | High     | Strong| No          |

---

## Step 1: Calculate the Entropy of the Target Variable

The target variable is **Play Tennis**, which has 9 "Yes" and 5 "No" values. The entropy H(S) is calculated as follows:

`Formula`
> ![H(S) = - p_{yes} \log_2(p_{yes}) - p_{no} \log_2(p_{no})](https://latex2png.com/pngs/adeba375467475dcfdb8598128c266b1.png)


Where:
- p(yes) = 9/14
- p(no) = 5/14

$$
  H(S) = - \\frac{9}{14} \log_2 \\left( \\frac{9}{14} \\right) - \\frac{5}{14} \log_2 \\left( \\frac{5}{14} \\right)
$$

$$
  H(S) ≈ -0.642 - 0.530 ≈ 1.172
$$

---

## Step 2: Calculate Information Gain for Each Feature

The Information Gain for a feature is the difference between the entropy of the original set and the weighted entropy after splitting the dataset on that feature.

$$
IG = H(S) - \\sum \\left( \\frac{|S_i|}{|S|} H(S_i) \\right)
$$

### Feature: **Outlook**

| Outlook  | Play Tennis (Yes/No) | Proportion |
|----------|----------------------|------------|
| Sunny    | 2 Yes, 3 No           | 5/14       |
| Overcast | 4 Yes, 0 No           | 4/14       |
| Rain     | 3 Yes, 2 No           | 5/14       |

#### Entropy for Outlook:

$$
H(Outlook) ≈ 0.693
$$

#### Information Gain for Outlook:

$$
IG(Outlook) ≈ 0.479
$$

### Feature: **Temperature**

| Temperature | Play Tennis (Yes/No) | Proportion |
|-------------|----------------------|------------|
| Hot         | 2 Yes, 2 No           | 4/14       |
| Mild        | 4 Yes, 2 No           | 6/14       |
| Cool        | 3 Yes, 1 No           | 4/14       |

#### Entropy for Temperature:

$$
H(Temperature) ≈ 0.911
$$

#### Information Gain for Temperature:

$$
IG(Temperature) ≈ 0.261
$$

### Feature: **Humidity**

| Humidity | Play Tennis (Yes/No) | Proportion |
|----------|----------------------|------------|
| High     | 3 Yes, 4 No           | 7/14       |
| Normal   | 6 Yes, 1 No           | 7/14       |

#### Entropy for Humidity:

$$
H(Humidity) ≈ 0.789
$$

#### Information Gain for Humidity:

$$
IG(Humidity) ≈ 0.383
$$

### Feature: **Wind**

| Wind   | Play Tennis (Yes/No) | Proportion |
|--------|----------------------|------------|
| Weak   | 6 Yes, 2 No           | 8/14       |
| Strong | 3 Yes, 3 No           | 6/14       |

#### Entropy for Wind:

$$
H(Wind) ≈ 0.892
$$

#### Information Gain for Wind:

$$
IG(Wind) ≈ 0.280
$$

---

## Step 3: Choose the Root Node

The Information Gain values for each feature are:
- **Outlook**: 0.479
- **Temperature**: 0.261
- **Humidity**: 0.383
- **Wind**: 0.280

Since **Outlook** has the highest Information Gain, it is chosen as the root node.
"""



## `Topics under Unsupervised Learning`:
   - K-means
   - DB Scan
   - Hierarchical CLustering
   - Dimensionality Reduction

`Unsupervised Learning:` - Unsupervised learning is a type of machine learning that involves training models using data that has no labels or specific output. The goal is to uncover hidden structures, patterns, or insights from the data. Unlike supervised learning, there is no direct feedback or target variable to guide the learning process.

## `Examples of Unsupervised Learning`:

### K-means Clustering Example:

- `Scenario`: A retail store wants to segment its customers based on purchasing behavior. Using K-means, the store can divide customers into different groups, such as "high spenders," "bargain shoppers," and "frequent visitors."


### `Principal Component Analysis (PCA) Example:`

- `Scenario`: A genetics researcher is working with a dataset containing thousands of gene expression levels. Using PCA, the researcher can reduce the dimensionality of the data, making it easier to analyze and visualize clusters of genes.


### `Anomaly Detection in Network Security:`
- `Scenario`: A cybersecurity team uses unsupervised learning to identify unusual patterns in network traffic that could indicate a potential security breach.


### `Market Basket Analysis Using Apriori Algorithm:`
- `Scenario`: An e-commerce company uses association rules to find patterns in customers’ buying behavior. For example, if a customer buys bread and peanut butter, they are likely to buy jam as well.

## `Advantages of Unsupervised Learning`
- `No Need for Labeled Data`: Unsupervised learning does not require labels, making it useful for scenarios where labeling is expensive or impractical.
- `Finding Hidden Patterns`: It helps discover unknown patterns in the data, leading to new insights.
- `Data Exploration`: Useful for data exploration and understanding the underlying structure of the data.



<!-- 
## `Supervised Learning`

- `Two Types:` `Classification and Regression`

- `Classification`
   - Logistic Regression
   - Naive Bayes
   - Linear Discriminant Analysis (LDA)

- `Regression`
   - Linear Regression
   - Ridge Regression
   - Lasso Regression

- `Both in Classification and Regression`
   - Decision Trees
   - Random Forest
   - Support Vector Machines (SVM)
   - k-Nearest Neighbors (KNN)
   - Gradient Boosting algorithms
   - Neural Networks


## `Unsupervised Learning`

   - `Main Types`:
      - Clustering
      - Association
      - Dimensionality Reduction

   - PCA
   - K-mean Clustering
   - Hierarchical Clustering
   - DB Scan Clustering
 -->


<!-- 
| Supervised Learning                       | Unsupervised Learning                   |
|------------------------------------------|-----------------------------------------|
| Classification          Regression       | 2                                       |
|        1                     2           | Unlabeled data                          |
|        1                     2           | Discover hidden patterns or structures  |
|        1                     2           | K-means, PCA, Apriori Algorithm         |
|        1                     2           | Customer segmentation, anomaly detection|
|        1                     2           | Difficult to evaluate objectively       |  
-->






# `Refresher Starts`

# Machine Learning: Supervised vs. Unsupervised Learning

## 1. Supervised Learning

### Definition
Supervised learning is a type of machine learning where the model is trained using labeled data. The algorithm learns the mapping function from the input data to the output labels, with the goal of making accurate predictions on unseen data.

---

### Key Concepts

- **Training Data:** Labeled dataset used to train the model. Each example consists of an input-output pair.
- **Target Variable:** The output that the model is trained to predict.
- **Objective:** Minimize the error between the predicted and actual outputs using techniques like regression and classification.

---

### Types of Supervised Learning

1. **Classification:**
   - **Purpose:** Predict a discrete label.
   - **Examples:** 
     - Email spam detection (spam or not spam)
     - Image recognition (cat, dog, car)
   - **Algorithms:**
     - Decision Trees
     - Support Vector Machines (SVM)
     - Logistic Regression
     - k-Nearest Neighbors (k-NN)

2. **Regression:**
   - **Purpose:** Predict a continuous value.
   - **Examples:**
     - Predicting house prices
     - Forecasting stock prices
   - **Algorithms:**
     - Linear Regression
     - Polynomial Regression
     - Random Forest Regression

---

### Applications

- **Medical Diagnosis:** Predicting the presence of diseases based on symptoms.
- **Finance:** Credit scoring to determine loan eligibility.
- **Marketing:** Predicting customer churn and targeted advertising.
- **Speech Recognition:** Translating spoken language into text.

---

### Advantages

- **High Accuracy:** With quality labeled data, supervised models can achieve high predictive accuracy.
- **Interpretable:** Many algorithms, like linear regression, are easy to understand and interpret.
- **Versatile:** Can be applied to a wide range of real-world problems.

### Challenges

- **Data Dependency:** Requires large amounts of labeled data, which can be expensive and time-consuming to obtain.
- **Overfitting:** The model may perform well on training data but poorly on unseen data if it is too complex.
- **Limited by Labels:** Performance is constrained by the quality and quantity of the labeled data.

---

## 2. Unsupervised Learning

### Definition
Unsupervised learning involves training models on data that has no labels. The goal is to identify patterns, group similar data points, or reduce data dimensionality without any specific guidance.

---

### Key Concepts

- **Training Data:** Unlabeled dataset used to uncover hidden structures.
- **Objective:** Learn the underlying distribution or structure of the data.

---

### Types of Unsupervised Learning

1. **Clustering:**
   - **Purpose:** Group similar data points into clusters.
   - **Examples:**
     - Customer segmentation for targeted marketing
     - Document classification
   - **Algorithms:**
     - K-means Clustering
     - Hierarchical Clustering
     - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

2. **Dimensionality Reduction:**
   - **Purpose:** Reduce the number of features while preserving as much information as possible.
   - **Examples:**
     - Image compression
     - Visualizing high-dimensional data
   - **Algorithms:**
     - Principal Component Analysis (PCA)
     - t-Distributed Stochastic Neighbor Embedding (t-SNE)
     - Autoencoders

3. **Anomaly Detection:**
   - **Purpose:** Identify data points that deviate from the normal pattern.
   - **Examples:**
     - Fraud detection in credit card transactions
     - Intrusion detection in network security
   - **Algorithms:**
     - Isolation Forest
     - Gaussian Mixture Models (GMM)
     - Local Outlier Factor (LOF)

4. **Association Rule Learning:**
   - **Purpose:** Find relationships between variables in large datasets.
   - **Examples:**
     - Market basket analysis (e.g., "Customers who buy bread also buy butter")
   - **Algorithms:**
     - Apriori Algorithm
     - Eclat Algorithm

---

### Applications

- **Market Segmentation:** Understanding customer groups for personalized marketing.
- **Data Compression:** Reducing file sizes in multimedia applications.
- **Anomaly Detection:** Identifying outliers in manufacturing processes or network security.
- **Recommendation Systems:** Grouping similar items or users for content recommendations.

---

### Advantages

- **No Need for Labeled Data:** Useful when labeled data is unavailable or expensive to collect.
- **Data Exploration:** Helps in understanding the structure and patterns in the data.
- **Versatility:** Applicable to various fields, including image processing, genetics, and natural language processing.

### Challenges

- **Interpretability:** Results are often difficult to interpret, especially with complex data.
- **Evaluation Metrics:** There is no definitive way to measure the performance of the model.
- **Scalability:** Some algorithms may struggle with very large datasets.

---

## Key Differences Between Supervised and Unsupervised Learning

| Aspect                    | Supervised Learning                       | Unsupervised Learning                   |
|---------------------------|-------------------------------------------|-----------------------------------------|
| **Data Type**             | Labeled data                              | Unlabeled data                          |
| **Objective**             | Make predictions or classifications       | Discover hidden patterns or structures  |
| **Algorithms**            | Linear Regression, SVM, Decision Trees    | K-means, PCA, Apriori Algorithm         |
| **Applications**          | Email filtering, loan prediction          | Customer segmentation, anomaly detection|
| **Evaluation**            | Measured using metrics like accuracy      | Difficult to evaluate objectively       |

---

## Conclusion

Both supervised and unsupervised learning have unique strengths and are used in different scenarios. Supervised learning is effective when labeled data is available, while unsupervised learning is valuable for exploring and understanding unlabeled data.

# `Refresher Ends`

# `Test`

## Random Forest: An Overview 

### What is Random Forest? 
**Random Forest**  is an ensemble learning method used for both classification and regression tasks. It operates by constructing multiple decision trees during training and outputs the average prediction (regression) or the majority vote (classification) of the individual trees.

---


### How Does Random Forest Work? 
 
1. **Bootstrap Sampling** :
  - The algorithm creates several subsets of the original dataset using sampling with replacement. Each subset is used to train an individual decision tree.
 
2. **Feature Selection** :
  - When building each decision tree, a random subset of features is chosen for splitting at each node. This introduces more diversity among trees and helps to reduce overfitting.
 
3. **Tree Construction** :
  - Each decision tree is grown to its maximum depth without pruning. The trees may become highly specialized and overfit the training data individually.
 
4. **Prediction Aggregation** : 
  - **For Classification** : Each tree votes for a class, and the class with the most votes becomes the model's prediction.
 
  - **For Regression** : The predictions of all trees are averaged to get the final output.


---


### Key Concepts 
 
1. **Ensemble Learning** :
  - Random Forest is based on the concept of ensemble learning, where multiple models (decision trees) are combined to improve overall performance.
 
2. **Bagging (Bootstrap Aggregating)** :
  - This technique helps in reducing variance and improving the stability of the model by training each tree on a random subset of the data.
 
3. **Feature Randomness** :
  - Introducing randomness in the feature selection reduces correlation among trees, further enhancing model generalization.


---


### Advantages of Random Forest 
 
- **Reduced Overfitting** : By averaging the results of multiple trees, the model becomes less likely to overfit the training data compared to individual decision trees.
 
- **Handles Missing Data** : It can maintain accuracy even when a significant portion of the data is missing.
 
- **Robust to Noise** : It is relatively robust to outliers and noise in the data.
 
- **Feature Importance** : Provides a way to evaluate the importance of each feature in making predictions.


---


### Disadvantages of Random Forest 
 
- **Computational Complexity** : Training multiple trees can be time-consuming, especially with large datasets.
 
- **Memory Usage** : It can require substantial memory due to the large number of decision trees.
 
- **Less Interpretability** : Compared to individual decision trees, understanding and interpreting a random forest model can be challenging.


---


### Hyperparameters in Random Forest 
 
1. **Number of Trees (`n_estimators`)** :
  - The number of decision trees to be built. A higher number generally improves performance but increases training time.
 
2. **Maximum Depth (`max_depth`)** :
  - The maximum depth of each tree. Controlling this can help in avoiding overfitting.
 
3. **Minimum Samples Split (`min_samples_split`)** :
  - The minimum number of samples required to split a node. Increasing this can prevent overfitting.
 
4. **Minimum Samples Leaf (`min_samples_leaf`)** :
  - The minimum number of samples required to be at a leaf node. Higher values can smooth the model.
 
5. **Maximum Features (`max_features`)** :
  - The maximum number of features considered for splitting a node. It introduces randomness and helps in reducing overfitting.


---


### Applications of Random Forest 
 
- **Medical Diagnosis** : Identifying diseases based on various symptoms and medical history.
 
- **Finance** : Credit scoring, fraud detection, and stock price prediction.
 
- **Image and Speech Recognition** : Classification tasks where high accuracy is crucial.
 
- **E-commerce** : Customer segmentation, product recommendation, and sentiment analysis.


---


### Example Use Case 
**Classification Example** :
Suppose you have a dataset of email messages and you want to build a spam filter. A Random Forest model would learn from features like the presence of certain words and message metadata, then classify each email as spam or not spam.**Regression Example** :
Predicting house prices based on features like location, number of bedrooms, square footage, etc. Random Forest aggregates the predictions from all trees to give a more accurate price estimate.

---


### Python Code Example for Random Forest 


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```


---


### Feature Importance in Random Forest 

Random Forest provides an estimate of the importance of each feature in making predictions. This is useful for understanding which features are most influential in the model.


---


### Conclusion 

Random Forest is a powerful and flexible model that works well for many tasks. Its ability to handle large datasets and provide feature importance insights makes it a popular choice for both regression and classification problems. However, it is essential to carefully tune hyperparameters to balance performance and computational efficiency.





# `Hierarchical Clustering`:

**Hierarchical clustering** is an **unsupervised learning**  algorithm used for clustering data points into a hierarchy of clusters. It is commonly used in exploratory data analysis when the number of clusters is unknown. The goal is to create a dendrogram (tree-like diagram) that visually represents the nested grouping of data.

---

**1. Types of Hierarchical Clustering** 
There are two main types of hierarchical clustering:
 
1. **Agglomerative Hierarchical Clustering (Bottom-Up Approach):** 
    - Starts with each data point as its own cluster.
    - Merges the closest clusters iteratively until a single cluster is formed.
    - Most common type of hierarchical clustering.
 
2. **Divisive Hierarchical Clustering (Top-Down Approach):**
    - Starts with all data points in a single cluster.
    - Splits the clusters iteratively until each data point is its own cluster.


**2. Distance Metrics** In hierarchical clustering, the similarity between data points or clusters is determined using **distance metrics** . Commonly used distance metrics include: 
- **Euclidean Distance** :

$$
  \large{d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}} 
$$
 
- **Manhattan Distance** :

$$
  d(x, y) = \sum_{i=1}^n |x_i - y_i| 
$$
 
- **Cosine Similarity** :

$$
  \text{similarity} = \frac{x \cdot y}{\|x\| \|y\|} 
$$


**3. Linkage Methods** 
Linkage methods determine how the distance between clusters is calculated:
 
1. **Single Linkage (Minimum Linkage):** 
    - Distance between two clusters is the minimum distance between any two points, one from each cluster.
    - Can result in “chaining,” where clusters form elongated shapes.
 
2. **Complete Linkage (Maximum Linkage):** 
    - Distance between two clusters is the maximum distance between any two points, one from each cluster.
    - Tends to create more compact clusters.
 
3. **Average Linkage:** 
    - Distance between two clusters is the average of all pairwise distances between points in the two clusters.
 
4. **Ward’s Linkage:** 
    - Minimizes the total within-cluster variance.
    - Generally produces clusters of similar size.


**4. Dendrogram** A **dendrogram**  is a tree-like diagram used to represent the hierarchical structure of clusters. The height of the branches represents the distance or dissimilarity between clusters. 
- **Cutting the Dendrogram** : By cutting the dendrogram at a certain height, you can choose the number of clusters.


**5. Steps in Hierarchical Clustering** 
### Agglomerative Hierarchical Clustering: 
 
1. **Compute Distance Matrix** : Calculate the pairwise distances between data points. 
2. **Merge Clusters** : Find the two closest clusters and merge them. 
3. **Update Distance Matrix** : Recalculate the distances between the new cluster and the remaining clusters using a linkage method.
4. **Repeat** : Continue merging until a single cluster remains.
5. **Create Dendrogram** : Plot the hierarchical structure of the clusters.


**6. Example in Python** 

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=10, centers=3, random_state=42)

# Perform hierarchical clustering using Ward's method
Z = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(8, 4))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
```

### Explanation: 
 
- **`make_blobs`**  creates synthetic data points.
- **`linkage`**  performs hierarchical clustering using Ward’s method.
- **`dendrogram`**  visualizes the hierarchical structure of clusters.


**7. Advantages of Hierarchical Clustering**  
- **No need to specify the number of clusters**  in advance (unlike K-Means).
- **Dendrogram**  provides a clear visual representation of the hierarchy of clusters.- Can handle non-spherical cluster shapes.

**8. Disadvantages of Hierarchical Clustering**  
- **Computationally expensive**  for large datasets ( $O(n^2 \log n)$ complexity).
- **Not robust to noise**  and outliers.
- Difficult to undo a merge (agglomerative) or a split (divisive) once made.


**9. Applications of Hierarchical Clustering**  
- **Gene expression analysis** : Group similar genes or samples based on expression patterns.
- **Document clustering** : Organize documents into hierarchies based on content similarity.
- **Market segmentation** : Group customers based on purchasing behavior.


---





# `K-Means Clustering` 
[Refresher for K-Means Clustering](https://youtu.be/CLKW6uWJtTc?si=oU2h6lLe_fS9XDX1) {target="_blank"}


**K-Means** Clustering is a popular **unsupervised learning**  algorithm used for **partitioning data**  into a specified number of clusters (K). The goal is to group data points into clusters such that points in the same cluster are more similar to each other than to those in other clusters. The algorithm iteratively refines the cluster centers (centroids) to minimize the sum of squared distances between each point and its nearest centroid.

**1. How K-Means Clustering Works** 
The K-Means algorithm follows these steps:
 
1. **Choose the Number of Clusters (K)** :
  - Decide the number of clusters, K.
  - This is a hyperparameter that needs to be chosen in advance.
 
2. **Initialize Centroids** :
  - Randomly select K data points as initial cluster centroids.
 
3. **Assign Data Points to Clusters** :
  - For each data point, calculate its distance to each centroid.
  - Assign the point to the cluster with the nearest centroid.
 
4. **Update Centroids** :
  - Calculate the new centroids as the mean of all points assigned to each cluster.
 
5. **Repeat** :
  - Repeat the assignment and update steps until the centroids no longer change significantly or a maximum number of iterations is reached.
 
6. **Converge** :
  - The algorithm stops when the centroids stabilize.


**2. Distance Metrics** The most common distance metric used in K-Means Clustering is **Euclidean distance** :

$$
 \text{Distance} = \sqrt{\sum_{i=1}^n (x_i - y_i)^2} 
$$


**3. Example of K-Means Clustering in Python** 

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Apply K-Means with K=4
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
**Explanation** : 
  - **`make_blobs`**  generates a dataset with four distinct clusters.
  - **`KMeans`**  is used to fit the model and assign cluster labels.
  - **`cluster_centers_`**  provides the coordinates of the centroids.


**4. Choosing the Number of Clusters (K)** 
Selecting the right value of K is crucial for effective clustering. Common methods include:
 
1. **Elbow Method** :
  - Plot the sum of squared errors (SSE) for different values of K.
  - The optimal K is often where the SSE starts to decrease more slowly (the "elbow").

$$
 \text{SSE} = \sum_{i=1}^{n} \|x_i - c_j\|^2 
$$
 
2. **Silhouette Score** :
  - Measures how similar a point is to its own cluster compared to other clusters.
  - A higher score indicates better clustering.


**5. Advantages of K-Means Clustering**  

  - **Simple and easy to implement** . 
  - **Scales well**  to large datasets. 
  - Efficient with a time complexity of $O(n \cdot k \cdot i)$, where n is the number of data points, k is the number of clusters, and i is the number of iterations.



**6. Disadvantages of K-Means Clustering**  
  - **Requires specification of K** : The number of clusters must be chosen beforehand. 
  - **Sensitive to initial centroids** : Poor initialization can lead to suboptimal clustering. 
  - **Assumes spherical clusters** : Struggles with non-spherical cluster shapes. 
  - **Not robust to noise and outliers** : Outliers can significantly affect the centroids.


**7. Limitations and Solutions**  
1. **Limitation** : K-Means is sensitive to the initial placement of centroids. 
  - **Solution** : Use the **K-Means++ initialization** , which selects initial centroids in a smart way to speed up convergence and improve accuracy.
 
2. **Limitation** : Struggles with clusters of varying densities or non-spherical shapes. 
  - **Solution** : Use density-based clustering algorithms like **DBSCAN**  or hierarchical clustering.



**8. Applications of K-Means Clustering**  
  - **Customer Segmentation** : Grouping customers based on purchasing behavior. 
  - **Image Compression** : Reducing the number of colors in an image by clustering pixel colors. 
  - **Anomaly Detection** : Identifying unusual data points as outliers. 
  - **Document Clustering** : Grouping similar documents based on text features.


**9. Practical Tips for Using K-Means**  
  - **Standardize your data**  before applying K-Means, especially if features have different scales. 
  - **Use the Elbow Method**  or **Silhouette Score**  to determine an optimal value for K. 
  - **Initialize centroids using K-Means++**  to avoid poor convergence.


---

**10. Comparison with Other Clustering Algorithms**
| Feature | K-Means | DBSCAN | Hierarchical Clustering | 
| --- | --- | --- | --- | 
| Number of Clusters | Must be specified | Not required | Can be decided using dendrogram | 
| Cluster Shape | Spherical | Arbitrary | Arbitrary | 
| Noise Handling | Poor | Robust (identifies noise) | Poor | 
| Scalability | High | Moderate | Low | 


#### `Practical Example of K-Means`

<details>
  <summary>Practical Example of K-Means</summary>


##### Let us take Practical example  of K-Means clustering, including manual centroid selection, distance calculation, and updating the centroids.**Problem Statement** We have a small dataset with two features: **Height (in cm)**  and **Weight (in kg)** . We want to cluster these points into **2 clusters (K = 2)**  using the K-Means algorithm.**Step 1: Dataset**

| Index | Height (in cm) | Weight (in kg) | 
| --- | --- | --- | 
| 1 | 150 | 50 | 
| 2 | 160 | 55 | 
| 3 | 170 | 65 | 
| 4 | 180 | 70 | 
| 5 | 155 | 52 | 
| 6 | 165 | 60 | 

**Step 2: Initial Centroid Selection** 
Let's randomly choose the first two data points as our initial centroids:
 
- **Centroid 1** : (150, 50) 
- **Centroid 2** : (160, 55)

**Step 3: Calculate Euclidean Distance** 
For each data point, we calculate the Euclidean distance to each centroid.

$$
 \text{Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} 
$$

**Distance calculations** : 
- For **Point 1**  (150, 50): 
  - Distance to Centroid 1: $\sqrt{(150-150)^2 + (50-50)^2} = 0$ 
  - Distance to Centroid 2: $\sqrt{(160-150)^2 + (55-50)^2} = \sqrt{100 + 25} = \sqrt{125} = 11.18$ 
  - Assign to **Cluster 1**  (closer to Centroid 1).
 
- For **Point 2**  (160, 55): 
  - Distance to Centroid 1: $\sqrt{(160-150)^2 + (55-50)^2} = 11.18$
  - Distance to Centroid 2: $\sqrt{(160-160)^2 + (55-55)^2} = 0$ 
  - Assign to **Cluster 2**  (closer to Centroid 2).
 
- For **Point 3**  (170, 65): 
  - Distance to Centroid 1: $\sqrt{(170-150)^2 + (65-50)^2} = \sqrt{400 + 225} = \sqrt{625} = 25$
  - Distance to Centroid 2: $\sqrt{(170-160)^2 + (65-55)^2} = \sqrt{100 + 100} = \sqrt{200} = 14.14$
  - Assign to **Cluster 2** .
 
- For **Point 4**  (180, 70): 
  - Distance to Centroid 1: $\sqrt{(180-150)^2 + (70-50)^2} = \sqrt{900 + 400} = \sqrt{1300} = 36.06$
  - Distance to Centroid 2: $\sqrt{(180-160)^2 + (70-55)^2} = \sqrt{400 + 225} = \sqrt{625} = 25$
  - Assign to **Cluster 2** .
 
- For **Point 5**  (155, 52): 
  - Distance to Centroid 1: $\sqrt{(155-150)^2 + (52-50)^2} = \sqrt{25 + 4} = \sqrt{29} = 5.39$ 
  - Distance to Centroid 2: $\sqrt{(155-160)^2 + (52-55)^2} = \sqrt{25 + 9} = \sqrt{34} = 5.83$
  - Assign to **Cluster 1** .
 
- For **Point 6**  (165, 60): 
  - Distance to Centroid 1: $\sqrt{(165-150)^2 + (60-50)^2} = \sqrt{225 + 100} = \sqrt{325} = 18.03$ 
  - Distance to Centroid 2: $\sqrt{(165-160)^2 + (60-55)^2} = \sqrt{25 + 25} = \sqrt{50} = 7.07$ 
  - Assign to **Cluster 2** .

**Step 4: Cluster Assignment**

| Index | Height | Weight | Assigned Cluster | 
| --- | --- | --- | --- | 
| 1 | 150 | 50 | Cluster 1 | 
| 2 | 160 | 55 | Cluster 2 | 
| 3 | 170 | 65 | Cluster 2 | 
| 4 | 180 | 70 | Cluster 2 | 
| 5 | 155 | 52 | Cluster 1 | 
| 6 | 165 | 60 | Cluster 2 | 

**Step 5: Update Centroids** 
Calculate the new centroids by taking the mean of the points in each cluster:
 
- **New Centroid 1** : Mean of points in Cluster 1: 
  - Height: $(150 + 155) / 2 = 152.5$
  - Weight: $(50 + 52) / 2 = 51$
  - New Centroid 1: (152.5, 51)
 
- **New Centroid 2** : Mean of points in Cluster 2: 
  - Height: $(160 + 170 + 180 + 165) / 4 = 168.75$
  - Weight: $(55 + 65 + 70 + 60) / 4 = 62.5$
  - New Centroid 2: (168.75, 62.5)
  
**Step 6: Repeat the Process** 
We repeat the distance calculation and cluster assignment steps with the new centroids until the centroids do not change (convergence).
**Final Clusters** 
After a few iterations, the centroids stabilize, and we obtain the final clusters:

| Index | Height | Weight | Final Cluster | 
| --- | --- | --- | --- | 
| 1 | 150 | 50 | Cluster 1 | 
| 2 | 160 | 55 | Cluster 2 | 
| 3 | 170 | 65 | Cluster 2 | 
| 4 | 180 | 70 | Cluster 2 | 
| 5 | 155 | 52 | Cluster 1 | 
| 6 | 165 | 60 | Cluster 2 | 

**Visualization**  
- **Cluster 1** : Represents shorter individuals with lower weight.
- **Cluster 2** : Represents taller individuals with higher weight.

This manual step-by-step example shows the core mechanism of the K-Means algorithm. In practice, libraries like **Scikit-Learn**  perform these steps efficiently.

`program here`

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Apply K-Means with K=4
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

</details>

---


# 2nd Insem


### **1. Random Forest**
#### **Definition**  
Random Forest is an ensemble machine learning method that builds multiple decision trees during training and merges their outputs for better accuracy and stability.

#### **Types**  
- **Classification**: Predicts categorical labels.  
- **Regression**: Predicts continuous numerical values.  

#### **Formula**  
Random Forest works by averaging results or taking majority votes:  
1. **For Classification**: 

$$
\hat{y} = \text{Mode}(T_1(x), T_2(x), ..., T_n(x))
$$  

   where $T_i(x)$ is the prediction from the $i^{th}$ decision tree.  

2. **For Regression**: 

$$
\hat{y} = \frac{1}{n} \sum_{i=1}^{n} T_i(x)
$$

#### **Real-Life Example**  
- **Use Case**: Predicting if a loan applicant will default.  
- **Scenario**: A bank analyzes historical data (income, credit score, debt) and builds a Random Forest model to predict loan default.

#### **Code Example** (Python)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
predictions = rf.predict(X_test)
print("Predictions:", predictions)
```

---

### **2. Decision Tree**
#### **Definition**  
A Decision Tree is a flowchart-like structure where each internal node represents a feature test, each branch represents a decision outcome, and each leaf node represents a final prediction.

#### **Types**  
1. **Classification Tree**: Classifies data into categories.  
2. **Regression Tree**: Predicts numerical values.  

#### **Formulas**  
- **Entropy**:  

$$
H(S) = - \sum_{i=1}^n p_i \log_2(p_i)
$$  

- **Gini Impurity**: 

$$
G = 1 - \sum_{i=1}^n p_i^2
$$ 

- **Information Gain**:  

$$
IG = H(S) - \sum_{i=1}^k \frac{|S_k|}{|S|} H(S_k)
$$

#### **Real-Life Example**  
- **Use Case**: Approving credit card applications.  
- **Scenario**: A bank uses historical data (age, income, etc.) and builds a decision tree to automate credit card approval decisions.

#### **Code Example** (Python)
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train model
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

# Predict and evaluate
predictions = dt.predict(X_test)
print("Predictions:", predictions)
```

---

### **3. Unsupervised Learning**
Unsupervised learning algorithms learn patterns from unlabeled data.

---

#### **3.1 K-Means Clustering**
#### **Definition**  
An algorithm that partitions data into $k$ clusters based on the proximity of data points to centroids.

#### **Formula**  
Objective function (Minimize within-cluster variance):  

$$
J = \sum_{i=1}^k \sum_{x \in C_i} ||x - \mu_i||^2
$$

#### **Real-Life Example**  
- **Use Case**: Customer segmentation in marketing.  
- **Scenario**: Group customers based on age, spending, and income.

#### **Code Example**  
```python
from sklearn.cluster import KMeans
import numpy as np

# Sample data
data = np.array([[150, 50], [160, 55], [170, 65], [180, 70], [155, 52], [165, 60]])

# Train model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)

# Get labels and centroids
print("Labels:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)
```

---

#### **3.2 DBSCAN**
#### **Definition**  
Density-Based Spatial Clustering of Applications with Noise. Groups points close to each other based on density, ignoring noise.

#### **Real-Life Example**  
- **Use Case**: Identifying geographic hotspots for crimes.  
- **Scenario**: Detect dense regions of crime in city data to allocate police resources.

#### **Code Example**  
```python
from sklearn.cluster import DBSCAN
import numpy as np

# Sample data
data = np.array([[1, 2], [2, 3], [2, 2], [8, 8], [8, 9], [25, 80]])

# Train model
dbscan = DBSCAN(eps=3, min_samples=2)
dbscan.fit(data)

print("Labels:", dbscan.labels_)
```

---

#### **3.3 Hierarchical Clustering**
#### **Definition**  
A clustering algorithm that creates a dendrogram representing nested groupings of data points.

#### **Real-Life Example**  
- **Use Case**: Analyzing genetic similarities.  
- **Scenario**: Grouping species based on DNA similarity.

#### **Code Example**  
```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.array([[1, 2], [2, 3], [2, 2], [8, 8], [8, 9]])

# Perform clustering
linked = linkage(data, method='ward')

# Plot dendrogram
dendrogram(linked)
plt.show()
```

---

### **4. Dimensionality Reduction**
#### **Definition**  
A technique to reduce the number of features in a dataset while retaining meaningful information.

#### **Types**  
1. **PCA (Principal Component Analysis)**: Transforms data to a new set of axes.  
2. **t-SNE**: For visualization of high-dimensional data.  
3. **LDA (Linear Discriminant Analysis)**: Optimized for classification tasks.

#### **Formula for PCA**  

$$
Z = XW
$$  

Where $W$ is the matrix of eigenvectors.

#### **Real-Life Example**  
- **Use Case**: Visualizing customer preferences.  
- **Scenario**: Compressing 100+ customer attributes into 2 dimensions for analysis.

#### **Code Example** (PCA)
```python
from sklearn.decomposition import PCA
import numpy as np

# Sample data
data = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]])

# Apply PCA
pca = PCA(n_components=1)
reduced_data = pca.fit_transform(data)

print("Reduced Data:", reduced_data)
```
$$
\Large \text{2nd Isem Ends Here}
$$


---


$$
\Large \text{End Of File}
$$




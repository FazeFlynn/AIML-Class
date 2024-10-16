

## On 29 August 2024
-----------------------------------------
Machine Learning:- 
    Prerequisites:
        statistics and probablity

Types:
    Supervised: Dependent and independent (Target and features).
    Unsupervised: Clustering of data.
    Reinforcement: Can include both (often used in robotics). We give machine/models some data and instructions then it learns itself. if not then grounded metaphorically.

Types of Variables:
    Numerical aka quatative
    Categorical ------------ not sure (contineous and categorical, quantative and qualitative, discrete (only whole number))


On 31 August 2024
------------------------------------------------

Supervised ML:- is divided into **Regression** and **Classification**
When we have Contineous data we use Regression
and when we have we have limited number of categories then we use Classification

For the Model the data is divided into two types:-
                                                Training data
                                                Testing data
Categorical Data is of Two types:-
                                Nominal data
                                Ordinal data

*Homework - Interval and Ratio*


*We have to look at target(Dependent variable) variable to determine which SML is used, if the target variable is contineous then used Regression if not then otherwise*

*if number of categories < 30 then classification otherwise Regression (in comanies it will be decided by domain expert)*

//Hyper Parameters


# KNN - k nearest neighbours
----------------------------------------------------------------

Imbalanced dataset - when we have one catergory more than the other

Techniques to handle unbalanced dataset - Undersampling
                                          Oversampling
                                          Smote - it picks some near variables (data points) to balance data


*knn cannot handle outliers*

Equilidian Distance (uses pythagoras) shortest distance
and the base and perpendicular distance is called manhatten distance


Train test split - x train, y train, x test, y test

Disadvantages - 
1. cannot use it when the data is big because calculating distance will become complex
2. does handle outliers so it needs a clean dataset free from any outliers
3. Imbalanced dataset can impact the efficiency and accuracy of knn algorithm 
For




# Topics for Exam - 17 sept 2024
--------------------------------------------------
unit 1 and 2

alogs - lineaer regression and knn
accuracy metrices (regression and classification) rsquare or r2score

superwised and unsuperwise, reinforcement
mean 
median 
mode 
central tendency
variable types (numerical and categorical) and its types, interval ratios
machine learning and stats



17 Sept 2024
---------------------------------------------------------
medium website have all the concept of ml and dl


Precision - rsquare and adjusted r2, We use both of these in case of regression
    R Square:-
        r2  = 1 - [ sum of square of residuals(ss resi) / sum of square of total(ss total) ]
        r2 has one problem i doesnot know relevant features to choose.

    Adjusted R Square:-
        Ar2 = 1 - [ (1-r2)(N-1) / N-P-1]
        p = no of independent featues
        n = no of data points


Recall - 

OverFitting, UnderFitting, BestFitting - 
    OverFitting - Train accuracy will be high but test accuracy will be low.
    UnderFiiting - we dont have sufficient data to train our model.
    BestFillting - both train and test  accuracy are good.


Training errors are called bias and testing errors are called variance




18 Sept 2024
--------------------------------------------------------------------------------------------
For Classification:

y = original
y cap = predicted
y overscore = mean

```bash

y       ycap        yoverscore
0       1
1       1
0       0
1       1
1       1
0       1
1       0

confusion matrix:-
                                     actual 
                                1             0
                         ---------------------------------
                         |               |               |
                     1   |       3       |        2      |
                         |               |               |
         Predicted       ---------------------------------
                         |               |               |
                         |        1      |          1    |
                     0   |               |               |
                         ---------------------------------
                                    actual                      These Matrices Could be wrong (Make sure you double check)
                                 1             0
                         ---------------------------------
                         |               |               |
                     1   |       tp      |        fp     |
                         |               |               |
         Predicted       ---------------------------------
                         |               |               |
                         |        tn     |          fn   |
                     0   |               |               |
                         ---------------------------------


Balanced Data:- 
                (tp+tn)/(tp+fp+fn+tn) = (3+1)/(3+2+1+1) = .57
```


19 Sept 2024
-----------------------------------------------------------------------------------------------------

# Performance matrices for imbalanced data


1. Precision:-
 - tp/tp+fn
```bash

                           t             f
                   ---------------------------------
                   |               |               |
                p  |       tp      |        fp     |
                   |               |               |
    Predicted      ---------------------------------    This Matrix Could be wrong (Make sure you double check)
                   |               |               |
                   |        fn     |        tn     |
                n  |               |               |
                   ---------------------------------
```

2. Recall:-
        fn/tp+fn
        if we want to reduce fn we use recall        

3. F-Beta:-




21 Sept 2024
------------------------------------------------------------------------------------
# Descriptive and Infrential Stats (Advanced)


# Measure of Central Tendancy

Descriptive stats - Mean, Median, Mode, std etc, that can be used to fill missing values in the data.

        Mean = sum of all numbers / counts of number

Medain = arrange them into ascending order and then the pick the middle value
When we have outliers in our data we use Median

Mode = Frequency
when we have categorical data then we use Mode


# Measure of Variance
Distribution of data around Central Tendency

    Variance = img in phone

Center Deviation = UnderRoot of Variance



# Precision, Recall, and F-Beta

F-Beta = On the phone

B = 1, If both are important
B = .5 If FP>>FN
B = 2 If FN>>FP



# KNN - Used for Both Regression and Classification

KNN for Classification

when we plotted data on graph

```bash
            |
            |     *   *
            |   * *  *    0   *  0
    Graph - |     *    *  0  X
            |    0    0    0   0        Graph Could be wrong (Make sure you double check)
            |   0 0     0    0
            |       0      
            __________________________ 
```


The category that we have more near to x then x will belong to that category - here in the case x is 0 because there are many zeros near x instead of *


KNN for Regression
```bash

            |
            | p  p   p
            |  p   p
    Graph - |    p   x   p             Graph Could be wrong (Make sure you double check)
            |  p   p
            |
            ______________________
```
In Regression we will take Average of the dataPoints near to x


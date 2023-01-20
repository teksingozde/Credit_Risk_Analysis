# Credit Risk Analysis
## Overview of Project
A client is looking for a date-based and highly reliable method of estimating the credit risk of candidates who can already take out loans about their company. Supervised machine learning models were used to identify risks based on credit card information obtained from LendingClub. The ideal model, in other words the fit model, should both have the data of a large number of candidates and give the highest reliability rate.

### Resources
#### Data Sources:
- LoanStats_2019Q1.csv
- credit_risk_ensemble.ipynb
- credit_risk_resampling.ipynb

#### Software Sources:
- scikit-learn 1.0.2
- imbalanced-learn 0.10.1
- Pandas 1.3.5
- Numpy 1.21.5

## Overview of Analysis
#### Ensemble Machine Learning
Ensemble models in machine learning work in the same way as we exemplified above. They combine decisions from multiple models to improve overall performance.
- Maximum Voting: Usually used in classification problems.

- Averaging: It can be used to make estimations in regression problems or when calculating probabilities for classification problems.

- Weighted Averaging: It is an extension of the average method. All models have different weights that define the importance of each model for prediction. For example, your colleagues' answers about your design are given more importance than others' answers.

- Stacking: It is an ensemble learning technique that uses predictions from multiple models (for example, decision tree, KNN, or SVM) to build a new model. This model is used to make predictions on the test set.

- Blending: Follows the same approach as stacking, but only uses a validation set by the training set to make predictions.

##### Techniques

###### 1. Bagging
The bootstrap aggregating method was developed by Breiman in 1996. An ensemble is formed by applying estimators to bootstrapped samples obtained from the original dataset. Here, the bootstrapping application is used to generate subsamples by performing a return random selection.
The subsamples created will be the same as the number in the original data set. For this reason, some observations are not included in the samples generated as a result of bootstrapping, while others may be seen two or more times. In the merging of the estimates, the average is taken for the regression trees, while the results in the classification trees are determined by voting.
- Multiple subsets are created from the original dataset.
- A basic model (weak model) has been established in each of these subgroups.
- Models run in parallel and are independent of each other.
- Final estimates are determined by combining estimates from all models.

###### 2. Boosting
The basic idea in the amplification method is to make inferences from the collection of trees obtained as a result of giving different weights to the data set. Initially, all observations have equal weight. As the tree community begins to grow, weightings are adjusted based on problem information. The weight of misclassified observations is increased, while the weight of rarely misclassified observations is decreased. In this way, trees gain the ability to regulate themselves in the face of difficult situations.

##### Algorithms Based on Bagging and Upgrading Techniques
Bagging and Boosting are two of the most widely used techniques in machine learning. In this section, we will look at them in detail. Below are the algorithms we will focus on:
###### Bagging algorithms:
- Bagging meta-estimator
- Random Forest
###### Upgrade algorithms:
- AdaBoost
- GBM
- XGBM

#### Resampling Machine Learning
Traditional inferential statistical methods produce the sampling distribution using a sample of size “n” obtained from the population and the characteristics of the sample when certain assumptions are met.

Although the samples provide a cost and time advantage, they cannot show the whole picture because they are a part of the population, not the whole. Statistics of each sample formed from the population differs from both the other samples formed from the population and the population parameter.
Often, we do not know the parameters (mean, standard deviation, etc.) of the population we are inferring about, and we try to estimate these parameters. To prove that sample statistics differ from population parameters, we will generate random numbers representing the population and select samples of the same size from them and compare their averages.

We can derive a statistic from a single sample. But with this statistic, different questions emerge that need to be answered:
- Is this value statistically significant or did it occur by chance?
- How is it different from other samples?
- How confident can we be in this result?
So far, we have assumed that the sample and statistics from the sample satisfy the assumptions of traditional methods and that the samples are "large enough". In real-world datasets:
- It is possible that the sample sizes are small or the sample distribution is uncertain, which does not comply with the assumptions of the traditional method.
- Situations where it is impossible or too expensive to collect new data are also possible.

Bootstrap sampling method is an effective and at the same time practical resampling method, which requires minimal assumptions about the data set, where the sample size is small or it is not possible to collect new data. It is also useful as an alternative to the traditional method in terms of its practicality.

Resampling methods generally offer the following advantages:
- It helps to make statistical inferences from a small data set when iteration of research is expensive or time consuming.
- Helps test how machine learning models can perform on a new dataset without collecting extra data.##

##### Techniques

##### 1. Random Oversampling
Random oversampling involves randomly duplicating examples from the minority class and adding them to the training dataset.
Examples from the training dataset are selected randomly with replacement. This means that examples from the minority class can be chosen and added to the new “more balanced” training dataset multiple times; they are selected from the original training dataset, added to the new training dataset, and then returned or “replaced” in the original dataset, allowing them to be selected again.
This technique can be effective for those machine learning algorithms that are affected by a skewed distribution and where multiple duplicate examples for a given class can influence the fit of the model. This might include algorithms that iteratively learn coefficients, like artificial neural networks that use stochastic gradient descent. It can also affect models that seek good splits of the data, such as support vector machines and decision trees.
It might be useful to tune the target class distribution. In some cases, seeking a balanced distribution for a severely imbalanced dataset can cause affected algorithms to overfit the minority class, leading to increased generalization error. The effect can be better performance on the training dataset, but worse performance on the holdout or test dataset.

##### 2. SMOTE Oversampling
Imbalanced classification involves developing predictive models on classification datasets that have a severe class imbalance.
The challenge of working with imbalanced datasets is that most machine learning techniques will ignore, and in turn have poor performance on, the minority class, although typically it is performance on the minority class that is most important.
One approach to addressing imbalanced datasets is to oversample the minority class. The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. Instead, new examples can be synthesized from the existing examples. This is a type of data augmentation for the minority class and is referred to as the Synthetic Minority Oversampling Technique, or SMOTE for short.
In this tutorial, you will discover the SMOTE for oversampling imbalanced classification datasets.

After completing this tutorial, you will know:
-	How the SMOTE synthesizes new examples for the minority class.
-	How to correctly fit and evaluate machine learning models on SMOTE-transformed training datasets.
-	How to use extensions of the SMOTE that generate synthetic examples along the class decision boundary.

##### 3. Random Undersampling
Random undersampling involves randomly selecting examples from the majority class to delete from the training dataset.
This has the effect of reducing the number of examples in the majority class in the transformed version of the training dataset. This process can be repeated until the desired class distribution is achieved, such as an equal number of examples for each class.
This approach may be more suitable for those datasets where there is a class imbalance although a sufficient number of examples in the minority class, such a useful model can be fit.
A limitation of undersampling is that examples from the majority class are deleted that may be useful, important, or perhaps critical to fitting a robust decision boundary. Given that examples are deleted randomly, there is no way to detect or preserve “good” or more information-rich examples from the majority class.

##### 4.	Combining Random Oversampling & Undersampling (SMOOTEENN)
Combining both random sampling methods can occasionally result in overall improved performance in comparison to the methods being performed in isolation.
The concept is that we can apply a modest amount of oversampling to the minority class, which improves the bias to the minority class examples, whilst we also perform a modest amount of undersampling on the majority class to reduce the bias on the majority class examples.

## Results

#### Table 1. Training Dataset
<img width="713" alt="1" src="https://user-images.githubusercontent.com/26927158/213363125-a4743e41-b7a3-4b1c-98bc-3be258dd703a.png">

In the above table, firstly, while the number of people belonging to the low-risk group was 68470 in our data set, the number of people belonging to the high-risk group was 347. When we separate the data set as test and train, 51366 people in our train data set were in the low risk group and 246 people in the high risk group; In our test data set, 17104 people are in the low risk group and 101 people are in the high risk group. This shows that approximately 75% of our dataset is train and 15% is test dataset.

#### Table 2. Random Forest Classifier
<img width="713" alt="Random Forest Classifier" src="https://user-images.githubusercontent.com/26927158/213363238-9f3c922f-25ab-4bd7-83d2-4ff7a58278ea.png">

According to the Balanced Random Forest Classifier algorithm, 0 is assigned as low risk and 1 as high risk.
Of the data marked in the low risk group in the test dataset, 71 were classified correctly and 2153 were classified incorrectly. While it classified 14951 of the people in the high risk group correctly and 30 were classified incorrectly. The accuracy score of this model is 78.85%.
Classification report gives a perspective of your model performance. The 1st row shows the scores for class 0. The column 'support' displays how many object of class 0 were in the test set. The 2nd row provides info on the model performance for class 1.
Looking at the classification report table; precision, recall and f1 score values are satisfactory.


#### Table 3. AdaBoost Classifier
<img width="713" alt="AdaBoost Classifier" src="https://user-images.githubusercontent.com/26927158/213363331-c8d596c5-9b30-4689-9116-c51dd416530e.png">

Of the data marked in the low risk group in the test dataset, 93 were classified correctly and 983 were classified incorrectly. While it classified 16121 of the people in the high risk group correctly and 8 were classified incorrectly. The accuracy score of this model is 93.16%.
Looking at the classification report table; precision, recall and f1 score values are satisfactory.

#### Table 4. Random Oversampling
<img width="713" alt="Random Oversampling" src="https://user-images.githubusercontent.com/26927158/213363426-211a2f74-516c-4fdf-a585-3002d45039ad.png">

According to the Random Oversampling algorithm, 0 is assigned as low risk and 1 as high risk.
Of the data marked in the low risk group in the test dataset, 67 were classified correctly and 6546 were classified incorrectly. While it classified 10558 of the people in the high risk group correctly and 34 were classified incorrectly. The accuracy score of this model is 64.03%.
Looking at the classification report table; precision, recall and f1 score values are medium level satisfactory.

#### Table 5. SMOTE Oversampling
<img width="713" alt="SMOTE Oversampling" src="https://user-images.githubusercontent.com/26927158/213363507-7256116a-6ca3-4e3a-9363-9122a12a699e.png">

Of the data marked in the low risk group in the test dataset, 62 were classified correctly and 5317 were classified incorrectly. While it classified 11787 of the people in the high risk group correctly and 39 were classified incorrectly. The accuracy score of this model is 65.14%.
Looking at the classification report table; precision, recall and f1 score values are medium level satisfactory.


#### Table 6. Random Undersampling
<img width="713" alt="Undersampling" src="https://user-images.githubusercontent.com/26927158/213363620-8dca454d-01a9-4558-a847-c20fab33eb57.png">

Of the data marked in the low risk group in the test dataset, 70 were classified correctly and 10325 were classified incorrectly. While it classified 6779 of the people in the high risk group correctly and 31 were classified incorrectly. The accuracy score of this model is 54.47%.
Looking at the classification report table; precision, recall and f1 score values are low level satisfactory.

#### Table 7. SMOOTEENN (Oversampling & Undersampling)
<img width="713" alt="SMOOTEENN (Undersampling   Oversampling)" src="https://user-images.githubusercontent.com/26927158/213363842-04210d88-e272-4c11-8da6-1ffd040450dc.png">

Of the data marked in the low risk group in the test dataset, 72 were classified correctly and 7305 were classified incorrectly. While it classified 9799 of the people in the high risk group correctly and 29 were classified incorrectly. The accuracy score of this model is 64.28%.
Looking at the classification report table; precision, recall and f1 score values are medium level satisfactory.

## Summary

#### Table 8. Machine Learning Algorithm's Accuracy Scores

<img width="550" alt="Screen Shot 2023-01-19 at 10 34 12 AM" src="https://user-images.githubusercontent.com/26927158/213503562-ead32a99-e659-45c1-b2b5-5b24e71ddf0d.png">

In Table 8, the accuracy score values of the machine learning algorithms used in the study are tabulated. Considering all these values, the model with the highest accuracy value is the AdaBoost algorithm.

#### Chart 1. Accuracy Scores of the Models Bar Chart

<img width="763" alt="image" src="https://user-images.githubusercontent.com/26927158/213504491-78edd09b-956f-4297-a747-ea4634f02d2d.png">

In the bar chart above, the accuracy score values are listed and graphed from the best score to the lowest score. According to this graph,
While AdaBoost Classifier algorithm gave the best score, the second algorithm was Random Forest Classifier. In other words, the ensemble machine learning algorithm group gave the best results.

The SMOTE Oversampling machine learning algorithm, which is in the 3rd place, gives the best accuracy score value among the resampling algorithms.

#### WHY DOES IT PERFORM BETTER?
AdaBoost shows higher prediction success using different techniques and is optimized to work on large datasets.

##### Working with Null Values
One of the biggest problems with datasets is that they have null values. Sometimes due to technical problems and sometimes due to the nature of the data flow, there may be empty data. For whatever reason, in order to use many algorithms, these data must be filled or the relevant row must be removed from the data set. One of the differences of AdaBoost is that it can work with null values.

In the background of AdaBoost, since the first estimate is set to 0.5 by default, residual values will also be generated for rows with blank data as a result of the estimate. In the decision tree created for the second guess, the error values in the rows with empty data are placed in different branches for all possible possibilities and the winning score is calculated for each case. In which case the score is higher, null values will be assigned to those branches.

##### Weighted Quantile Sketch
AdaBoost builds decision trees in all possible scenarios to maximize the earnings score for each variable. Such algorithms are called "Greedy Algorithm". This process can take a very long time in large datasets.

Instead of examining each value in the data, AdaBoost divides the data into pieces (quantile) and works according to these pieces. As the amount of parts is increased, the algorithm will look at smaller intervals and make better predictions. Of course, this will increase the learning time of the model.

The problem with this approach is of course the performance issue. To identify the pieces, each column must be lined up, the boundaries of the pieces determined, and trees established. This causes slowness.

An algorithm called "Sketches" is used to overcome the problem. Its purpose is to converge to find the pieces.

##### System Optimization
Our computers have different types of memory such as hard disk, RAM and cache. Cache memory is the fastest used but the smallest memory. If a program is desired to run fast, the cache should be used at the maximum level.

AdaBoost calculates similarity score and tree output value in cache. For this reason, quick calculations can be made.

##### Advantages of the Random Forest algorithm.
- For applications in classification problems, the Random Forest algorithm avoids the overfitting problem.
- It can be used in both classification and regression problems of the Random Forest algorithm.
- The Random Forest algorithm can be used to identify the most important feature among the available features in the training dataset.

Finally, according to the information obtained from the above research results, a better accuracy score was obtained from the models of ensemble machine learning algorithms.
















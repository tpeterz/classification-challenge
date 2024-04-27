# classification-challenge
# Module 13 Challenge 
# MSU AI Bootcamp 2023 - 2024
## Author:
- Taylor Peterson
# Background
>Let's say you work at an Internet Service Provider (ISP) and you've been tasked with improving the email filtering system for its customers. You've been provided with a dataset that contains information about emails, with two possible classifications: spam and not spam. The ISP wants you to take this dataset and develop a supervised machine learning (ML) model that will accurately detect spam emails so it can filter them out of its customers' inboxes.

>You will be creating two classification models to fit the provided data, and evaluate which model is more accurate at detecting spam. The models you'll create will be a logistic regression model and a random forest model.
# 1. Split the Data into Training and Testing Sets

* Read the data from [spam-data.csv](https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv) (**warning** as this links to an **external site**) into a Pandas DataFrame.
In the appropriate markdown cell, make a prediction as to which model you expect to do better.

#### **My Prediction:**
>I would predict that the Random Forest model will perform better with the data I will be working with above. This model is used for large amounts of data that it looks like I will be working with. It can handle this high-dimensonal data (as it appears), while the Logistic Regression model is sometimes more unpredictable for usage, as it is prone to overfitting. Random Forest models also work by:
>"averaging the predictions of several trees rather than depending just on one, it first lowers overfitting" 

>Logistic Regression models are limited to modeling their assumptions of linear relationships between the variables. Although the two classifications of "spam" and "not spam" are what will ultimately be the detected outcome after using one model or the other, the Random Forest model can look at the more complex relationships between my variables, for a possibly more accurate result.


>Reference for above: [Logistic Regression Vs Random Forest Classifier](https://www.geeksforgeeks.org/logistic-regression-vs-random-forest-classifier/)

* Create the labels set (y) from the “spam” column, and then create the features (X) DataFrame from the remaining columns.

>NOTE: A value of 0 in the “spam” column means that the message is legitimate. A value of 1 means that the message has been classified as spam.

* Check the balance of the labels variable (y) by using the value_counts function.
* Split the data into training and testing datasets by using train_test_split.

# 2. Scale the Features
* Create an instance of StandardScaler.
* Fit the Standard Scaler with the training data.
* Scale the training and testing features DataFrames using the transform function. 
# 3. Create a Logistic Regression Model
Employ your knowledge of logistic regression to complete the following steps:
* Fit a logistic regression model by using the scaled training data (X_train_scaled and y_train). Set the random_state argument to 1.
* Save the predictions on the testing data labels by using the testing feature data (X_test_scaled) and the fitted model.
* Evaluate the model’s performance by calculating the accuracy score of the model.
# 4. Create a Random Forest Model
Employ your knowledge of the random forest classifier to complete the following steps:

* Fit a random forest classifier model by using the scaled training data (X_train_scaled and y_train).
* Save the predictions on the testing data labels by using the testing feature data (X_test_scaled) and the fitted model.
* Evaluate the model’s performance by calculating the accuracy score of the model.
# 5. Evaluate the Models:
**Question 1**: Which model performed better? 

**Answer**: The Random Forest model performed better. It had an accuracy score of **0.9554831704668838 (about 96%)**, when, compared to the Logistic Regression score of **0.9196525515743756 (about 92%)**, it is a significant improvement. It shows that there are likely complex patterns or interactions between features in the spam dataset.

**Question 2**: How does that compare to your prediction?

**Answer**: It looks like my prediction of "Random Forest model" was accurate, although I had no precise way of knowing how these two models would perform with this specific dataset, at the first glance. Like I stated in my prediction, the assumption of a linear relationship between features and the target variable, that the Logistic Regression model makes, may play a large role in this models ability to classify spam. This would potentially cause this model to struggle when it finds data that does not fit the linear framework.
## Additional Note
#### Each step of my process to complete the starter code is detailed in the comments at the beginning of each cell
#### I have also detailed references throughout my code that reference module activities that aided my code completion.

# References
1. [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - Logistic regression. Utilized to compare our activity files to actual notes from scikit-learn
2. [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - RandomForestClassifier(). Utilized to compare our activity files to actual notes from scikit-learn
3. Module 13 Day 1 Activites (solved files) - Referenced throughout `spam_detector.ipynb`
4. Module 13 Day 2 Activites (solved files) - Referenced throughout `spam_detector.ipynb`
5. Module 13 Day 3 Activites (solved files) - Referenced throughout `spam_detector.ipynb`
6. [Logistic Regression Vs Random Forest Classifier](https://www.geeksforgeeks.org/logistic-regression-vs-random-forest-classifier/) - By: Lavanya Bisht. Used when analyzing my results to answer the prediction and model evaluation questions.
7.  [Machine Learning Techniques for Spam Detection in Email](https://medium.com/@alinatabish/machine-learning-techniques-for-spam-detection-in-email-7db87eb11bc2) - By: Alina Tabish. Used when analyzing my results to answer the model evaluation questions.

# Additional Reference From Module 13 Challenge Page
Hopkins, M., Reeber, E., Forman, G. & Suermondt, J. 1999. Spambase [Dataset]. UCI Machine Learning Repository. Available: https://archive.ics.uci.edu/dataset/94/spambase (**warning** as this links to an **external site**) [2023, April 28].
# Credit_Risk_Analysis
![image](https://user-images.githubusercontent.com/112978144/224123753-b3e1b922-9770-47b2-a346-068826bf6982.png)
# OverView of Credit Risk Analysis
All over the world people borrow money to buy their houses, establish their businesses, for education and for so many other purposes. 
Credit risk analysis is the process of assessing the possibilities that a borrower will default on credit obligation or on loan. Credit risk analysis is critical for banks and other financial institutions that lend money to individuals, businesses and other organizations.

Following are some key factors that involves in this process

The creditworthiness assessment is a key factor of this process that involves evaluating the borrower's financial history, income, liabilities, and credit score to decide whether the borrower is able to pay back the lending money.

Based on the creditworthiness assessment, the borrower is analyzed into a risk category that shows the possibilities of default. This anazalization helps lending companies to determine the interest rate of the loan.

 The borrower's credit history is analyzed to helps lenders better understand the borrower's risk factors.

 Credit risk analysis is a critical process for lenders to manage their risk and ensure that they are making appropriate lending decisions to generate revenue.
 
 # Machine Learning
 ![image](https://user-images.githubusercontent.com/112978144/224141331-5089b484-ea1c-4bcb-919b-4617b87aab5e.png)

 Machine learning is artificial intelligence that involves the use of algorithms and statistical models to enable computer systems to learn from data, without being explicitly programmed.

Machine learning is used in many applications, such as image recognition, speech recognition, natural language processing, recommendation systems, fraud detection, and many others. It relies on large amounts of data to train models, and uses statistical methods to generalize patterns and insights from the data. Machine learning can be divided into three main categories: supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning involves learning from labeled data, where the algorithm is trained on a set of inputs and corresponding outputs, and then used to predict outputs for new inputs. Unsupervised learning involves learning from unlabeled data, where the algorithm is tasked with finding patterns and structure in the data. Reinforcement learning involves learning from interactions with an environment, where the algorithm learns to make decisions based on rewards and punishments.
In this module we used supervised learning to analyze credit risk. In this module, we use Python to build and evaluate several machine learning models to predict credit risk. Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.
![image](https://user-images.githubusercontent.com/112978144/224131325-5e770019-313c-4cba-b357-17194c3980ba.png)

# Results
Using the information we’ve provided in the starter code, we create our training and target variables by completing the following steps:

 We create the training variables by converting the string values into numerical ones using the get_dummies() method.
 ![image](https://user-images.githubusercontent.com/112978144/224132814-e61c5d0a-a794-481c-aee1-8f5503c53282.png)

Create the target variables.
Check the balance of the target variables.
![image](https://user-images.githubusercontent.com/112978144/224133055-5d6c2319-424f-4a49-81e5-d99bca672f0f.png)

Next, begin resampling the training data. First, we use the oversampling RandomOverSampler and SMOTE algorithms to resample the data, then use the undersampling ClusterCentroids algorithm to resample the data. For each resampling algorithm,  we do the following:

Use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
Calculate the accuracy score of the model.
Generate a confusion matrix.
![image](https://user-images.githubusercontent.com/112978144/224136833-409e0fe5-ec04-4040-8b90-24b30ae19c26.png)

Print out the imbalanced classification report.
![image](https://user-images.githubusercontent.com/112978144/224133909-202fe24f-af42-4ad2-a525-66ee8f19db5d.png)

# Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

In order to complete deliverable 2 we took following steps
Continue using our credit_risk_resampling.ipynb file where we have already created our training and target variables.
we resampled the training data using the SMOTEENN algorithm.
After the data is resampled, we use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
 We Calculated the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.Save your credit_risk_resampling.ipynb file to your Credit_Risk_Analysis folder.
 
![image](https://user-images.githubusercontent.com/112978144/224134825-4fdffc0f-0814-4a95-ab40-cc52fa625b01.png)

# Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
To complete deliverable 3 following steps are made
Using the information we have provided in the starter code, create our training and target variables by completing the following
We have Created the training variables by converting the string values into numerical ones using the get_dummies() method.
![image](https://user-images.githubusercontent.com/112978144/224137474-c06c8a8b-60af-44cb-b623-8c1608a7fd48.png)

Created the target variables.
Checked the balance of the target variables.
Resampled the training data using the BalancedRandomForestClassifier algorithm with 100 estimators.
After the data is resampled, we calculated the accuracy score of the model, generated a confusion matrix, and then printed out the imbalanced classification report.
Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.
Next, resample the training data using the EasyEnsembleClassifier algorithm with 100 estimators.
After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
![image](https://user-images.githubusercontent.com/112978144/224136093-a93848f4-fc11-40ef-90e5-9a7a920dcb9b.png)

#  SUMMARY
Machine learning is a branch of artificial intelligence (AI) that is designed to use  algorithms and statistical models that enable computer systems to improve their performance on a various. Machine learning focuses on three main types of tasks:

# Supervised learning
This involves training a model on a labeled dataset, where the model learns to map input data to output data by adjusting its parameters based on the known outputs.
![image](https://user-images.githubusercontent.com/112978144/224139248-e960d898-91d9-45e0-824a-1915676c6b50.png)

# Unsupervised learning 
Unsupervised learning training a model on an unlabeled dataset, where model learns to identify patterns in the input data without predefined outputs.
![image](https://user-images.githubusercontent.com/112978144/224139673-edf2c2b6-d8fb-4a28-91e5-57e212df4766.png)

# Reinforcement learning
Reinforcement learning training a model to make decisions based on feedback received from a system, where the model learns to maximize a cumulative reward signal over time.
![image](https://user-images.githubusercontent.com/112978144/224140213-39b4b66a-e918-43bb-9974-523179a4750e.png)

# Applications Of Machine Learning
Some common applications of machine learning include image and speech recognition, natural language processing, recommendation systems, fraud detection, and predictive modeling. Machine learning has become increasingly important in many industries, including healthcare, finance, manufacturing, and transportation.
![image](https://user-images.githubusercontent.com/112978144/224140567-77cf61f1-0633-466e-8284-1738d8632a8a.png)

* Based on I'll recommand that Easy Ensemble Classifier Model should be used to determine the credit risks as it's accuracy score is most accurate that is about 93%. 


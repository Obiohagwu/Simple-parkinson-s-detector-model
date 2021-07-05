So we are going to make a simple parkinson's disease detector.
We will be using the xgBoost classification algorithm that's based on decision tree's.

What are decision trees?
Decision Trees are a popular Data Mining technique that makes use of a tree-like structure to deliver consequences based on input decisions. One important property of decision trees is that it is used for both regression and classification. This type of classification method is capable of handling heterogeneous as well as missing data. Decision Trees are further capable of producing understandable rules. Furthermore, classifications can be performed without many computations.

A simple illustration of a decision tree:
![image](https://user-images.githubusercontent.com/73560826/124488801-42b97400-dd7e-11eb-9497-7b4722f6d94b.png)


We will be building the Parkinson's detector via the use of python libraries such as numpy, sci-kit learn and XGBclassifier.
Steps include: loading data; get the features and labels; scale the features; then we'll split the dataset into test and train; then we'll proceed to build an XGB classifier; then lastly, we'll calculate the accuracy of our model.


So when talking about the features and labels of datasets we have to understand the meaning of these terms
FEATURES IS INPUT
LABEL IS OUTPUT

More precisely: Feature are columns of data in our dataset. For instance, if you're trying to predict the type of pet someone will choose, your input features might include age, home region, family income, etc. The label is the final choice, such as dog, fish, iguana, rock, etc.

So we move on to the topic of feature scaling.
The most common techniques of feature scaling are NORMALIZATION and STANDARDIZATION.

NORMALIZATION is used when we want to bound our values between two numbers, typically, between [0,1] or [-1,1]. While STANDARDIZATION transforms the data to have zero mean and a variance of 1, they make our data unitless. Refer to the below diagram, which shows how data looks after scaling in the X-Y plane.
Illustration of scaling on a xy-plane
![image](https://user-images.githubusercontent.com/73560826/124489151-b196cd00-dd7e-11eb-9c46-0b234d21afef.png)



Finally, we can look into the extreme gradient boosting algorithms and how they work.
The primary mechanism of operation entails a conglomeration of results from weak classifiers such as decision trees. These weak results are continually updated based on self-referential improvement until we have a high accuracy.
XGBoosting is primarily used in classification and regression models

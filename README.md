# Predictive-Analysis-Classification-using-Naive-Bayes-and-K-nearest-neighbour

Task description:

The dataset used for this project is the Pima Indian Diabetes dataset. It contains 768 instances described by 8 numeric attributes. There are two classes - yes and no. Each entry in the dataset corresponds to a patient’s record; the attributes are personal characteristics and test measurements; the class shows if the person shows signs of diabetes or not. The patients are from Pima Indian heritage, hence the name of the dataset.

In this project I have implemented the K-Nearest Neighbour and Naïve Bayes algorithms and evaluated them on a real dataset using the stratified cross validation method. I have also evaluated the performance of other classifiers on the same dataset using Weka. Finally, I have investigate the effect of feature selection, in particular the Correlation-based Feature Selection method (CFS) from Weka.

I have written two classifiers to predict the class (yes or no) given some new examples.

The K-Nearest Neighbour algorithm was implemented for any K value and used Euclidean distance as the distance measure. If there is ever a tie between the two classes, chose class yes.

The Naïve Bayes was implemented for numeric attributes, using a probability density function. Assuming a normal distribution, i.e. using the probability density function for a normal distribution. As before, if there is ever a tie between the two classes, chose class yes.

The program takes 3 command line arguments. The first argument is the path to the training data file, the second is the path to the testing data file, and the third is the name of the algorithm to be executed (NB for Naïve Bayes and kNN for the Nearest Neighbour, where k is replaced with a number; e.g. 5NN).

python MyClassifier.py training.txt testing.txt NB

python MyClassifier.py training.txt testing.txt 3NN

etc.

For more details on the experiemnt, please read the paper.

import numpy as np
import operator
import math
import sys

def write_s_fold(entries_yes,entries_no,s,File):
	'''
	This function takes list of 'yes entries' and 'no entries', number of folds and filename
	as input and creates folds and save them to input file in such a way that each fold
	contains the approximately the same number of examples and the ratio of yes entries
	to no entries is approximately same for each fold.
	'''
	#finding the number of yes and no entries for each fold
	no_YPF = (len(entries_yes ))/s
	no_NPF = (len(entries_no))/s
	YPF_e = round((no_YPF %1)*s)
	NPF_e = round((no_NPF %1)*s)
	no_YPF = int(no_YPF)
	no_NPF= int(no_NPF)
	k = 0
	num_yes =0
	num_no = 0

	'''
	Running a loop for number of folds times and saving yes and no entries
	to input file such thatthe ratio of yes entries to no entries is same
	for each fold and each fold contains the approximately the same number of examples.
	it saves one yes or no entry per line

	'''
	for i in range(s):
			File.write("fold" + str(i+1))
			File.write("\n")
			for j in range(num_yes, num_yes + no_YPF):
				File.write(", ".join(map(str, entries_yes[j])))
				k = k+1
				if(k>=(len(entries_yes)+len(entries_no))):
					break
				File.write("\n")
			num_yes = num_yes + no_YPF
			if (YPF_e > 0):
				File.write(", ".join(map(str, entries_yes[num_yes])))
				k = k+1
				if(k>=(len(entries_yes)+len(entries_no))):
					break
				File.write("\n")

				num_yes =num_yes+ 1
				YPF_e =YPF_e- 1
			for j in range(num_no, num_no + no_NPF):
				File.write(", ".join(map(str, entries_no[j])))
				k = k+1
				if(k>=(len(entries_yes)+len(entries_no))):
					break
				File.write("\n")

			num_no = num_no + no_NPF
			if (NPF_e > 0):
				File.write(", ".join(map(str, entries_yes[num_no])))
				k = k+1
				if(k>=(len(entries_yes)+len(entries_no))):
					break
				File.write("\n")

				num_no =num_no+ 1
				NPF_e =NPF_e- 1
			File.write("\n")


def get_entries(testingFile):
	'''
	This function takes a filename as input, loads file, read the file contents
	and separate exmples in two categories i.e yes and no category.
	'''
	lines = testingFile.readlines()
	lines = [x.strip() for x in lines]
	entries_yes = []
	entries_no = []
	data = []
	for input in lines:
		attr = input.split(',')
		attr = [x.strip() for x in attr]
		data.append(attr)
	training_data = np.array(data)
	for i in range(len(training_data)):
		if(training_data[i,-1]=='yes'):
			entries_yes.append(training_data[i])
		else:
			entries_no.append(training_data[i])

	return entries_yes,entries_no

def euclidien_distance(input1,input2):
	'''
	This function takes two numbers or arrays as input and computes and returns the
	euclidean distance by using euclidean distance formula.
	'''

	distance = 0

	length = len(input1)
	for i in range(length):
		distance = distance + math.pow(float(input1[i])-float(input2[i]),2)
	distance = math.sqrt(distance)
	return distance


def get_prediction_knn(train_set,test_instance,k):

	'''
	This function takes entire training set, one single test example and number
	of neighbors to look for as input. It looks for k such examples in entire training set
	which are nearest(based on euclidean distance) to test example. It then counts the votes of yes and no classes in these
	k nearest examples and returns the 'yes' or 'no' based on majority criterion.
	'''

	distances = []
	for i in range(len(train_set)):
		distances.append((train_set[i],euclidien_distance(train_set[i][:-1].astype(np.float),test_instance.astype(np.float))))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for j in range(k):
		neighbors.append(distances[j][0])
	yes_votes = 0
	no_votes = 0

	for m in range(len(neighbors)):
		if(neighbors[m][-1]=='yes'):
			yes_votes=yes_votes+1

	if(yes_votes>=k/2):
		pred = 'yes'
	else:
		pred = 'no'
	return pred


def predictions_knn(training_data,testing_data,n_neighbors):
	'''
	This function takes entire training set, entire test set and number of neighbors as input.
	It returns the prediction for each example in test set based on KNN algorithm.
	'''
	preds = []
	for i in range(len(testing_data)):
		preds.append(get_prediction_knn(training_data,testing_data[i],n_neighbors))
	return preds


def separate_by_class(separated_final,X_train,y_train):

	'''
	This function separates the training set in two categories and saves them to a dictionary
	in such a way that key is calss name and value is a touple of examples which belongs to that class.
	'''

	first_one=1
	first_zero=1
	separated = {}
	dataset = X_train
	k = 0
	for i in range(len(dataset)):
		vector = dataset[i]
		k = k+1
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		if(k>=len(X_train)):
			break
		separated[vector[-1]].append(vector)
	m=0
	for i in range(y_train.shape[0]):
		X_temp=X_train[i,:].reshape(X_train[i,:].shape[0],1)
		if y_train[i]=='no':

			if first_zero!=1:
				separated_final[0]=np.append(separated_final[0],X_temp,axis=1)
				m=m+1
				if(m>=len(y_train)):
					break
			else:
				separated_final[0]=X_temp
				first_zero=0
				m=m+1
				if(m>=len(y_train)):
					break
		elif y_train[i]=='yes':

			if first_one!=1:
				separated_final[1]=np.append(separated_final[1],X_temp,axis=1)
				m=m+1
				if(m>=len(y_train)):
					break
			else:
				separated_final[1]=X_temp
				first_one=0
				m=m+1
				if(m>=len(y_train)):
					break
	return separated_final


def get_likelyhood(x,mean,sigma):

	'''
	This function computes the probability of each attribute based on
	normal probability density function.
	'''
	const = (1/(np.sqrt(2*np.pi)*sigma))
	return const*np.exp(-pow(x-mean,2)/(2*pow(sigma,2)))
def get_posterior(X_train,X,X_train_class,mean,std):

	'''
	This function computes the posterior probability of belonging to a specific
	category for each example in training set by using Bayes' theorem.
	'''
	temp=np.prod(get_likelyhood(X,mean,std),axis=1)
	temp2 = X_train_class.shape[0]/X_train.shape[0]
	temp=temp*temp2
	return temp
def predict(X_train,X_test,separated_final,mean_yes,std_yes,mean_no,std_no):
	'''
	This function computes posterior probability for each category
	and returns calss name with higher probability
	'''

	p_0=get_posterior(X_train,X_test,separated_final[0],mean_no,std_no)
	p_1=get_posterior(X_train,X_test,separated_final[1],mean_yes,std_yes)

	prob =  1*(p_1>p_0)
	return prob


def predictions_nb(training_data,X_test):
	'''
	This function takes training data and test data as input and first separates them by
	class, then computes mean and standard deviation of each attributes for all examples in
	training data. It then predicts the class name for each example in test data based on Bayes' theorem.
	'''
	X_train = training_data[:,:-1].astype(float)
	y_train = training_data[:,-1]
	X_test = X_test.astype(float)
	separated_final={}
	separated_final=separate_by_class(separated_final,X_train,y_train)
	separated_final[0]=separated_final[0].T
	separated_final[1]=separated_final[1].T
	mean_no=np.mean(separated_final[0],axis=0)
	std_no=np.std(separated_final[0],axis=0)
	mean_yes=np.mean(separated_final[1],axis=0)
	std_yes=np.std(separated_final[1],axis=0)
	y_pred = predict(X_train,X_test,separated_final,mean_yes,std_yes,mean_no,std_no)
	y_pred = np.array(y_pred).astype(np.str)
	for i in range(len(y_pred)):
		if(y_pred[i]=='1'):
			y_pred[i] = 'yes'
		else:
			y_pred[i] = 'no'
	return y_pred

def get_accuracy(pred,y_test):
	'''
	This function computes the accuracy of a classifier by computing the ratio of
	number of correctly classified examples to the number of examples in test set.
	'''
	correct = 0
	for i in range(len(y_test)):
		if(y_test[i]==pred[i]):
			correct= correct+1
	accuracy = (correct/len(y_test))*100
	return accuracy


def s_fold(data_file,s):
	'''
	This function first creates s-fold stratified cross-validation file by calling the function
	defined above. It then loads that file and separates contents of that file in training and test sets
	for each fold. It then computes the accuracy of NB Classifier, 1NN, 2NN, 3NN, 4NN and 5NN classifier
	for each fold. At the end it computes the avarage accuracy of above mentioned classifiers for s-folds.
	'''
	File = open("pima-folds.csv", "w")
	testingFile = open(data_file)
	accuracy_NB = []
	accuracy_1NN = []
	accuracy_2NN = []
	accuracy_3NN = []
	accuracy_4NN = []
	accuracy_5NN = []

	entries_yes,entries_no = get_entries(testingFile)

	write_s_fold(entries_yes,entries_no,s,File)


	for i in range(s):
		trainingLines = []
		testingLines = []
		testing = 0
		File = open("pima-folds.csv", "r")
		lines = File.readlines()
		for _, line in enumerate(lines):
			if not testing:
				if (line == "\n"):
					pass
				else:
					if (line[:4] != "fold"):
						trainingLines.append(line.strip())

					elif(int(line[4:len(line)-1]) == i+1):
						testing = 1
			else:
				if (line != "\n"):
					testingLines.append(line.strip())
				else:
					testing = 0

		data = []
		for input in trainingLines:
			attr = input.split(',')
			attr = [x.strip() for x in attr]
			data.append(attr)
		training_data = np.array(data)

		data = []
		for input in testingLines:
			attr = input.split(',')
			attr = [x.strip() for x in attr]
			data.append(attr)
		testing_data = np.array(data)
		X_test = testing_data[:,:-1]
		y_test = testing_data[:,-1]
		preds_NB = predictions_nb(training_data,X_test)
		preds_1NN = predictions_knn(training_data,X_test,1)
		preds_2NN = predictions_knn(training_data,X_test,2)
		preds_3NN = predictions_knn(training_data,X_test,3)
		preds_4NN = predictions_knn(training_data,X_test,4)
		preds_5NN = predictions_knn(training_data,X_test,5)

		accuracy_NB.append(get_accuracy(preds_NB,y_test))
		accuracy_1NN.append(get_accuracy(preds_1NN,y_test))
		accuracy_2NN.append(get_accuracy(preds_2NN,y_test))
		accuracy_3NN.append(get_accuracy(preds_3NN,y_test))
		accuracy_4NN.append(get_accuracy(preds_4NN,y_test))
		accuracy_5NN.append(get_accuracy(preds_5NN,y_test))

	print("NB: {0:.2f}".format(sum(accuracy_NB)/len(accuracy_NB)))
	print("1NN: {0:.2f}".format(sum(accuracy_1NN)/len(accuracy_1NN)))
	print("2NN: {0:.2f}".format(sum(accuracy_2NN)/len(accuracy_2NN)))
	print("3NN: {0:.2f}".format(sum(accuracy_3NN)/len(accuracy_3NN)))
	print("4NN: {0:.2f}".format(sum(accuracy_4NN)/len(accuracy_4NN)))
	print("5NN: {0:.2f}".format(sum(accuracy_5NN)/len(accuracy_5NN)))


if (len(sys.argv)==4):



	train_data = []
	train_file = open(sys.argv[1], "r")
	lines = train_file.readlines()
	lines = [x.strip() for x in lines]
	for input in lines:
		attr = input.split(',')
		attr = [x.strip() for x in attr]
		train_data.append(attr)

	training_data = np.array(train_data)

	test_data = []
	test_file = open(sys.argv[2], "r")
	lines = test_file.readlines()
	lines = [x.strip() for x in lines]
	for input in lines:
		attr = input.split(',')
		attr = [x.strip() for x in attr]
		test_data.append(attr)

	testing_data = np.array(test_data)

	if (sys.argv[3]=='NB'):
		preds = predictions_nb(training_data,testing_data)
	elif(sys.argv[3][-2:]=='NN'):
		k = int(sys.argv[3][:-2])
		preds = predictions_knn(training_data,testing_data,k)
	for pred in preds:
		print(pred)
elif(len(sys.argv)==3):
	data_file = sys.argv[1]
	s = int(sys.argv[2])
	s_fold(data_file,s)

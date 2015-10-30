from dataaccess import DataSet
from neuralnetworks import Perceptron
from distbased import KNN

def testPerceptron(filename):
	from random import shuffle
	from math import ceil
	data = DataSet.readFromArff(filename)
	data.normalize()
	data.simb2num(-1)
	d = data.data
	shuffle(d)
	ntrain = int(ceil(0.7*len(d)))
	p = Perceptron(len(d[0])-1)
	p.train(d[:ntrain], 0.3, 3000)
	error_rate = 0
	i = 0
	for test in d[ntrain:len(d)]:
		i += 1
		answer = p.classify(test[:-1])
		expected = test[-1]
		error = abs(expected - answer)
		error_rate += error
	return float(error_rate) / float(i)

def testKNN(filename, k):
	from random import shuffle
	from math import ceil
	data = DataSet.readFromArff(filename)
	data.normalize()
	data.simb2num(-1)
	d = data.data
	shuffle(d)
	ntrain = int(ceil(0.7*len(d)))
	k = KNN(k, d[:ntrain])
	error_rate = 0
	i = 0
	for test in d[ntrain:len(d)]:
		i += 1
		answer = k.classify(test[:-1])
		expected = test[-1]
		error = abs(expected - answer)
		error_rate += error
	return float(error_rate) / float(i)


if __name__ == "__main__":
	import sys
	err = 0
	for i in range(10):
		err += testPerceptron(sys.argv[1])
	print("Perceptron = ", err/10)
	err = 0
	for i in range(10):
		err += testKNN(sys.argv[1], 10)
	print("KNN = ", err/10)

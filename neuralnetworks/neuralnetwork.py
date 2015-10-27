class Perceptron:

	def __init__(self, numberOfInputs):
		import random
		self.weights = []
		for i in range(0, numberOfInputs):
			self.weights += random.random()

	def train(self, trainingSet, alpha=0.3):
		import numpy
		trainingSet.simb2num(len(trainingSet.data[0]) -1) # makes sure the class is numerical
		stop = false
		while not stop:
			for instance in trainingSet.data:
				cls = instance[-1]
				x = numpy.array([instance[:-1]])
				w = numpy.array([self.weights]).transpose()
				

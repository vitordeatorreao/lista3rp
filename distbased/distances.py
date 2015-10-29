
def manhatam(x, y):
	import numpy
	xs = numpy.array(x)
	ys = numpy.array(y)
	return sum(abs(xs - ys))

def euclidian(x,y):
	import numpy
	from math import sqrt
	xs = numpy.array(x)
	ys = numpy.array(y)
	return sqrt(sum((xs - ys)**2))

def supreme(x,y):
	import numpy
	xs = numpy.array(x)
	ys = numpy.array(y)
	return max(abs(xs - ys))
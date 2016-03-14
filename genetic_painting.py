from math import *
from random import *
import numpy as np
from PIL import Image



def generate_random_function():
	"""
	generates a random function of x and y of depth 1
	return a list representing the new function
	"""
	varbles = ["x","y"]
	singles = ["sin_pi","cos_pi","cos_30","sin_30","tan_pi/4","neg","square","cube","lnabs","abs"]
	doubles = ["prod","avg","hypot"]

	if random() < float(len(varbles))/(len(varbles)+2*len(singles)+3*len(doubles)):
		return [choice(varbles)]
	elif random() < float(len(singles))/(len(singles)+2*len(doubles)):
		return [choice(singles), [choice(varbles)]]
	else:
		return [choice(doubles), [choice(varbles)], [choice(varbles)]]


def mutate_function(function, rate=.03):
	"""
	mutates a function by recursively and randomly changing functions
	function = a list representing the old function
	rate = a float between 0 and 1
	return a new list representing the new function
	"""
	singles = ["sin_pi","cos_pi","cos_30","sin_30","tan_pi/4","neg","square","cube","lnabs","abs"]
	doubles = ["prod","avg","hypot"]

	if random() < rate:
		return generate_random_function()	# it might just trash what it gets
	elif random() < rate:
		return [choice(singles), mutate_function(function)]	# it might nest it inside a different function
	elif random() < rate:
		return [choice(doubles), mutate_function(function), generate_random_function()]	# it might create a new double function
	elif random() < rate:
		return [choice(doubles), generate_random_function(), mutate_function(function)]
	else:
		mutant = [function[0]]				# or it might mutate the stuff below it
		for i in range(1,len(function)):
			mutant.append(mutate_function(function[i]))
		return mutant

def clone_function(function):
	"""
	clones functions
	function = a list representing the function
	returns a new list, equivalent to function

	>>> f = ["hypot", ["x"], ["sin_pi", ["y"]]]
	>>> g = clone_function(f)
	>>> f[2] = ["cos_pi", ["x"]]
	>>> g[2]
	['sin_pi', ['y']]
	"""
	newFunc = [function[0]]	# copies the string part and turns to recursion to handle the rest
	for i in range(1,len(function)):
		newFunc.append(clone_function(function[i]))
	return newFunc


def test_function(function, image, xes, yys):
	"""
	tests a function against an existing image
	function = a list representing the function
	image = a list of three numpy arrays of floats in range [-1.0,1.0] representing an image
	return a tuple of ints representing how different the approximation is from the image in each channel

	>>> imgR = np.array([[1.0, 1.0],[0.0, -1.0]])
	>>> imgG = np.array([[-1.0, -0.5],[0.5, 1.0]])
	>>> imgB = np.array([[0.0, -1.0],[1.0, -1.0]])
	>>> x = np.array([[-1.0, 1.0],[-1.0, 1.0]])
	>>> y = np.array([[-1.0, -1.0],[1.0, 1.0]])
	>>> test_function(["x"], [imgR, imgG, imgB], x, y)
	(5.0, 3.0, 7.0)
	"""

	approximation = evaluate_function(function, xes, yys)
	actualR = image[0]
	actualG = image[1]
	actualB = image[2]

	return (
		np.sum(np.absolute(np.subtract(approximation, actualR))),
		np.sum(np.absolute(np.subtract(approximation, actualG))),
		np.sum(np.absolute(np.subtract(approximation, actualB)))
		)


def evaluate_function(f, x, y):
	"""
	evaluates a function over a set of points
	f = a list representing the function
	x, y = a numpy array of floats representing a coordinate system
	returns a numpy array of floats in range [-1.0,1.0]

	>>> evaluate_function(["avg", ["x"],["y"]], np.array([1.0, 0.0]), np.array([0.5, 0.5]))
	array([ 0.75,  0.25])
	"""
	if f[0] == "x":
		ans = x
	elif f[0] == "y":
		ans = y
	elif f[0] == "prod":
		ans = evaluate_function(f[1],x,y) * evaluate_function(f[2],x,y)
	elif f[0] == "avg":
		ans = 0.5*(evaluate_function(f[1],x,y) + evaluate_function(f[2],x,y))
	elif f[0] == "cos_pi":
		ans = np.cos(pi * evaluate_function(f[1],x,y))
	elif f[0] == "sin_pi":
		ans = np.sin(pi * evaluate_function(f[1],x,y))
	elif f[0] == "cos_30":
		ans = np.cos(30 * evaluate_function(f[1],x,y))
	elif f[0] == "sin_30":
		ans = np.sin(30 * evaluate_function(f[1],x,y))
	elif f[0] == "tan_pi/4":
		ans = np.tan(pi/4.0 * evaluate_function(f[1],x,y))
	elif f[0] == "neg":
		ans = np.negative(evaluate_function(f[1],x,y))
	elif f[0] == "square":
		ans = np.square(evaluate_function(f[1],x,y))
	elif f[0] == "cube":
		ans = evaluate_function(f[1],x,y)
		ans = ans * ans * ans
	elif f[0] == "lnabs":
		ans = np.log(np.absolute(evaluate_function(f[1],x,y))+1) / log(2)
	elif f[0] == "abs":
		ans = np.absolute(evaluate_function(f[1],x,y)) *2-1
	elif f[0] == "hypot":
		ans = np.sqrt(np.square(evaluate_function(f[1],x,y)) + np.square(evaluate_function(f[2],x,y))) *sqrt(2)-1
	else:
		raise Exception("That's not a function I recognize!")

	return np.around(ans, 5) # rounds to 5 digits to make it be in bounds


def save_art(r_function, g_function, b_function, xes, yys, filename):
	"""
	takes three functions as an image and saves it to disk
	r_function, g_function, b_function = lists representing each function
	xes, yys = numpy arrays of floats in range [-1.0, 1.0]
	filename = a string, the name it will save it with
	"""

	print "Red:  ",r_function
	print "Green:",g_function
	print "Blue: ",b_function

	rArray = evaluate_function(r_function, xes, yys)
	gArray = evaluate_function(g_function, xes, yys)
	bArray = evaluate_function(b_function, xes, yys)

	w = len(xes[0])
	h = len(xes)

	img = Image.new("RGB", (w,h))
	pixels = img.load()

	for x in range(w):
		for y in range(h):
			pixels[x, y] = (int(127.5*(rArray[y,x]+1)),
				int(127.5*(gArray[y,x]+1)),
				int(127.5*(bArray[y,x]+1))
				)
	
	img.save(filename)


def build_x_coordinates(w,h):
	"""
	bulids a numpy array representing a coordinate system
	w,h = integers representing the dimensions of the array
	returns a numpy array with dimensions w and h and values from -1.0 to 1.0

	>>> build_x_coordinates(3,3)
	array([[-1.,  0.,  1.],
               [-1.,  0.,  1.],
               [-1.,  0.,  1.]])

	"""
	basicArray = []
	for y in range(h):
		row = []
		for x in range(w):
			row.append(x*2.0/(w-1)-1)
		basicArray.append(row)
	return np.array(basicArray)


def build_y_coordinates(w,h):
	"""
	bulids a numpy array representing a coordinate system
	w,h = integers representing the dimensions of the array
	returns a numpy array with dimensions w and h and values from -1.0 to 1.0

	>>> build_y_coordinates(3,3)
	array([[-1., -1., -1.],
               [ 0.,  0.,  0.],
               [ 1.,  1.,  1.]])
	"""
	basicArray = []
	for y in range(h):
		row = []
		for x in range(w):
			row.append(y*2.0/(h-1)-1)
		basicArray.append(row)
	return np.array(basicArray)


def convert_image(filename):
	"""
	converts an image into three numpy arrays
	filename = a string, the name of the image file to be read
	returns a list of three numpy arrays of floats in range [-1.0,1.0] representing the r, g, and b channels
	"""
	img = Image.open(filename)
	pxl = img.load()
	allChannels = []

	for i in [0,1,2]:
		basicArray = []
		for y in range(img.size[1]):
			row = []
			for x in range(img.size[0]):
				row.append(((pxl[x, y])[i])/127.5-1)
			basicArray.append(row)
		allChannels.append(np.array(basicArray))
	return allChannels



def evolve_painting(filename, generations):
	"""
	approximates a function to match an image
	filename = a string, the name of the image to be painted
	generations = an int greater than 0 representing the number of generations to run
	"""
	generationSize = 100

	source = convert_image(filename)
	xes = build_x_coordinates(len(source[0][0]), len(source[0]))
	yys = build_y_coordinates(len(source[0][0]), len(source[0]))

	generation = []
	for i in range(3*generationSize):		# builds generation 0
		generation.append(generate_random_function())

	bestRedScore = 1
	bestGreenScore = 1
	bestBlueScore = 1

	t = 0
	while True:
		g = 0
		while g < generations:	# thes actual genetic algorithm
			bestRedScore = 10**100	# for lack of an actual max value, I've arbitrarily picked Googol, because I'm lazy.
			bestGreenScore = 10**100
			bestBlueScore = 10**100
			bestRedFunc = []
			bestGreenFunc = []
			bestBlueFunc = []

			for f in generation:		# pulls out the best function for each channel
				scores = test_function(f, source, xes, yys)
				if scores[0] < bestRedScore:
					bestRedScore = scores[0]
					bestRedFunc = f
				if scores[1] < bestGreenScore:
					bestGreenScore = scores[1]
					bestGreenFunc = f
				if scores[2] < bestBlueScore:
					bestBlueScore = scores[2]
					bestBlueFunc = f

			generation = [bestRedFunc, bestGreenFunc, bestBlueFunc]
			for i in range(1,generationSize):			# crafts a new generation in the winners' images (pun not intended)
				generation.append(mutate_function(bestRedFunc))
				generation.append(mutate_function(bestGreenFunc))
				generation.append(mutate_function(bestBlueFunc))

			g = g+1
			print g

		save_art(bestRedFunc, bestGreenFunc, bestBlueFunc, xes, yys, "approx{0:03d}.png".format(t))
		t = t+1


if __name__ == '__main__':
	import doctest
	doctest.testmod()
	evolve_painting("icon.jpg", 100)
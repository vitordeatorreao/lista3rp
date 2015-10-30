# Main.py

The main script runs each of the algorithms 10 times and takes an average of their error rates.

Usage:

```$ python main.py <FILEPATH> <PERCEPTRON-LEARNING-RATE> <PERCEPTRON-NUMBER-OF-EPOCHS> <KNN-K>```

# Using as library

To use it as a python module, you can see ```main.py``` for an example.
But importing each module separately like you would with any other python library should work.
Each module was designed so it can be used separately.

## dataaccess

A module for loading data from disk into memory. It is designed to read from CSV files and load them 
as Python's multidimensional arrays.

## neuralnetworks

A module containing classification algorithms based on Artificial Neural Networks. Currently, only 
Perceptron is available.

## distbased

A module containing classification algorithms based on Distance. Currently, only KNN is available.
There are 3 distance metrics available: euclidian, supreme and manhatam.

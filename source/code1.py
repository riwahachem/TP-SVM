import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Chargement Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# On garde seulement classes 1 et 2, et les deux premières variables
X = X[y != 0, :2]
y = y[y != 0]



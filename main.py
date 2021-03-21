# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:10:48 2021

@author: major
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 1].values

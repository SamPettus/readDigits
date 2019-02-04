import sys
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("digit-recognizer/train.csv").as_matrix()
test = pd.read_csv("digit-recognizer/test.csv").as_matrix()
clf = DecisionTreeClassifier() #creates tree object

#training dataset 
xTrain = data[0:, 1:]
train_label = data[0:,0]

clf.fit(xTrain, train_label)

#testing data
xTest = test[0:, 0:]
#d = xTest[8]


#d.shape= (28,28) #converts the image into a matrix
#pt.imshow(255 - d, cmap = 'gray') #plots the image using imshow()
#if pt.get_fignums():
#	print("YES")
#print(clf.predict([xTest[8]]))
#pt.show()

for i in range(0,28000):
	d = xTest[i]
	d.shape= (28,28) 
	pt.imshow(255 - d, cmap = 'gray')
	print(clf.predict([xTest[i]]))
	pt.show()
	done = input("0 if done ")
	if(done=='0'):
		break




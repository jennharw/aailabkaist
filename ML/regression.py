import numpy as np
import csv
import matplotlib.pyplot as plt

#X : Feature Variable https://raw.githubusercontent.com/aailabkaist/Introduction-to-Artificial-Intelligence-Machine-Learning/master/Week02/X.csv
#Y : Dependent Variable https://raw.githubusercontent.com/aailabkaist/Introduction-to-Artificial-Intelligence-Machine-Learning/master/Week02/Y.csv
X = [], Y = [] 
with open('X.csv', 'r') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        X.append(row)
with open ('Y.csv', 'r') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        Y.append(Y)

X = np.asarray(X, dtype = 'float64')
Y = np.asarray(Y, dtype = 'float64')

#X, theta(θ), Y_est 

xTemp = X[:, 0:2] #One Feature Variable

theta = np.dot( np.dot( np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)) ,Y)

Y_est = np.dot(X, theta)


# m0, c0 = argmin |Y - (m0 * xYemp + c0)|^2
# m1, c1 = argmin |Y_est - (m1 * xYemp + c1)|^2
m0, c0 = np.linalg.lstsq(xTemp, Y)[0]
m1, c1 = np.linalg.lstsq(xTemp, Y_est)[0]

#graph
plt.figure(1, figsize=(17,5))
ax = plt.plot()
plt.plot(X[:, 1], Y, 'ro')
plt.plot(X[:, 1], Y_est, 'bo')
plt.plot(X[:, 1], m1+c1*X[:, 1], 'b-')
plt.plot(X[:, 1], m0+c0*X[:, 1], 'r-')

plt.xlabel('Feature Variable', fontsize = 14)
plt.xlabel('Dependent Variable', fontsize = 14)



#polynomial
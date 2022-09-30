import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math


def getMeasurement(updateNumber):  #simulating measurement of position and velocity
    if updateNumber == 1:  #start position and velocity
        getMeasurement.currentPosition = 0
        getMeasurement.currentVelocity = 60  # m/s

    dt = 0.1  #time interval

    w = 8 * np.random.randn(1)  #noise of measuring the velocity
    v = 8 * np.random.randn(1)  #noise of measuring the position

    z = getMeasurement.currentPosition + getMeasurement.currentVelocity * dt + v
    getMeasurement.currentPosition = z - v
    getMeasurement.currentVelocity = 60 + w
    return [z, getMeasurement.currentPosition, getMeasurement.currentVelocity]


def filter(z, updateNumber):
    dt = 0.1
    if updateNumber == 1:  # Initialize State
        filter.x = np.array([[0], [20]])
        filter.Sigma = np.array([[5, 0], [0, 5]])  #start covariance matrix

        filter.A = np.array([
            [1, dt],  #transition from previous state
            [0, 1]
        ])
        filter.H = np.array([[1, 0]])  #state to measurement transition matrix
        filter.HT = np.array([[1], [0]])
        filter.R = 10  #measurement covariance matrix(value)
        filter.Q = np.array([
            [1, 0],  #inaccuracy of system model
            [0, 3]
        ])

    # Predict State Forward
    x_overline = filter.A.dot(filter.x)
    # Predict Covariance Forward
    Sigma_overline = filter.A.dot(filter.Sigma).dot(filter.A.T) + filter.Q
    # Compute Kalman Gain
    S = filter.H.dot(Sigma_overline).dot(filter.HT) + filter.R
    K = Sigma_overline.dot(filter.HT) * (1 / S)

    # Estimate State
    residual = z - filter.H.dot(x_overline)  #difference between measurement and temporary predicted
    filter.x = x_overline + K * residual

    # Estimate Covariance
    filter.Sigma = Sigma_overline - K.dot(filter.H).dot(Sigma_overline)

    return [filter.x[0], filter.x[1], filter.Sigma]
    #position, velocity and covariance matrix

def testFilter():
    dt = 0.1
    t = np.linspace(0, 10, num=300)
    numOfMeasurements = len(t)

    measTime = []
    measPos = []
    measDifPos = []
    estDifPos = []
    estPos = []
    estVel = []
    sumDifMeas = 0
    sumDifFilt = 0

    for k in range(1,numOfMeasurements):
        z = getMeasurement(k)#on the first time everything will be initialized
        # Call Filter and return new State
        f = filter(z[0], k)
        # Save off that state so that it could be plotted
        measTime.append(k)
        measPos.append(z[0])
        measDifPos.append(z[0]-z[1])#noise of measurement
        estDifPos.append(f[0]-z[1])#filter error
        sumDifMeas += (z[0] - z[1]) ** 2
        sumDifFilt += (f[0] - z[1]) ** 2
        estPos.append(f[0])
        estVel.append(f[1])

    return [measTime, measPos, estPos, estVel, measDifPos, estDifPos, sumDifMeas, sumDifFilt];

t = testFilter()
print(math.sqrt(t[6]), math.sqrt(t[7]))

plot1 = plt.figure(1)
plt.scatter(t[0], t[1])
plt.plot(t[0], t[2], color = 'red')
plt.ylabel('Position')
plt.xlabel('Time')
plt.grid(True)

plot2 = plt.figure(2)
plt.plot(t[0], t[3])
plt.ylabel('Velocity (meters/seconds)')
plt.xlabel('Update Number')
plt.grid(True)

plot3 = plt.figure(3)
plt.scatter(t[0], t[4], color = 'red')
plt.plot(t[0], t[5])
plt.scatter(t[0], t[5], color = 'orange')
plt.legend(['Estimate', 'Measurement'])
plt.title('Position Errors On Each Measurement Update \n', fontweight="bold")
#plt.plot(t[0], t[6])
plt.ylabel('Position Error (meters)')
plt.xlabel('Update Number')
plt.grid(True)

plt.show()




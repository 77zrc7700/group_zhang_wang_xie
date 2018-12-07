import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from CVAclass import *
CDSSpread=pd.DataFrame([[18.01,20.98,23.06,31.16,34.92,39.13],[52.06,78.08,77.89,104.7,119.97,125.96],[247.19,237.08,229.04,217.03,199.01,149],[59.32,75.13,83,105.45,116,126]],columns=['1Y','2Y','3Y','4Y','5Y','10Y'],index=[1,2,3,4])
index=CDSSpread.index
# curve=CDSSpread.loc[index[0]]
# for i in range(4):
#     testBootstrap=CDSBootstrapper(0.01,0.5,CDSSpread.loc[index[i]])
#     testBootstrap.getQ()
#     plt.plot(testBootstrap.Q)
#     plt.show()


r=0.01
price=100
maturity=2
strike=100
vol=0.31
recover=0.5
NumofPaths=1500
NumofIntervals=365*2
parameter=[0.1,0.1,0.1]
greek_letterz=[chr(code) for code in range(945,970)]
greek=greek_letterz[16]
rousample=np.linspace(-0.75,0.75,7)
time=np.linspace(0,2,NumofIntervals+1)
CVA1=np.zeros((len(index),len(rousample)))
AverageExposure=np.zeros((len(index),NumofIntervals+1))
ExpectedLoss=np.zeros((len(index),NumofIntervals+1))
CVAsensitivity=np.zeros((len(index),len(rousample)))
# rou = rousample[0]
# curve = CDSSpread.loc[index[1]]
# testCVA = CVAclass(price, maturity, strike, r, vol, rou, recover, curve, NumofPaths, NumofIntervals)
# results=testCVA.optimizeParameter(parameter)
# print(results)
# testCVA.MCSimulation(results)
# testCVA.AverageExposure()
# plt.plot(testCVA.CVA())
# plt.show()


for i in range(len(rousample)):
    for j in range(len(index)):
        rou=rousample[i]
        curve=CDSSpread.loc[index[j]]
        testCVA = CVAclass(price, maturity, strike, r, vol, rou, recover, curve, NumofPaths, NumofIntervals)
        x=testCVA.time
        results = testCVA.optimizeParameter(parameter)
        testCVA.MCSimulation(results)
        CVA1[j,i]=testCVA.CVA()
        AverageExposure[j,:]=testCVA.AverageExposure()
        ExpectedLoss[j,:]=testCVA.expectedLoss
        CVAsensitivity[j,i]=testCVA.SensitivityBy1bp()
    plt.subplot(211)
    l1, = plt.plot(time, AverageExposure[0, :])
    l2, = plt.plot(time, AverageExposure[1, :])
    l3, = plt.plot(time, AverageExposure[2, :])
    l4, = plt.plot(time, AverageExposure[3, :])
    plt.xlabel('Time')
    plt.ylabel('Average Exposure')
    plt.title('Average Exposure for 4 CDS spread curves when ' + greek + '='+str(rou))
    plt.legend(handles=[l1, l2, l3, l4, ], labels=['Curve1', 'Curve2', 'Curve3', 'Curve4'],loc='center right')
    plt.subplot(212)
    l1, = plt.plot(time, ExpectedLoss[0, :])
    l2, = plt.plot(time, ExpectedLoss[1, :])
    l3, = plt.plot(time, ExpectedLoss[2, :])
    l4, = plt.plot(time, ExpectedLoss[3, :])
    plt.xlabel('Time')
    plt.ylabel('Expected Loss')
    plt.title('Expected Loss for 4 CDS spread curves when ' + greek + '='+str(rou))
    plt.legend(handles=[l1, l2, l3, l4, ], labels=['Curve1', 'Curve2', 'Curve3', 'Curve4'],loc='center right')
    plt.show()


l1, = plt.plot(rousample, CVA1[0,:])
l2, = plt.plot(rousample, CVA1[1,:])
l3, = plt.plot(rousample, CVA1[2,:])
l4, = plt.plot(rousample, CVA1[3,:])
plt.xlabel('Correlation Values')
plt.ylabel('CVA')
plt.title('CVA for 4 CDS spread curves when '+greek+' takes different values')
plt.legend(handles=[l1, l2, l3,l4,], labels=['Curve1','Curve2','Curve3','Curve4' ])
plt.show()
l1, = plt.plot(rousample, CVAsensitivity[0,:])
l2, = plt.plot(rousample, CVAsensitivity[1,:])
l3, = plt.plot(rousample, CVAsensitivity[2,:])
l4, = plt.plot(rousample, CVAsensitivity[3,:])
plt.xlabel('Correlation Values')
plt.ylabel('CVA Sensitivity')
plt.title('CVA sensitivity for 4 CDS spread curves when '+greek+' takes different values')
plt.legend(handles=[l1, l2, l3,l4,], labels=['Curve1','Curve2','Curve3','Curve4' ])
plt.show()
print(rousample)
print(index)
print(CVA1)
print(CVAsensitivity)
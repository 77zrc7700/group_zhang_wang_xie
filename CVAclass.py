import numpy as np
import pandas as pd
import math
import re
from scipy.optimize import minimize

class CDSBootstrapper(object):
    def __init__(self,r1,R1,curve1):#curve - pandas.Series, r-interest rate, R-recover rate
        self.r=r1
        self.curve=curve1.copy()
        self.Q=[0]*curve1.size
        self.R=R1
        self.time=0
        self.h0=0

    def getQ(self):
        spread=self.curve
        time=spread.index.tolist()    #index format (...)Y
        index=['0Y']+time
        for i in range(len(spread)):
            spread[i]=spread[i]*0.0001
        r=self.r
        for i in range(len(time)):
            time[i]=float(re.findall(r"\d+\.?\d*",time[i])[0])
        sum1=0
        for i in range(len(self.Q)):
            self.Q[i]=((1-self.R)*math.exp(-time[i]*self.r)-spread[i]*sum1)/(1-self.R+spread[i])*math.exp(-time[i]*self.r)
            sum1=sum1+self.Q[i]*math.exp(-self.r*time[i])
        self.Q=[1]+self.Q
        self.Q=pd.Series(self.Q,index=index)
        time=[0]+time
        self.time=time
        self.h0 = -math.log(self.Q[ 1] / self.Q[0]) / (time[1] - time[0])

class CVAclass(object):
    def __init__(self,stockprice,maturity,strike,r,vol,rou,Recover,curve,NumofPaths,NumofIntervals):#r-interest rate; vol-stock volatility; rou-correlation between dW(t) and dV(t);R-recover rate; curve-par spread
        self.StockPrice0=stockprice
        self.r=r
        self.vol=vol
        self.maturity=maturity #how long the underlying products will expire (in years)
        self.strike=strike
        self.Q=0
        self.rou=rou
        self.StockPaths=0
        self.hPaths=0
        self.R=Recover
        self.curve=curve#curve-pd.Series
        self.NumofPaths=NumofPaths
        self.NumofIntervals=NumofIntervals
        self.time=np.linspace(0,maturity,NumofIntervals+1)
        self.expectedLoss=0
        Bootstrapper=CDSBootstrapper(r,Recover,curve)
        Bootstrapper.getQ()
        self.h0=Bootstrapper.h0

    def errorfunc(self,x,Q,time):
        AnalyticalQ=self.AnalyticalDefaultProbability(x,time)
        for i in range(len(Q)):
            AnalyticalQ[i]=AnalyticalQ[i]-Q[i]
        error=sum([c*c for c in AnalyticalQ])
        return error

    def optimizeParameter(self,x):
        temp=CDSBootstrapper(self.r,self.R,self.curve)
        temp.getQ()
        Q=temp.Q
        time=temp.time
        results=minimize(fun=self.errorfunc, x0=x,args=(Q,time),method="Nelder-Mead")
        return results.x

    def MCSimulation(self,x):#x-parameters in MC simulation;
        NumOfIntervals=self.NumofIntervals
        NumOfPaths=self.NumofPaths
        dt=self.maturity/NumOfIntervals
        #x[0]=kappa, x[1]=theta, x[2]=sigma
        sigma=x[2]
        stockt=np.zeros((NumOfPaths,NumOfIntervals+1))
        ht=np.zeros((NumOfPaths,NumOfIntervals+1))
        for i in range(NumOfPaths):
            stockt[i,0]=self.StockPrice0
            ht[i,0]=self.h0
            for j in range(NumOfIntervals):
                temp=np.random.normal(0, 1, 1)[0]
                stockt[i,j+1]=stockt[i,0]+stockt[i-1,j]*(self.r*dt+self.vol * temp* math.sqrt(dt)) # sigma of stock price is different from sigma of hazard rate
                ht[i,j+1]=ht[i,0]+x[0]*(x[1]-ht[i,j])*dt+sigma*(self.rou*temp*math.sqrt(dt)+math.sqrt(dt*(1-self.rou*self.rou))*np.random.normal(0, 1, 1)[0])
        self.hPaths=ht
        self.StockPaths=stockt
        Q=np.zeros((NumOfPaths,NumOfIntervals+1))
        for i in range(NumOfPaths):
            Q[i,0]=1
            for j in range(NumOfIntervals):
                Q[i,j+1]=Q[i,j]*math.exp(-dt*ht[i,j])
        self.Q=Q

    def expectation(self,time1=None):#time is when to estimate  survival probability Q time format:[1,3,5,7] number as elements
        time=self.time
        if (time1 is not None):
            time=time1
        NumOfPaths=self.NumofPaths
        NumOfIntervals=self.NumofIntervals
        self.maturity=time[-1]
        Q=[0]*(NumOfIntervals+1)
        testQ=[0]*len(time)
        for j in range(NumOfIntervals+1):
            sum1=0
            for i in range(NumOfPaths):
                sum1=sum1+self.Q[i, j]
            Q[j]=sum1/NumOfPaths
        dt=self.maturity/NumOfIntervals# dt in years
        for i in range(len(time)):
            temp=math.floor(time[i]/dt)
            if (time[i]/dt*dt-time[i]!=0):
                testQ[i]=Q[temp]*((temp+1)*dt-time[i])+Q[temp+1]*(time[i]-temp*dt)
            else:
                testQ[i]=Q[temp]
        return testQ#return certain dates' survival probability

    def AverageExposure(self):
        FV=np.zeros(self.StockPaths.shape)
        Qdensity=np.zeros(self.Q.shape)
        dt=self.maturity/self.NumofIntervals
        for i in range(Qdensity.shape[0]):
            for j in range(Qdensity.shape[1]):
                Qdensity[i,j]=self.Q[i,j]*self.hPaths[i,j]
                loss=max(0,self.StockPaths[i,j]-self.strike*math.exp(-(self.maturity-i*dt)))
                FV[i,j]=loss*Qdensity[i,j]  #E[FV(t)* lambda(t)*Q(t)]
        expectedFV=[0]*FV.shape[1]
        expectedQdensity=[0]*Qdensity.shape[1]
        AverageExpo=[0]*FV.shape[1]
        for i in range(FV.shape[1]):
            sum1=0
            sum2=0
            for j in range(FV.shape[0]):
                sum1=sum1+FV[j,i]
                sum2=sum2+Qdensity[j,i]
            expectedFV[i]=sum1/FV.shape[0]
            expectedQdensity[i]=sum2/FV.shape[0]
            AverageExpo[i]=expectedFV[i]/expectedQdensity[i]
        self.expectedLoss=expectedFV
        return AverageExpo

    def CVA(self):
        self.AverageExposure()
        sum1=0
        dt=self.maturity/self.NumofIntervals
        for i in range(len(self.expectedLoss)):
            temp=math.exp(-i*dt)*self.expectedLoss[i]*dt
            sum1=sum1+temp
        return sum1

    def SensitivityBy1bp(self,Up=1):
        sum1=self.CVA()
        if (Up==1):
            self.curve=pd.Series([c+1 for c in self.curve],index=self.curve.index)
        else:
            self.curve=pd.Series([c-1 for c in self.curve],index=self.curve.index)
        results=self.optimizeParameter([1,1,1,1])
        self.MCSimulation(results)
        sum2=self.CVA()
        return (sum2-sum1)

    def AnalyticalDefaultProbability(self,x,time):
        #x[0]=kappa, x[1]=theta, x[2]=sigma
        AnalyticalQ=[0]*len(time)
        for i in range(len(AnalyticalQ)):
            B=(1-np.exp(-x[0]*time[i]))/x[0]
            A=np.exp((x[1]-x[2]*x[2]/x[0]/x[0]/2)*(B-time[i])-x[2]*x[2]*B*B/x[0]/4)
            AnalyticalQ[i]=A*np.exp(-B*self.h0)
        return AnalyticalQ
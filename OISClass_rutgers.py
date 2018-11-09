import matplotlib.pyplot as plt
import quandl
from fredapi import Fred
from scipy.optimize import minimize
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from numba import jit
import os


class Scheduler(object):
    def getSchedule(self, start, end, freq):
        delay = self.extractDelay(freq=freq)
        maturity = start + self.extractDelay(end)
        date0 = maturity
        self.datelist = [date0]
        while (date0 > start):
            date0 -= delay
            self.datelist.append(date0)
        self.datelist = sorted(self.datelist)
        return self.datelist

    def extractDelay(self, freq):
        delta = relativedelta(months=+ 0)
        freqValue = np.int(self.only_numerics(seq=freq))
        if (freq.find('M') != -1):
            delta = relativedelta(months=+ freqValue)
        if (freq.find('Y') != -1):
            delta = relativedelta(years=+ freqValue)
        if (freq.find('D') != -1):
            delta = relativedelta(days=+ freqValue)
        if (freq.find('W') != -1): delta = relativedelta(days=+ 7)
        return delta

    @classmethod
    def datediff(self, date1, date2):
        if type(date1) == pd.DataFrame:
            datedifference = pd.DataFrame([x[0].days / 365.0 for x in (date2.values - date1.values)])
            return datedifference
        else:
            if type(date1) in [date, datetime]:
                if date1 > date2:
                    return (date1 - date2).days / 365
                return (date2 - date1).days / 365

    def only_numerics(self, seq):
        seq_type = type(seq)
        return seq_type().join(filter(seq_type.isdigit, seq))



class OISClass(object):
    def __init__(self):

        #TODO change your .bash_profile to include an export QUANDL_API_KEY and export FRED_API_KEY
        # this is how you keep your important information safe.

        self.QUANDL_API_KEY = os.environ.get('QUANDL_API_KEY')
        self.FRED_API_KEY = os.environ.get('FRED_API_KEY')
        self.CorpRates={}
        self.OISRates={}
        self.OIS=pd.DataFrame()
        self.Libor=pd.DataFrame()
        self.Z = pd.DataFrame()

    # Function getOIS was provided. It captures both OIS from Quandl and Corporate Spreads from FRED.
    def getOIS(self):
        trim_start = date(2018, 1, 1)
        trim_end = date(2018, 11, 1)
        myscheduler = Scheduler()
        self.OIS = 0.01 * quandl.get("USTREASURY/YIELD", authtoken=self.QUANDL_API_KEY, trim_start=trim_start,
                                trim_end=trim_end)
        self.OIS.columns = ['1M', '2M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30 Y']

        self.ntenors = len(self.OIS.columns)
        self.delays = [myscheduler.extractDelay(x) for x in self.OIS.columns]
        timebase = [date.today() + x for x in self.delays]
        self.OIS = self.OIS.T
        self.OIS["timebase"] = timebase
        self.OIS = self.OIS.T

        ones = np.ones([1, len(self.OIS.index)])
        corp = {}
        fred = Fred(api_key=self.FRED_API_KEY)
        self.index = {"AAA":"BAMLC0A1CAAA","AA":"BAMLC0A2CAA", "A":"BAMLC0A3CA", "BBB":"BAMLC0A4CBBB",
        "BB":"BAMLH0A1HYBB", "B":"BAMLH0A2HYB","C":"BAMLH0A3HYC"}
        for rating,code in self.index.items():
            corp[rating] = fred.get_series(code, observation_start=trim_start, observation_end=trim_end)

        corpDF = pd.DataFrame.from_dict(corp, orient="index").T
        self.OIS = pd.concat([self.OIS, corpDF], axis=1, join='outer')
        self.OIS= self.OIS.fillna(axis=1, method="ffill")

    # Function to extract OIS rates
    def getOISRates(self,thisdate):
    #TODO create this method.  This method save OISrates for a given date onto self.OISRates
        return self.OISRates

    # Function to extract corporate rates
    def getCorpRates(self,thisdate, rating):
        spread= self.OIS.loc[thisdate][rating]
    #TODO create this method This method saves OISrates yields for a given date and rating onto self.CorpRates[rating]
        self.CorpRates[rating] = pd.DataFrame(data=y.values, index=x, columns=[str(thisdate.date())])
        return self.CorpRates[rating]

    # Function to create Zero Coupon Discount Factor from curves (either OIS or Corp)
    def getZ(self, thisdate, rating):
        if rating=="OIS":
            rates =self.getOISRates(thisdate)
        else:
            rates = self.getCorpRates(thisdate,rating)
        y = {thisdate:1}

    # TODO create this method to create the Zero Coupon Discount Rate for a rating ["OIS","AAA", "AA"...] for a given date

        return pd.DataFrame.from_dict(data=y, orient="index", columns=[str(thisdate.date())])



    # Function to generate simnumber Monte Carlo Trajectories for the Vasicek Model
    # Starting at trim_start and ending at trim_end using the vector of parameters x
    def generateMCTrajectories(self, trim_start, trim_end, x, simnumber):
        # % Models - Vasicek CIR - Functional solution to Vasicek SDE
        # dLambda(t) = kappa*(theta-Lambda(t))*dt + sigma*dW(t)
        # self.kappa = x[0]
        # self.theta = x[1]
        # self.sigma = x[2]
        # self.r0 = x[3]

    # TODO complete this method  Generate Monte Carlo Trajectories.  I integrate daily, that means that for the
        # tenor spanning 40 years, there will be 14600 days or thereabouts (times the number of trajectories simnumber)
        # so this is  large array. I created an index on the resulting curve and added a date column.
        # so, you just need to create the random gaussian numbers and integrate VASICEK SDE

        self.Libor=pd.DataFrame(np.exp(-self.Libor.cumsum(axis=0)))
        self.Libor["date"]=self.datelistlong
        self.Libor.set_index("date")

    # Function to recreate large Monte Carlo Array and select only the datalist
    # or just select the datelist items if the large Monte Carlo Array is already created
    def getsmallZ(self,datelist,trim_start=None, trim_end=None, x=None, simnumber=None):
        if len(self.Libor)==0:
            self.generateMCTrajectories(trim_start, trim_end, x, simnumber)
        self.Z=pd.DataFrame(index=datelist)
        for t in datelist:
            ind = self.Libor.loc[self.Libor['date'] == t].index[0]
            self.Z.loc[t] = (self.Libor.iloc[ind].values).tolist()
        return self.Z

    # Function to get the expectation curve (simnumber of them) for the Vasicek Model and x=parameters vector
    def expectation(self, x, datelist, simNumber=1, start=None):
        # % Models - Vasicek CIR - Functional solution to Vasicek SDE
        # dLambda(t) = kappa*(theta-Lambda(t))*dt + sigma*dW(t)
        # self.kappa = x[0]
        # self.theta = x[1]
        # self.sigma = x[2]
        # self.r0 = x[3]
    # TODO create this method with the VASICED Expected Bond Value (Z).
        Q=None
        return Q

    # Function to create errorfunc for discrepancies between curve and expectation created with datelist and z
    def errorfunc(self, x, datelist, curve):
        results = np.sum((self.expectation(datelist=datelist, x=x) - curve) ** 2)
        return results[0]

    # Standard input for the minimize function of the scipy package
    def optmizeparameters(self, datelist, x, curve):
        results = minimize(fun=self.errorfunc, x0=np.array(x), args=(datelist, curve.values), method="Nelder-Mead")
        return results.x

 #####################################################################################
    @jit
    def return_indices1_of_a(self, a, b):
        b_set = set(b)
        ind = [i for i, v in enumerate(a) if v in b_set]
        return ind

    #####################################################################################
    @jit
    def return_indices2_of_a(self, a, b):
        index = []
        for item in a:
            index.append(np.bisect.bisect(b, item))
        return np.unique(index).tolist()

    def pickleMe(self, file):
        pickle.dump(self, open(file, "wb"))

    def unPickleMe(self, file):
        if os.path.exists(file):
            self = pickle.load(open(file, "rb"))



if __name__=="__main__":
    myOIS=OISClass()
    myOIS.getOIS()
    thisdate=myOIS.OIS.index[0]
    datelist=[thisdate + x for x in myOIS.delays]
    curve=myOIS.expectation(datelist=datelist,x=[0.4,0.3,0.04,0.5])
    x=[0.1,0.1,0.1,0.1]
    results = myOIS.optmizeparameters(datelist,x,curve)
    a=1
    curveOut = myOIS.expectation(datelist=datelist, x=results)
    plt.plot(datelist,curve,datelist, curveOut)
    plt.show()

    OISPlot = myOIS.getOISCurve(thisdate)
    OISPlot.plot()
    plt.show()
    y=myOIS.getZ(thisdate, "OIS")
    plt.plot(y)
    plt.show()

    trim_start=thisdate
    trim_end=trim_start+relativedelta(years=+ 2)

    myOIS.generateMCTrajectories(trim_start=trim_start,trim_end=trim_end,x=results,simnumber=10)


    a=1

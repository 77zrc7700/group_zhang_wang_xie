from unittest import TestCase
from OISClass_rutgers import OISClass
import matplotlib.pyplot as plt
import pandas as pd

curves=pd.DataFrame()
myOIS = OISClass()
myOIS.getOIS()
thisdate = myOIS.OIS.index[0]
trim_start=thisdate
trim_end=thisdate + myOIS.delays[-1:][0]
simnumber=10
datelist = [thisdate + x for x in myOIS.delays]
curve = myOIS.expectation(datelist=datelist, x=[0.4, 0.3, 0.04, 0.5])
x = [0.1, 0.1, 0.1, 0.1]
results = myOIS.optmizeparameters(datelist, x, curve)
a = 1
curveOut = myOIS.expectation(datelist=datelist, x=results)
plt.plot(datelist, curve, datelist, 1.1 * curveOut + 0.02)
plt.show()

OISPlot = myOIS.getOISRates(thisdate)
OISPlot.plot()
plt.show()
y = myOIS.getZ(thisdate, "OIS")
plt.plot(y)
plt.show()


class TestOISClass(TestCase):
    def test_errorfunc(self):
        myOIS.errorfunc(datelist=datelist, x=x, curve=curve)

    def test_optmizeparameters(self):
        x = [0.1, 0.1, 0.1, 0.1]
        curve = myOIS.expectation(x, datelist)
        plt.plot(datelist, curve, datelist, 1.1 * curve + 0.02)
        plt.show()
        results = myOIS.optmizeparameters(datelist, x, curve)
        print(results)
        curveOut = myOIS.expectation(results, datelist)
        plt.plot(datelist, curve, datelist, 1.1 * curveOut + 0.02)
        plt.show()

    def test_getOIS(self):
        myOIS.getOIS()

    def test_getOISRates(self):
        myOIS.getOISRates(thisdate=thisdate)
        OISPlot.plot()
        plt.show()

    def test_getCorpRates(self):
        print(myOIS.getCorpRates(thisdate=thisdate, rating="AA"))
        a=1

    def test_getZ(self):
        y = myOIS.getZ(thisdate=thisdate, rating="OIS")
        plt.plot(y)
        plt.show()
        y = myOIS.getZ(thisdate=thisdate, rating="AA")
        plt.plot(y)
        plt.show()


    def test_expectation(self):
        curve = myOIS.expectation(datelist=datelist, x=[0.4, 0.3, 0.04, 0.5])
        curveOut = myOIS.expectation(datelist=datelist, x=results)

    def test_generateMCTrajectories(self):
        myOIS.generateMCTrajectories(trim_start=trim_start, trim_end=trim_end,x=results,simnumber=10)

    def test_getsmallZ(self):
        z = myOIS.getsmallZ(datelist,trim_start, trim_end, x, simnumber)
        a=1


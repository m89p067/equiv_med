import numpy as np
import scipy.stats as stats
from statistics import stdev
from math import sqrt
class Retesting():
    is_normal=True
    same_var=True
    def __init__(self,meas1,meas2):
        ''' Repeated measurements using the same instrument (at t1 and t2)'''
        print('Input data are two repeated measurements')
        print('using the same instrument at t1 and t2')
        stat, p = stats.levene(meas1,meas2)
        if p<0.05:
            print('Input measurements do not have equal variances')
            Retesting.same_var=False
        shapiro_test1 = stats.shapiro(meas1)
        shapiro_test2 = stats.shapiro(meas2)
        if shapiro_test1.pvalue>0.05 and shapiro_test2.pvalue>0.05:
            Retesting.is_normal=True            
        else:
            Retesting.is_normal=False
            print('Input measurements are not normally distributed')
        self.x=np.asarray(meas1)
        self.y=np.asarray(meas2)
    def mdc(self,confidence_level=0.95,one_tail=False):
        if confidence_level>1 or confidence_level<99:
            confidence_level=confidence_level/100
        data1=self.x.copy()
        data2=self.y.copy()
        if Retesting.is_normal:
            r_value=stats.pearsonr(data1,data2)[0]
        else:
            r_value=stats.spearmanr(data1,data2)[0]
        SEM=stdev(data1)*sqrt(1-r_value)
        if one_tail:
            Z_score=stats.norm.ppf(confidence_level)
        else:
            Z_score=abs(stats.norm.ppf((1-confidence_level)/2))
        MDC=Z_score*sqrt(2*SEM)
        print('Minimal Detectable Change :',MDC)
        return MDC
        

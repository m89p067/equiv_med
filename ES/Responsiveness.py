import numpy as np
import scipy.stats as stats
from statistics import stdev
from math import sqrt
class Responsiveness():
    is_normal=True
    same_var=True
    def __init__(self,control_t1,control_t2,treatment_t1,treatment_t2):
        print('First two inputs assumed as baseline/control measurement at t1 and t2')
        print('Second two inputs assumed as treatment/improved measurement at t1 and t2')
        stat, p = stats.levene(control_t1,control_t2,treatment_t1,treatment_t2)
        if p<0.05:
            print('Input measurements do not have equal variances')
            Retesting.same_var=False
        shapiro_test1 = stats.shapiro(control_t1)
        shapiro_test2 = stats.shapiro(control_t2)
        shapiro_test3 = stats.shapiro(treatment_t1)
        shapiro_test4 = stats.shapiro(treatment_t2)
        if shapiro_test1.pvalue>0.05 and shapiro_test2.pvalue>0.05 and shapiro_test3.pvalue>0.05 and shapiro_test4.pvalue>0.05:
            Retesting.is_normal=True            
        else:
            Retesting.is_normal=False
            print('Input measurements are not normally distributed')
        self.c1=np.asarray(control_t1)
        self.c2=np.asarray(control_t2)
        self.t1=np.asarray(treatment_t1)
        self.t2=np.asarray(treatment_t2)
    def grc(self):
        # Guyatt’s responsiveness coefficient
        # Requires a second measurement at time t2 of the control/basaline group
        if Retesting.is_normal :
            grc=(np.mean(self.t2)-np.mean(self.t1))/stdev(self.c2-self.c1)
        else:
            print('Computation discouraged cause not normally distributed data')
            grc=np.nan
        return grc
    def srm(self):
        # Standardized response mean
        # Variability in change scores
        total_t1=np.concatenate((self.c1,self.t1))
        total_t2=np.concatenate((self.c2,self.t2))
        if Retesting.is_normal:
            srm=(np.mean(total_t2)-np.mean(total_t1))/stdev(total_t2-total_t1)
        else:
            print('Computation discouraged cause not normally distributed data')
            srm=np.nan
        return srm
    def rci(self):
        # Jacobson Reliable change index
        # Include factor 1.96 as regularization term
        total_t1=np.concatenate((self.c1,self.t1))
        total_t2=np.concatenate((self.c2,self.t2))
        if Retesting.is_normal:
            rci=(np.mean(total_t2)-np.mean(total_t1))/(1.96*stdev(total_t2-total_t1))
        else:
            print('Computation discouraged cause not normally distributed data')
            rci=np.nan
        return rci
    def es(self):
        # Effect size
        total_t1=np.concatenate((self.c1,self.t1))
        total_t2=np.concatenate((self.c2,self.t2))
        if Retesting.is_normal:
            es=(np.mean(total_t2)-np.mean(total_t1))/stdev(total_t1)
        else:
            print('Computation discouraged cause not normally distributed data')
            es=np.nan
        return es
    def nr(self):
        # Normalized ratio
        if Retesting.is_normal:
            nr=(np.mean(self.t2)-np.mean(self.t1))/stdev(self.c1)
        else:
            print('Computation discouraged cause not normally distributed data')
            nr=np.nan
        return nr        
    def ses(self):
        # Standardized effect size
        if Retesting.is_normal:
            ses=(np.mean(self.t2)-np.mean(self.t1))/stdev(self.t1)
        else:
            print('Computation discouraged cause not normally distributed data')
            ses=np.nan
        return ses         

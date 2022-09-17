import numpy as np
import scipy.stats as stats
from statistics import stdev,variance
from math import sqrt
def pt(q,df,ncp=0):
    if ncp==0:
        result=stats.t.cdf(x=q,df=df,loc=0,scale=1)
    else:
        result=stats.nct.cdf(x=q,df=df,nc=ncp,loc=0,scale=1)
    return result
class IoS():
    is_normal=True
    def __init__(self,x1,y1):
        ''' 
        Non-inferiority or Superiority test for 2 samples
        Args:
        x1,y1 two set of measurements, could be lists or numpy 1D vectors 
        '''
        self.data1=np.asarray(x1)
        self.data2=np.asarray(y1)
        normality1=stats.shapiro(x1)
        normality2=stats.shapiro(y1)
        if normality1.pvalue>0.05 and normality2.pvalue>0.05:
            IoS.is_normal=True
            print('Data are normally distributed')
        else:
            IoS.is_normal=False
            print('Data are not normally distributed, test is discouraged')
    def boot_ci(self,Conf_Int=0.95):
        mu=np.mean(self.x)
        sem=stats.sem(self.x)
        if Conf_Int>1 or Conf_Int<99:
            Conf_Int=Conf_Int/100
        out_ci=stats.norm.interval(alpha=Conf_Int,df=len(self.x)-1, loc=mu, scale=sem)
        estimates=[(1-Conf_Int)/2,Conf_Int+((1-Conf_Int)/2)]
        return out_ci,estimates
    def non_inferiority(self,ni_bound=0.1,shift=0):
        actual_zero = shift
        x=self.data1
        y=self.data2
        if actual_zero !=0:
            print('Shift should usually be zero for two-sample problems')
        # Assuming the boundary is below zero
        if ni_bound<0:
            ni_bound=abs(ni_bound)
        if IoS.is_normal==True:
            nx=len(x)
            ny=len(y)
            mx=np.mean(x)
            vx=variance(x)
            my=np.mean(y)
            vy=variance(y)
            v = (nx - 1) * vx + (ny - 1) * vy
            df=nx+ny-2
            v=v/df
            stderr =sqrt(v * (1/nx + 1/ny))
            ncp = sqrt(nx * ny) * ni_bound/sqrt(nx + ny)
            if stderr < (10 * np.finfo(float).eps * abs(mx)) :
                stop("data are essentially constant")
            tstat= (mx -my- actual_zero)/stderr
            d = (mx - my-actual_zero)/sqrt(v)
            # Greater aka Non-Inferiority
            pval = 1-pt(tstat, df, ncp = -ncp)
            print("Two Sample non-inferiority test")
            print('Lower null value = -Inf')
            print('Upper null value = ',-ni_bound)
            print('t-statistic :',tstat)
            print('Degrees of Freedom :',df,' NCP :',ncp)
            print('p value :',pval)
            if pval<0.05:
                print('-> Non-inferiority proved (aka no worse than)')
                print('NOTE: this result does not mean superiority')
            else:
                print('New method is worse than the other by more than -',ni_bound) # H0
        else:
            print('Not normally distributed input data')
    def superiority(self,sup_bound=0.1,shift=0):
        actual_zero = shift
        x=self.data1
        y=self.data2
        if actual_zero !=0:
            print('Shift should usually be zero for two-sample problems')
        if sup_bound<0:
            sup_bound=abs(sup_bound)        
        if IoS.is_normal==True:
            nx=len(x)
            ny=len(y)
            mx=np.mean(x)
            vx=variance(x)
            my=np.mean(y)
            vy=variance(y)
            v = (nx - 1) * vx + (ny - 1) * vy
            df=nx+ny-2
            v=v/df
            stderr =sqrt(v * (1/nx + 1/ny))
            ncp = sqrt(nx * ny) * sup_bound/sqrt(nx + ny)
            if stderr < (10 * np.finfo(float).eps * abs(mx)) :
                stop("data are essentially constant")
            tstat= (mx -my- actual_zero)/stderr
            d = (mx - my-actual_zero)/sqrt(v)
            # Lesser aka Superiority
            pval = pt(tstat, df, ncp = ncp)
            print("Two Sample non-superiority test")
            print('Lower null value = ',sup_bound)
            print('Upper null value = Inf')            
            print('t-statistic :',tstat)
            print('Degrees of Freedom :',df,' NCP :',ncp)
            print('p value :',pval)
            if pval>0.05: #H0
                print('-> Superiority proved (there is a difference between the provided measurements)')
            else:
                print('Non-superiority (no true difference between the provided measurements)')
        else:
            print('Not normally distributed input data')    
class IoS_one():
    is_normal=True
    def __init__(self,x1):
        ''' Non-inferiority or Superiority test for 1 sample'''
        self.data3=np.asarray(x1)
        
        normality3=stats.shapiro(x1)
        
        if normality3.pvalue>0.05 :
            IoS_one.is_normal=True
            print('Data are normally distributed')
        else:
            IoS_one.is_normal=False
            print('Data are not normally distributed, test is discouraged')        
    def non_inferiority(self,ni_bound=0.1,ref_parameter =0):
        
        x=self.data3
        
        # Assuming the boundary is below zero
        if ni_bound<0:
            ni_bound=abs(ni_bound)
        if IoS_one.is_normal==True:
            nx=len(x)
            
            mx=np.mean(x)
            vx=variance(x)
            
            
            df=nx-1
            
            stderr = sqrt(vx/nx)
            ncp = sqrt(nx) * ref_parameter
            if stderr < (10 * np.finfo(float).eps * abs(mx)) :
                stop("data are essentially constant")
            tstat= (mx -ref_parameter )/stderr
            d = (mx - ref_parameter )/sqrt(vx)
            # Greater aka Non-Inferiority
            pval = 1-pt(tstat, df, ncp = -ncp)
            print("One Sample non-inferiority test")
            print('Lower null value = -Inf')
            print('Upper null value = ',-ni_bound)
            print('t-statistic :',tstat)
            print('Degrees of Freedom :',df,' NCP :',ncp)
            print('p value :',pval)
            if pval<0.05:
                print('-> Non-inferiority proved (aka no worse than the known value)')
                print('NOTE: this result does not mean superiority')
            else:
                print('New method is worse than the known value -',ni_bound) # H0
        else:
            print('Not normally distributed input data')
    def superiority(self,sup_bound=0.1,ref_parameter=0):
        
        x=self.data3
        
        if sup_bound<0:
            sup_bound=abs(sup_bound)        
        if IoS_one.is_normal==True:
            nx=len(x)
            
            mx=np.mean(x)
            vx=variance(x)
            
           
            df=nx-1
            
            stderr = sqrt(vx/nx)
            ncp = sqrt(nx) * ref_parameter
            if stderr < (10 * np.finfo(float).eps * abs(mx)) :
                stop("data are essentially constant")
            tstat= (mx -ref_parameter)/stderr
            d = (mx - ref_parameter)/sqrt(vx)
            # Lesser aka Superiority
            pval = pt(tstat, df, ncp = ncp)
            print("One Sample non-superiority test")
            print('Lower null value = ',sup_bound)
            print('Upper null value = Inf')            
            print('t-statistic :',tstat)
            print('Degrees of Freedom :',df,' NCP :',ncp)
            print('p value :',pval)
            if pval>0.05: #H0
                print('-> Superiority proved (there is a difference between the provided measurement\n and the reference parameter)')
            else:
                print('Non-superiority (no true difference between the provided\n measurement and the reference value)')
        else:
            print('Not normally distributed input data')        
        
        

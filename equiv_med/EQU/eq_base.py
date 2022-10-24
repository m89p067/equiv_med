import numpy as np
import scipy.stats as stats
from math import sqrt,exp
from statistics import stdev,mean,variance
import statsmodels.api as sm
class init_eq:
    def __init__(self, x1, y1):
        self.x = x1
        self.y = y1
    def pooled_std(self):
        if len(self.x)==len(self.y):
            return sqrt( ( (stdev(self.x) **2)+(stdev(self.y) **2)   )/2)
        else:
            return sqrt( (((len(self.x) - 1)*(stdev(self.x) **2)) +( (len(self.y)-1)*(stdev(self.y) **2) ) )/ (len(self.x) + len(self.y)-2))
    def run_regression(self):
        X=self.y
        X = sm.add_constant(X)
        Y=self.x
        model =sm.OLS(Y,X)   
        results = model.fit()
  
        return results
    def degreesOfFreedom(self):
        _,check_var=stats.levene(self.x,self.y,center='mean')
        if check_var<0.05:
            s1 = (stdev(self.x)**2)
            s2 = (stdev(self.y)**2)
            dof1 = ((s1 / len(self.x)) +( s2 / len(self.y)))**2 
            dof2= (s1**2)/((len(self.x)**2) * (len(self.x)-1))
            dof3= (s2**2)/((len(self.y)**2) * (len(self.y)-1))
            dof=dof1/(dof2+dof3)
        else:
            dof=(len(self.x)+len(self.y)-2)
        return dof
    def standard_error(self):
        X=self.x
        Y=self.y
        _,check_var=stats.levene(X,Y,center='mean')
        if check_var<0.05:
            s1 =( stdev(X)**2) / len(X)
            s2 = (stdev(Y)**2)     / len(Y)   
            std_error=sqrt(s1+s2)
        else:
            s1 =( stdev(X)**2) *(len(X)-1)
            s2 = (stdev(Y)**2)  *( len(Y)-1)   
            first_term=(s1+s2)/(len(X)+len(Y)-2)
            second_term=(1/len(X))+(1/len(Y))
            std_error=sqrt(first_term*second_term)
        return std_error
    def do_bootstrap(self,my_data, n,p): # n number for boostrapping operations
        #unlike confidence intervals obtained from a normal or t-distribution, 
        #the bootstrapped confidence interval is not symmetric about the mean, 
        #which provides an indication of the degree of skewness of the population in question
        simulations = list()
        sample_size = len(my_data)
        # CI around the MEAN
        for c in range(n):
            itersample = np.random.choice(my_data, size=sample_size, replace=True)
            simulations.append(np.mean(itersample))
        simulations.sort()        
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        ci=[simulations[l_indx],simulations[u_indx] ]
        return ci
    def rescale_linear(self,array, new_min, new_max):
        if type(array) == list:
            array = np.asarray(array)
        
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        return m * array + b

    def normality_test(self,z):
        normality1=stats.shapiro(z)        
        if normality1.pvalue<0.05:
            outcome_is_normal=False
        else:
            outcome_is_normal=True
        return outcome_is_normal
    def find_nearest(self,array, values):  
        
        # the last dim must be 1 to broadcast in (array - values) below.
        values = np.expand_dims(values, axis=-1) 
    
        indices = np.abs(array - values).argmin(axis=-1)
    
        return array[indices],indices
    def lin_corr(self,conf_level=0.95,ci = "z-transform"): # INDEP MEASURESES
        vectorx=self.x
        vectory=self.y
        N= 1 - ((1 - conf_level)/2)
        zv = stats.norm.ppf(N, loc = 0, scale = 1)
        k  =len(vectory)
        yb = mean(vectory)
        sy2 = variance(vectory) * (k - 1)/k
        sd1 = stdev(vectory)
        xb = mean(vectorx)
        sx2 = variance(vectorx) * (k - 1)/k
        sd2 = stdev(vectorx)
        #r1 = np.correlate(vectorx, vectory)[0]
        r1=stats.pearsonr(vectorx,vectory)[0]
        sl = r1 * sd1/sd2
        sxy = r1 * sqrt(sx2 * sy2)
        p1 = 2 * sxy/(sx2 + sy2 + (yb - xb)**2)
        delta = vectorx-vectory

        dat=np.column_stack((vectorx,vectory))
        rmean = np.mean(dat, axis=1)
        blalt = np.column_stack(( rmean, delta))

        v = sd1/sd2
        u = (yb - xb)/((sx2 * sy2)**0.25)
        C_b = p1/r1
        sep = sqrt(((1 - (r1**2)) * (p1**2) * (1 - ((p1**2)))/(r1**2) + (2 * (p1**3) * (1 - p1) * (u**2)/r1) - 0.5 * (p1**4) * (u**4)/(r1**2))/(k - 2))
        ll = p1 - (zv * sep)
        ul = p1 + (zv * sep)
        t = np.log((1 + p1)/(1 - p1))/2
        set1 = sep/(1 - (p1**2))
        llt = t - (zv * set1)
        ult = t + (zv * set1)
        llt = (exp(2 * llt) - 1)/(exp(2 * llt) + 1)
        ult = (exp(2 * ult) - 1)/(exp(2 * ult) + 1)
        delta_sd = sqrt(np.var(delta))

        ba_p = mean(delta)
        ba_l =ba_p - (zv * delta_sd)
        ba_u = ba_p + (zv * delta_sd)
        sblalt = {'est' : ba_p, 'delta_sd' : delta_sd, 'lower' : ba_l,  'upper': ba_u}
        if (ci == "asymptotic") :
            rho_c = {"est":p1, "lower":ll,"upper": ul}        
            rval = {"rho" : rho_c, "s_shift" : v, "l_shift" : u,"C_b" : C_b, "blalt" : blalt, "sblalt" : sblalt}

        elif (ci == "z-transform") :
            rho_c ={"est":p1,"lower": llt,"upper": ult}        
            rval = {"rho" : rho_c, "s_shift" : v, "l_shift" : u,"C_b" : C_b, "blalt" : blalt, "sblalt" : sblalt}

        return rho_c #,rval

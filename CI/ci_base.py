import numpy as np
import scipy.stats as stats
from math import sqrt
from statistics import stdev
from scipy.stats.kde import gaussian_kde
class init_ci:
    def __init__(self, x1, y1):
        self.x = x1
        self.y = y1
    def pooled_std(self):
        if len(self.x)==len(self.y):
            return sqrt( ( (stdev(self.x) **2)+(stdev(self.y) **2)   )/2)
        else:
            return sqrt( (((len(self.x) - 1)*(stdev(self.x) **2)) +( (len(self.y)-1)*(stdev(self.y) **2) ) )/ (len(self.x) + len(self.y)-2))
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
        """Rescale an arrary linearly."""
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
    def make_the_eye1(self,my_var):
        x_axis = np.linspace(min(my_var), max(my_var), num=1000,endpoint=True)
        pdf = stats.norm.pdf(x_axis, np.mean(my_var), stdev(my_var))
        return x_axis,pdf
    def make_the_eye2(self,my_var):        
        if len(my_var)<1000:
            dist_space = np.linspace(min(my_var), max(my_var), 1000)
            x_values= np.linspace(min(my_var), max(my_var), len(my_var))
            yinterp = np.interp(dist_space, x_values, my_var)                
            kde = gaussian_kde( yinterp )                
        else:
            kde = gaussian_kde( my_var )
            dist_space = np.linspace( min(my_var), max(my_var), len(my_var) )
            kde(dist_space)
        evaluated = kde.evaluate(dist_space)        
        return dist_space,evaluated
    def by_bootstrap(self,values): # unbiased
        CI=[]
        ci1=0.00001
        ci2=0.99         
        confidence=np.linspace(ci2,ci1,num=1000,endpoint=True)
        for i in confidence:
            outcome=np.percentile(values,[100*(1-i)/2,100*(1-(1-i)/2)],method='normal_unbiased')
            CI.append(outcome)
        return confidence,CI

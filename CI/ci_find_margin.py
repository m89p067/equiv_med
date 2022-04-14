# Help to select DELTA MARGIN using CI permutation
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statistics import stdev,mean
class Id_margin:
    def __init__(self,var_values,confidence_Interval=0.95):
        z,pval = stats.normaltest(var_values)        
        if pval < 0.05:
            print("Not normal distribution")
        else:
            print("Data follows a normal distribution")
            self.values=var_values
        if confidence_Interval > 1:
            confidence_Interval = confidence_Interval / 100.
        if confidence_Interval <= 0 or confidence_Interval>1:
            raise ValueError(f'"confidence_Interval" must be a number in the range 0 to 1, "{confidence_Interval}" provided.')
        else:
            self.v=confidence_Interval
    def percentile_method(self,val1,num_repeats=1000):
        #print('using the sample quantiles to find the bootstrap confidence interval')        
        val1=np.asarray(val1)
        alpha=(1-self.v)/2
        lower_bound=[]
        upper_bound=[]
        for i in range(num_repeats):
            val2=np.random.choice(val1,size=len(val1),replace=True)
            bootstrap_estimate=np.mean(val2)
            val2=np.append(val2,bootstrap_estimate)
            lower_bound.append(np.quantile(val2,alpha))
            upper_bound.append(np.quantile(val2,1-alpha)            )
        return lower_bound,upper_bound
    def boostrap_method(self,val1,num_repeats=1000):
        #print('two-sided bootstrap confidence interval')        
        #val_array=np.vstack([val1]*100)        
        res1 = stats.bootstrap(val1, np.mean,confidence_level=self.v, 
                               n_resamples=num_repeats,method='percentile')
        ci_l, ci_u = res1.confidence_interval
        return ci_l,ci_u
    def t_method(self):
        print('two-sample t-based confidence interval')        
        val1=self.values
        lower_bound,upper_bound=stats.t.interval(alpha=self.v, df=len(val1)-1, 
                                                 loc=np.mean(val1), scale=stats.sem(val1))
        return lower_bound,upper_bound
    def normal_method(self):
        print('confidence Intervals using the Normal Distribution')
        val1=self.values
        lower_bound,upper_bound=stats.norm.interval(alpha=self.v,loc=np.mean(val1), scale=stats.sem(val1))
        return lower_bound,upper_bound
    def decision_perc(self,ideal_margin,noise_variability=1,n_trials=100):        
        var2=self.values
        media=mean(var2)
        st_dev=stdev(var2)        
        ci_lower2=[]
        ci_upper2=[]
        for i in range(n_trials):
            variabile=np.random.normal(loc = media, scale = st_dev, size = (n_trials,))
            noise=np.random.normal(loc = 0, scale = noise_variability, size = (n_trials,))
            variabile2=np.add(variabile,noise)
            ci_lower,ci_upper=self.boostrap_method(variabile2[None])
            ci_lower2.append(ci_lower)
            ci_upper2.append(ci_upper)
        ci_lower2=np.asarray(ci_lower2)
        ci_upper2=np.asarray(ci_upper2)
        ci_ind=np.argsort(ci_lower2)
        ci_lower=ci_lower2[ci_ind]
        ci_upper=ci_upper2[ci_ind]
        ind=0
        fig,ax = plt.subplots()
        for i in range(ci_lower2.shape[0]):
            if ci_lower[i]<=ideal_margin:
                plt.plot([ci_lower[i],ci_upper[i]], [i,i], linestyle='-',color='gray')
                ind +=1
            else:
                plt.plot([ci_lower[i],ci_upper[i]], [i,i], linestyle='-',color='lightgrey')
        plt.axvline(x=ideal_margin, color='red', linestyle='--')    
        plt.ylabel('Series of measurements')
        perc_val=np.where(ci_lower<=ideal_margin,1,0)
        perc_val2=round(np.sum(perc_val)/n_trials,2)*100
        testo=str(perc_val2)+'% Lower C.I.\n are below Margin'
        plt.text(ci_upper[ind],ind,testo, wrap=True)        
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        #plt.savefig('my_plot69.tif', dpi = 300, bbox_inches='tight')
        plt.show()
    def decision_margin(self,ideal_perc,noise_variability=1,n_trials=100):        
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
        var2=self.values
        media=mean(var2)
        st_dev=stdev(var2)        
        ci_lower2=[]
        ci_upper2=[]
        for i in range(n_trials):
            variabile=np.random.normal(loc = media, scale = st_dev, size = (n_trials,))
            noise=np.random.normal(loc = 0, scale = noise_variability, size = (n_trials,))
            variabile2=np.add(variabile,noise)
            ci_lower,ci_upper=self.boostrap_method(variabile2[None])
            ci_lower2.append(ci_lower)
            ci_upper2.append(ci_upper)
        ci_lower2=np.asarray(ci_lower2)
        ci_upper2=np.asarray(ci_upper2)
        ci_ind=np.argsort(ci_lower2)
        ci_lower=ci_lower2[ci_ind]
        ci_upper=ci_upper2[ci_ind]
        ind=0
        ind1=round((ideal_perc*n_trials)/100)
        ideal_margin=ci_lower[ind1]
        fig,ax = plt.subplots()
        for i in range(ci_lower2.shape[0]):
            if ci_lower[i]<=ideal_margin:
                plt.plot([ci_lower[i],ci_upper[i]], [i,i], linestyle='-',color='gray')      
                ind +=1
            else:
                plt.plot([ci_lower[i],ci_upper[i]], [i,i], linestyle='-',color='lightgrey')
        plt.axvline(x=ideal_margin, color='red', linestyle='--')    
        plt.ylabel('Series of measurements')       
        testo=str(ideal_perc)+'% Lower C.I.\n are below Margin of '+str(round(ideal_margin,2))
        plt.text(ci_upper[ind],ind,testo, wrap=True)
        #print('Theoretical margin equal to ',ideal_margin)
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        #plt.savefig('my_plot70.tif', dpi = 300, bbox_inches='tight')
        plt.show()

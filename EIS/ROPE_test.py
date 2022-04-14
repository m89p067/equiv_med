# ROPE
import numpy as np
import scipy.stats as stats
from statistics import stdev,mean
from math import ceil,floor
import matplotlib.pyplot as plt
class ROPE():
    is_normal=True
    
    def __init__(self, x1, rope_range=[-0.1,0.1],credib_interv=0.95):
        print('Test for Practical Equivalence')
        self.x = np.asarray(x1)
        
               
        shapiro_test1 = stats.shapiro(x1)
        
        if shapiro_test1.pvalue>0.05 :
            ROPE.is_normal=True
            print('Normal distribution: HDI and ETI should return same results')
        else:
            ROPE.is_normal=False
            print('Not normal distribution: HDI and ETI might not show same outcomes')
        self.rope_limits=rope_range
        self.cred_int=credib_interv
    def rope_calc(self):
        # equal-tailed interval
        my_data=self.x
        my_range=self.rope_limits
        ci=self.cred_int
        CI_low=np.quantile(my_data,(1 - ci)/2) 
        CI_high=np.quantile(my_data,(1 + ci)/2)
        hdi_area=my_data[np.logical_and(my_data >=CI_low, my_data <= CI_high)]
        area_within=hdi_area[np.logical_and(hdi_area >=min(my_range), hdi_area <= max(my_range) )]
        rope_percentage = len(area_within)/len(hdi_area)
        ROPE_low =my_range[0]
        ROPE_high =my_range[1]
        if ci<1:
            if rope_percentage==1:
                out1="Accepted"
            else:
                out1="Undecided"
            if rope_percentage==0:
                out2="Rejected"
            else:
                out2=out1
        else:
            if rope_percentage>0.975:
                out1="Accepted"
            else:
                out1="Undecided"
            if rope_percentage<  0.025:
                out2="Rejected"
            else:
                out2=out1
        perc=3        
        print('ROPE Region [',min(my_range),' : ',max(my_range),']')
        print('Proportion of samples inside the ROPE Region :',round(rope_percentage*100,perc),' %')
        print('Equivalence based on ROPE :',out2)
        return rope_percentage
    def rope_hdi(self):
        # highest density interval
        my_data=self.x
        my_range=self.rope_limits
        ci=self.cred_int
        x_sorted = np.sort(my_data, kind = "quicksort")
        window_size = ceil(ci * len(x_sorted))
        if window_size<2:
            print('Warning : credibility interval or the dataset too small')
        nCIs=len(x_sorted)-window_size
        if nCIs<1:
            print('Warning : credibility interval too large or the dataset too small')
        for i in range(nCIs):
            ci_width=x_sorted[i+window_size]-x_sorted[i]
        min_i=np.where(ci_width==np.amin(ci_width)        )
        n_minima = len(min_i)
        if n_minima>1:
            if any(np.diff(np.sort(min_i)) != 1):
                print("Found identical values, selecting one...")
                min_i=max(min_i)
            else:
                min_i=floor(mean(min_i))
        CI_low = x_sorted[min_i[0]]        
        CI_high = x_sorted[min_i[0] +    window_size]
        hdi_area=my_data[np.logical_and(my_data >=CI_low, my_data <= CI_high)]
        area_within=hdi_area[np.logical_and(hdi_area >=min(my_range), hdi_area <= max(my_range) )]
        rope_percentage = len(area_within)/len(hdi_area)
        ROPE_low =my_range[0]
        ROPE_high =my_range[1]
        if ci<1:
            if rope_percentage==1:
                out1="Accepted"
            else:
                out1="Undecided"
            if rope_percentage==0:
                out2="Rejected"
            else:
                out2=out1
        else:
            if rope_percentage>0.975:
                out1="Accepted"
            else:
                out1="Undecided"
            if rope_percentage<  0.025:
                out2="Rejected"
            else:
                out2=out1
        perc=3        
        print('ROPE Region [',min(my_range),' : ',max(my_range),']')
        print('Proportion of samples inside the ROPE Region :',round(rope_percentage*100,perc),' %')
        print('Equivalence based on ROPE :',out2)
        return rope_percentage
    def plot_rope(self,which_rope='eti'):
        perc=1
        if which_rope=='eti':
            area_raw=self.rope_calc()
            stringa="equal-tailed interval"
        else:
            area_raw=self.rope_hdi()
            stringa="highest density interval"
        fig, ax = plt.subplots(figsize=(6, 6))
        wedgeprops = {'width':0.3, 'edgecolor':'black', 'linewidth':3}
        area_1=round(area_raw*100,perc)
        ax.pie([area_1,100-area_1], wedgeprops=wedgeprops, startangle=90, colors=['#5DADE2', '#515A5A'])
        testo='Proportion of samples inside\nthe ROPE ['+str(self.rope_limits[0])+','+str(self.rope_limits[1])+']'
        plt.title(testo, fontsize=14, loc='left')
        plt.text(0, 0, str(area_1)+"%", ha='center', va='center', fontsize=42)
        plt.text(-1.2, -1.2, "Region of practical equivalence\nusing "+stringa, ha='left', va='center', fontsize=12)
        #plt.savefig('my_plot800.tif', dpi = 300, bbox_inches='tight')
        plt.show()

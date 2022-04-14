import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from . import eq_base
from statistics import stdev
class ICI_bounds(eq_base.init_eq):
    is_normal=True
    def __init__(self, x1, y1,paired=False):
        self.x = x1
        self.y = y1
        self.is_paired=paired
    
    def check_z(self,v,the_bias,use_t=False):
        if not (v < 99.9) & (v > 1):
            raise ValueError(f'"confidence_Interval" must be a number in the range 1 to 99, "{v}" provided.')
        out=v/100
        if use_t==False:
            coeff_1=stats.norm.ppf((1-out)/2)
        else:
            coeff_1=stats.t.ppf((1-out)/2, df=len(self.x)-1)
        out1=abs(coeff_1)           
        return out1
    def run_ICI(self,imprec1,imprec2,confidence_int=95,bias=0,strict_test=True):
        def chebyshev_inequality(k0):
            return 1 / k0**2
        self.x_imprec=imprec1
        self.y_imprec=imprec2
        if self.is_paired==False:
            comb_inh_impr=np.sqrt(self.x_imprec**2 + self.y_imprec**2 )/100
        else:
            comb_inh_impr=np.sqrt(((self.x_imprec**2)/2) + ((self.y_imprec**2)/2) )/100        
        mean_meas= (self.x+self.y)/2
        diff=(self.y-self.x)
        ICI_bounds.is_normal=self.normality_test(diff)
        minima=min(mean_meas)
        maxima=max(mean_meas)
        mean1=np.linspace(minima-(0.2*minima),maxima+(0.2*maxima),num=1000,endpoint=True)        
        
        M1=[]
        M2=[]
        crit_value=self.check_z(confidence_int,bias)
        
        print('Difference between measurements are normally distributed : ',ICI_bounds.is_normal)
        if ICI_bounds.is_normal:
            for valore1 in mean1:
                M1.append(bias+(crit_value*comb_inh_impr)*valore1)
                M2.append(bias-(crit_value*comb_inh_impr)*valore1)
        else:
            ci_cheb=1-chebyshev_inequality(crit_value)
            tmp=(round(100*ci_cheb)*crit_value)/confidence_int            
            crit_value=tmp            
            for valore1 in mean1:
                M1.append(bias+(crit_value*comb_inh_impr)*valore1)
                M2.append(bias-(crit_value*comb_inh_impr)*valore1)            

        count_good=0
        count_bad=0
        
        for valore,differenza in zip(mean_meas,diff):
            M_1=bias+(crit_value*comb_inh_impr)*valore
            M_2=bias-(crit_value*comb_inh_impr)*valore
            if strict_test:
                if differenza<M_1 and differenza>M_2:
                    count_good += 1
                else:
                    count_bad +=1
            else:
                if differenza<=M_1 and differenza>=M_2:
                    count_good += 1
                else:
                    count_bad +=1                
        perc_good=100*(count_good/len(diff))
        perc_bad=100*(count_bad/len(diff))
        print('\nInherent imprecision acceptance limits:')       
        if ICI_bounds.is_normal:
            print('Number of values falling inside margins :',count_good,'[',round(perc_good,2),'%], Setting '+str(confidence_int)+'% CI')
            colore='orange'
        else:
            print('Applied Chebyshev bounds at '+str(round(100*ci_cheb))+'%, instead of expected '+str(confidence_int)+'% CI')
            print('Number of values falling inside margins :',count_good,'[',round(perc_good,2),'%]')
            colore='firebrick'
        print('Number of values falling outside margins :',count_bad,'[',round(perc_bad,2),'%]')    
        # Should fall in the margin 95% of the measurements
        fig = plt.figure()
        plt.plot(mean_meas,diff,color='k',marker='o',linestyle='None')
        plt.plot([  min(mean1), max(mean1)],[bias,bias],color="gray",linestyle='--')
        plt.plot(mean1,M1,color=colore,linestyle='--')
        plt.plot(mean1,M2,color=colore,linestyle='--')                          
        plt.title('Acceptability based on inherent imprecision ['+str(round(perc_good,1))+'%]')
        plt.xlabel('Mean of both procedures')
        plt.ylabel('Procedure 2 - Procedure 1')
        spazio=abs((maxima+(0.525*maxima))-(maxima+(0.475*maxima)))
        for x_diff in diff:
            plt.plot([maxima+(0.4*maxima)-spazio,maxima+(0.4*maxima)+spazio],[x_diff,x_diff],color='gray',alpha=0.25)
        plt.plot([maxima+(0.4*maxima)-spazio,maxima+(0.4*maxima)+spazio],[np.median(diff),np.median(diff)],color='forestgreen',alpha=0.85)
        plt.plot([maxima+(0.4*maxima)-spazio,maxima+(0.4*maxima)+spazio],[np.mean(diff),np.mean(diff)],color='dodgerblue',alpha=0.85)
        plt.plot([maxima+(0.4*maxima),maxima+(0.4*maxima)],[np.mean(diff)-stdev(diff),np.mean(diff)+stdev(diff)],color='magenta',alpha=0.45)
        plt.xlim(minima-(0.4*minima),maxima+(0.6*maxima))
        locs, labels=plt.xticks()
        indeces=np.where(locs>maxima+(0.4*maxima))
        locs=np.delete(locs, indeces)        
        new_ticks=locs
        plt.xticks(locs,new_ticks)        
        #plt.savefig('my_plot44.tif', dpi = 300, bbox_inches='tight')
        plt.show()


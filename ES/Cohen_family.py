import numpy as np
import scipy.stats as stats
from statistics import stdev,variance
from math import sqrt,lgamma,exp
from scipy.stats.kde import gaussian_kde
from matplotlib import pyplot as plt
from Cohen.find_inters import intersection
from scipy.special import gamma

class Cohen_es:
    is_normal=False
    var_equal=False
    design_type='indep'
    dep_divider=''
    def __init__(self, x1, y1,design='indep',dep_divider_type=''):
        self.x = np.asarray(x1)
        self.y = np.asarray(y1)
        if len(x1)!=len(y1):
            print('The samples should have same size')
            raise ValueError()
        x1_norm=stats.shapiro(x1)
        x2_norm=stats.shapiro(y1)
        if x1_norm.pvalue>0.05 and x2_norm.pvalue>0.05:
            Cohen_es.is_normal=True
        else:
            Cohen_es.is_normal=False
            print('Warning: Input measurements do not follow a normal distribution')
        Cohen_es.design_type=design
        _,check_var=stats.levene(x1,y1,center='mean')
        if check_var<0.05: # unequal variances
            Cohen_es.var_equal=False
            print('Warning: Homogeneity of variances not found')
        else:
            Cohen_es.var_equal=True
        if design=='dep' and dep_divider_type=='':
            print('Using pooled standard deviation also in dependent design')
            print('Otherwise change the dep_divider_type parameter')

        Cohen_es.dep_divider=dep_divider_type
    def dof_cohen(self, dof_method=True,classic=True):
        n1, n2 = len(self.x), len(self.y)
        SD1 = stdev(self.x)
        SD2 = stdev(self.y)
        if Cohen_es.design_type=='indep' and Cohen_es.var_equal==False:
            if dof_method==True :
                # Delacre
                dof_out1=(n1-1)*(n2-1)*(SD1**2+SD2**2)**2
                dof_out2=((n2-1)*(SD1**4))+((n1-1)*(SD2**4))
                dof_out=dof_out1/dof_out2
            else:
                # Shieh
                s1 = (SD1**2)
                s2 = (SD2**2)
                dof1 = ((s1 / n1) +( s2 / n2))**2 
                dof2=( ((s1/n1)**2)/(n1-1)) + (((s2/n2)**2)/(n2-1))
                dof3= dof1/dof2
                dof_out=dof1/(dof2+dof3)
        elif Cohen_es.design_type=='indep' and Cohen_es.var_equal==True:
            dof_out=n1+n2-2
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='sd_diff' and Cohen_es.var_equal==True:            
            if classic:
                dof_out=2*(n1-1)
            else:
                r_value=pearsonr(self.x,self.y)[0]
                dof_out=2/(1+r_value**2)*(N_pairs-1) #(Cousineau, 2020)
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='conv_sd_diff' and Cohen_es.var_equal==True:
            if classic:
                dof_out=2*(n1-1)
            else:
                r_value=pearsonr(self.x,self.y)[0]
                dof_out=2/(1+r_value**2)*(N_pairs-1) #(Cousineau, 2020)
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='' and Cohen_es.var_equal==True:
            dof_out=2*(n1-1)     
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='aver_sd' and Cohen_es.var_equal==True:
            dof_out=2*(n1-1)
        else: # Dependent samples with heterogeneity of variances
            s1 = (SD1**2)
            s2 = (SD2**2)
            dof1=((s1/n1)+(s2/n2))**2
            dof2=((s1/n1)**2)/(n1-1)
            dof3=((s2/n2)**2)/(n2-1)
            dof_out=dof1/(dof2+dof3)        
        return dof_out
    def standard_dev_measure(self):
        #r_value is cross measurement correlation
        if Cohen_es.dep_divider=='conv_sd_diff' or 'sd_diff':
            r_value=stats.pearsonr(self.x,self.y)[0]

        n1, n2 = len(self.x), len(self.y)
        SD1 = stdev(self.x)
        SD2 = stdev(self.y)        
        if Cohen_es.design_type=='indep' and Cohen_es.var_equal==True:
            S_value=sqrt(((n1 - 1)*(SD1**2) + (n2-1)*(SD2**2) ) /   (n1 + n2-2))
        elif Cohen_es.design_type=='indep' and Cohen_es.var_equal==False:
            S_value=sqrt( (SD1**2 + SD2**2)/2)
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='':
            S_value=sqrt(((n1 - 1)*SD1 * SD1 + (n2-1)*SD2 * SD2) /   (n1 + n2-2))
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='sd_diff' :            
            S_value=sqrt( SD1**2+SD2**2-2*SD1*SD2*r_value)
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='conv_sd_diff':
            S_value=sqrt( SD1**2+SD2**2-2*SD1*SD2*r_value)/(2*(1-r_value))
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='aver_sd' and Cohen_es.var_equal==True:
            #ignores any correlation between measures
            S_value=sqrt( (SD1**2 + SD2**2)/2)
        else:
            print('Encountering problems in defining the divider in repreated measures design')
        return S_value
    def bias_correction(self): #Cousineau and Goulet-Pelletier (2021)
        df=self.dof_cohen()
        J = exp ( lgamma(df/2) - np.log(sqrt(df/2)) - lgamma((df-1)/2) )
        return J
    def Cohen_d(self,do_correction=False):
        u1, u2 = np.mean(self.x), np.mean(self.y)      
        df=self.dof_cohen()
        if Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='conv_sd_diff':
            r_value=pearsonr(self.x,self.y)[0]
            S_val=self.standard_dev_measure()
            d_val=(u1-u2)/S_val
            d_val=d_val/(sqrt(2*(1-r_value)))
        else:
            S_val=self.standard_dev_measure()
            d_val=(u1-u2)/S_val
        if df<100: # Correction not required if df>=100
           do_correction=True # Automatically set the correction if df<100
           print('Automatical Hedges correction for unbiased Cohen d')
        if do_correction==True: # Hedges            
            J = self.bias_correction()
            d_val=d_val*J
        return d_val
    def harmonic_mean_cohen(self,value1,value2):
        return (2*value1*value2)/(value1+value2)
    def standard_error_cohen(self):
        d_value=self.Cohen_d()
        df=self.dof_cohen()
        n1, n2 = len(self.x), len(self.y)
        n_hm=self.harmonic_mean_cohen(n1,n2)
        r_value=stats.pearsonr(self.x,self.y)[0]
        if Cohen_es.design_type=='indep' and Cohen_es.var_equal==True:
            sigma1=(df/(df-2))*(2/n_hm)*(1+((d_value**2)*(n_hm/2)))
            sigma=sigma1-((d_value**2)/self.bias_correction())
            sigma=sqrt(sigma)
        elif Cohen_es.design_type=='indep' and Cohen_es.var_equal==False:
            sigma1=(df/(df-2))*(2/n_hm)*(1+((d_value**2)*(n_hm/2)))
            sigma=sigma1-((d_value**2)/(self.bias_correction()**2))
            sigma=sqrt(sigma)
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='sd_diff' :
            sigma1=(df/(df-2))*((2*(1-r_value))/n1)*(1+((d_value**2)*(n/(2*(1-r_value)))))
            sigma=sigma1-((d_value**2)/(self.bias_correction()**2))
            sigma=sqrt(sigma)
            sigma=sigma*sqrt(2*(1-r_value))
        elif Cohen_es.design_type=='dep' and Cohen_es.dep_divider=='conv_sd_diff':
            sigma1=(df/(df-2))*((2*(1-r_value))/n1)*(1+((d_value**2)*(n/(2*(1-r_value)))))
            sigma=sigma1-((d_value**2)/(self.bias_correction()**2))
            sigma=sqrt(sigma)
        else:
            sigma=self.standard_dev_measure()/n1
            #sigma=(self.standard_dev_measure()/n1)*sqrt(1-r_value) # correlation adjusted by Cousineau
        return sigma
    def lambda_par(self,classic=False):
        d_value=self.Cohen_d()
        n1, n2 = len(self.x), len(self.y)
        r_value=stats.pearsonr(self.x,self.y)[0]
        SE_val=self.standard_error_cohen()
        if Cohen_es.design_type=='indep' and Cohen_es.var_equal==False:
            if classic:               
                out_lambda=d_value*(1/sqrt((1/n1)+(1/n2))) #Chen
            else:
                part1=n1*n2*(stdev(self.x)**2  +stdev(self.y)**2 )            
                part2=(n2*stdev(self.x)**2)+(n1*stdev(self.y)**2)         
                out_lambda=sqrt(part1/(2*part2))*d_value
        elif Cohen_es.design_type=='indep' and Cohen_es.var_equal==True:
            harm_mean=self.harmonic_mean_cohen(n1,n2)
            out_lambda=d_value*sqrt(harm_mean/2)           
        else: # Dep design
            out_lambda=d_value*sqrt(n1/(2*(1-r_value)))
        return out_lambda
    def _measures_of_nonoverlap(self):
        d=self.Cohen_d()
        if Cohen_es.is_normal==True and Cohen_es.var_equal==True:            
            U3=stats.norm.cdf(d,loc=0,scale=1)
            U2=stats.norm.cdf(d/2,loc=0,scale=1)
            U2X=stats.norm.cdf(np.abs(d)/2,loc=0,scale=1)
            U1=(2*U2X-1)/U2X
            out_nonoverlap=[U1,U2,U3]
        else:
            out_nonoverlap=[np.nan, np.nan, np.nan]
        return out_nonoverlap
    def CI_cohen(self,ci_type='non_central',required_ci=0.95):
        def qt(p,df,ncp=0):
            if ncp==0:
                result=stats.t.ppf(q=p,df=df,loc=0,scale=1)
            else:
                result=stats.nct.ppf(q=p,df=df,nc=ncp,loc=0,scale=1)
            return result
        cohen_d=self.Cohen_d()
        lambda1=self.lambda_par()
        d_df=self.dof_cohen()
        if ci_type=='non_central':                       
            # Non central method [Goulet-Pelleter and Cousineau]
            alpha=1-required_ci
            lower_lim=qt(1/2 - (1 - alpha)/2,  d_df,  lambda1)
            high_lim=qt(1/2 + (1 - alpha)/2,  d_df, lambda1)
            dlow = lower_lim/(lambda1 / cohen_d)
            dhigh = high_lim/(lambda1 / cohen_d)
        elif ci_type=='central':
            dlow = cohen_d-(self.std_error_cohen()*qt(1 - required_ci/2,  d_df))
            dhigh = cohen_d +(self.std_error_cohen()*qt(1 - required_ci/2,  d_df) )
        elif ci_type=='boot':
            print('Not implemented yet')        
        return dlow,dhigh   
    def nonoverlap_plotting(self):
        fig, ax = plt.subplots()
        N = 3
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence
        op_list=np.array(self._measures_of_nonoverlap())*100
        p1 = plt.bar(ind, op_list, width,color='royalblue')
        p2 = plt.bar(ind, 100-op_list, width,bottom=op_list,color='lightgrey')
        ax.bar_label(p1, padding=3,fmt='%.2f')
        plt.ylabel('%')
        plt.title('Measures of nonoverlap')
        plt.xticks(ind, ('U1', 'U2', 'U3'))
        plt.yticks(np.arange(0, 110, 10))        
        #plt.savefig('my_plot59.tif', dpi = 300, bbox_inches='tight')
        plt.show()
    def nonoverlap_measures(self):
        op_list=self._measures_of_nonoverlap()
        if np.isnan(op_list).all()==False:
            print('Measures of nonoverlap')
            print('U1 aka % of nonoverlap between 2 distributions')
            print('U1 :',op_list[0]*100,' %')
            print('U2 aka highest % in measurement 1 that exceds the same lowest % in measurement 2')
            print('U2 :',op_list[1]*100,' %')
            print('U3 aka percentile standing')
            print('U3 :',op_list[2]*100,' %')
            self.nonoverlap_plotting()
        else:
            print('Nonoverlap measurement discouraged')
    def plotting(self):
        font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 10
        }
        d_df=self.dof_cohen()
        kde1 = gaussian_kde( self.x )
        kde2 = gaussian_kde( self.y )
        dist_space1 = np.linspace( min(self.x), max(self.x), 1000 )
        dist_space2 = np.linspace( min(self.y), max(self.y), 1000 )
        plt.plot( dist_space1, kde1(dist_space1),label='new' ,color='blue')
        plt.plot( dist_space2, kde2(dist_space2),label='ref',color='red' )
        prima_curva=kde1(dist_space1)
        seconda_curva=kde2(dist_space2)
        if max(self.x)>max(self.y):
            massimo=max(self.x)
        else:
            massimo=max(self.y)
        if min(self.x)<min(self.y):
            minimo=min(self.x)
        else:
            minimo=min(self.y)
        if max(self.x)>max(self.y):
            massimo2=max(self.y)
        else:
            massimo2=max(self.x)
        if min(self.x)<min(self.y):
            minimo2=min(self.y)
        else:
            minimo2=min(self.x)
        #x_val = np.linspace( minimo, massimo, 1000 )
        
        plt.fill_between(dist_space1 , kde1(dist_space1 ), color='green',alpha=0.1)
        plt.fill_between(dist_space2 , kde2(dist_space2 ), color='green',alpha=0.1)
        plt.axvline(x=np.mean(self.x), color='blue', linestyle='--')
        plt.axvline(x=np.mean(self.y), color='red', linestyle='--')
        
        ci_low,ci_high=self.CI_cohen()
        if d_df<100:
            plt.title('Hedges g '+str(np.round(self.Cohen_d(),3)))
        else:
            plt.title('Cohen d '+str(np.round(self.Cohen_d(),3)))
        plt.ylabel('Probability density')
        x_inters, y_inters = intersection(dist_space1, kde1(dist_space1),dist_space2, kde2(dist_space2))
        plt.plot(x_inters, y_inters, ".k", markersize=6)        
        if len(x_inters)>0 and len(y_inters)>0: # THERE IS INTERSECTION POINT
            x_val2 = np.linspace( minimo2, massimo2, 1000 )
            y_val = np.minimum(kde1(x_val2), kde2(x_val2))
            plt.fill_between(x_val2, y_val, color='yellow',alpha=0.75) 
            area1=np.trapz(kde1(dist_space1),dist_space1)
            area2=np.trapz(kde2(dist_space2),dist_space2)        
            minimi=[min(dist_space1),min(dist_space2)]
            massimi=[max(dist_space1),max(dist_space2)]
            ind1=minimi.index(max(minimi))
            ind2=massimi.index(min(massimi))
            #my_array=np.column_stack((dist_space1, dist_space2))
            if ind1==0 and ind2==1  :
                area3=np.trapz(kde1(dist_space1[dist_space1<=x_inters]),dist_space1[dist_space1<=x_inters])
                area4=np.trapz(kde2(dist_space2[dist_space2>x_inters]),dist_space2[dist_space2>x_inters])
            else:
                area3=np.trapz(kde2(dist_space2[dist_space2<=x_inters]),dist_space2[dist_space2<=x_inters])
                area4=np.trapz(kde1(dist_space1[dist_space1>x_inters]),dist_space1[dist_space1>x_inters])
            overlap_area=(area3+area4)/(area1+area2)            
            plt.text(x_inters-(x_inters*0.2), y_inters/2, str(round( 100*overlap_area,2) )+' %', fontdict=font)
        plt.axvline(x=self.Cohen_d(), color='green', linestyle=':')
        y_limits=plt.ylim()
        val_y=max(y_limits)
        #mass1=max(prima_curva)
        #mass2=max(seconda_curva)
        #mass_ass=max([mass1,mass2])
        #valore_spazio=(val_y-mass_ass)/2
        plt.axvline(x=ci_low, color='green', linestyle=':',alpha=0.3)
        plt.axvline(x=ci_high, color='green', linestyle=':',alpha=0.3)
        plt.ylim(0.0,val_y)
        #plt.savefig('my_plot60.tif', dpi = 300, bbox_inches='tight')
        plt.show()
            
            
                
        
    
        
        

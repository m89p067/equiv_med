import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from statistics import stdev
from . import ci_base
# CAT-EYE plot

class Cat_Eye_2var(ci_base.init_ci):
    is_normal=True
    def __init__(self, x, y):
        super().__init__(x, y)
        normality1=stats.shapiro(self.x)    
        normality2=stats.shapiro(self.y)
        if normality1.pvalue>0.05 and normality2.pvalue>0.05:
            Cat_Eye_2var.is_normal=True
        else:
            Cat_Eye_2var.is_normal=False  
    def get_CI(self,my_var):
        ci_out=[]        
        alpha1=0.00001
        alpha2=0.99999        
        p_values=np.linspace(alpha2,alpha1,num=1000,endpoint=True)
        for i in p_values:
            if Cat_Eye_2var.is_normal==True:
                # create confidence interval for population
                #ci_out.append(stats.norm.interval(i, loc=np.mean(a), scale=a.std(ddof=1) ) )
                # create confidence interval for the sample
                #ci_out.append(stats.norm.interval(i, loc=np.mean(a), scale=stats.sem(a) ) )
                ci_out.append(stats.norm.interval(i, loc=np.mean(my_var), scale=stdev(my_var) ) )
            else:
                print('Not normal, sorry')
        return p_values,ci_out
    def get_Z(self,CIlev):
        if CIlev>1:
            CIlev=CIlev/100
        alpha=1-CIlev #0.95        
        out1=stats.norm.ppf(1-(alpha/2))
        out2=stats.norm.ppf(alpha/2)
        return out1,out2
    def control_of_CI_bounds(self,my_values):
        low_of_a=my_values[0]
        high_of_a=my_values[1]
        low_of_b=my_values[2]
        high_of_b=my_values[3]
        print('\nConfidence Interval inclusion check:')
        if low_of_b<low_of_a and high_of_a<high_of_b: #Open intervals are defined as those which don’t include their endpoints.
            sentence='The first variable C.I. are entirely inside the C.I. of the second variable (open interval)'
        elif low_of_a<low_of_b and high_of_b<high_of_a: #Open intervals are defined as those which don’t include their endpoints.
            sentence='The second variable C.I. are entirely inside the C.I. of the first variable (open interval)'
        elif low_of_b<=low_of_a and high_of_a<=high_of_b:
            sentence='The first variable C.I. are contained by the C.I. of the second variable (close interval)'
        elif low_of_a<=low_of_b and high_of_b<=high_of_a:
            sentence='The second variable C.I. are contained by the C.I. of the first variable (close interval)'
        else:
            sentence='Nothing significant'
        return sentence
    def run_ce(self,level_CI_as_perc):
        if Cat_Eye_2var.is_normal:
            while( level_CI_as_perc<1 or level_CI_as_perc>99):
                print('C.I. valid range between 1 and 99')
                level_CI_as_perc=input('Please insert a value between 1 and 99 : ')
                level_CI_as_perc=float(level_CI_as_perc)
            
            fig,ax = plt.subplots()
            a=self.x
            p_y_values,val_ci=self.get_CI(a)
            se1 = stats.sem(a)            
            color_a='skyblue' 
            color_a2='royalblue'               
            plt.axvline(x=np.mean(a), color=color_a2, linestyle='--')
            level_CI,eye_height=2,1
            x_values,pdf1=self.make_the_eye1(a)
            pdf1_rescaled1=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI+eye_height))
            pdf1_rescaled2=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI-eye_height))
            plt.plot( x_values,pdf1_rescaled1 ,color=color_a,alpha=0.3,linewidth=2)
            plt.plot( x_values,pdf1_rescaled2 ,color=color_a,alpha=0.3,linewidth=2)            
            std_lim,_=self.get_Z(level_CI_as_perc)            
            low_a = np.mean(a)-std_lim*stdev(a)
            high_a = np.mean(a)+std_lim*stdev(a)
            plt.fill_between(x_values,pdf1_rescaled2, pdf1_rescaled1, where=(low_a < x_values) & (x_values < high_a),
                             interpolate=True,color=color_a,alpha=0.2)
            plt.plot(np.mean(a),level_CI, 'o',color=color_a2)
            plt.plot([low_a,high_a],[level_CI,level_CI], '|',color=color_a2, linestyle="--")

            
            b=self.y
            p_y_values,val_ci=self.get_CI(b)
            se2 = stats.sem(b)            
            color_b='lightcoral'
            color_b2='red'
            plt.axvline(x=np.mean(b), color=color_b2, linestyle='--')
            level_CI,eye_height=5,1
            x_values,pdf1=self.make_the_eye1(b)
            pdf1_rescaled1=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI+eye_height))
            pdf1_rescaled2=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI-eye_height))
            plt.plot( x_values,pdf1_rescaled1 ,color=color_b,alpha=0.3,linewidth=2)
            plt.plot( x_values,pdf1_rescaled2 ,color=color_b,alpha=0.3,linewidth=2)
            std_lim,_=self.get_Z(level_CI_as_perc)            
            low_b = np.mean(b)-std_lim*stdev(b)
            high_b = np.mean(b)+std_lim*stdev(b)
            plt.fill_between(x_values,pdf1_rescaled2, pdf1_rescaled1, where=(low_b < x_values) & (x_values < high_b),
                             interpolate=True,color=color_b,alpha=0.2)
            plt.plot(np.mean(b),level_CI, 'o',color=color_b2)
            plt.plot([low_b,high_b],[level_CI,level_CI], '|',color=color_b2, linestyle="--")
            
            
            #plt.plot([np.mean(a)-se1,np.mean(a)+se1],[6.8,6.8], color=color_a2, linestyle="-")
            #plt.plot([np.mean(b)-se2,np.mean(b)+se2],[7.2,7.2], color=color_b2, linestyle="-")
            
            plt.yticks([2,5],["Var.1","Var. 2"])        

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            
            the_alpha=round(1-(level_CI_as_perc/100) ,3)
            plt.title('Variables C.I. at '+str(level_CI_as_perc)+' %'+' [p '+str(the_alpha )+']')
            
            outcome_sentence=self.control_of_CI_bounds([low_a,high_a,low_b,high_b])
            print('At ',the_alpha*100,'% probability : ',outcome_sentence)
            #plt.savefig('my_plot2.tif', dpi = 300, bbox_inches='tight')
    def run_ce_unbiased(self,level_CI_as_perc):
        if Cat_Eye_2var.is_normal:
            while( level_CI_as_perc<1 or level_CI_as_perc>99):
                print('C.I. valid range between 1 and 99')
                level_CI_as_perc=input('Please insert a value between 1 and 99 : ')
                level_CI_as_perc=float(level_CI_as_perc)
            
            fig,ax = plt.subplots()
            a=self.x
            p_y_values,val_ci=self.get_CI(a)
            se1 = stats.sem(a)            
            color_a='skyblue' 
            color_a2='royalblue'               
            plt.axvline(x=np.mean(a), color=color_a2, linestyle='--')
            level_CI,eye_height=2,1
            x_values,pdf1=self.make_the_eye2(a)
            pdf1_rescaled1=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI+eye_height))
            pdf1_rescaled2=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI-eye_height))
            plt.plot( x_values,pdf1_rescaled1 ,color=color_a,alpha=0.3,linewidth=2)
            plt.plot( x_values,pdf1_rescaled2 ,color=color_a,alpha=0.3,linewidth=2)            
            std_lim,_=self.get_Z(level_CI_as_perc)            
            low_a = np.mean(a)-std_lim*stdev(a)
            high_a = np.mean(a)+std_lim*stdev(a)
            plt.fill_between(x_values,pdf1_rescaled2, pdf1_rescaled1, where=(low_a < x_values) & (x_values < high_a),
                             interpolate=True,color=color_a,alpha=0.2)
            plt.plot(np.mean(a),level_CI, 'o',color=color_a2)
            plt.plot([low_a,high_a],[level_CI,level_CI], '|',color=color_a2, linestyle="--")

            
            b=self.y
            p_y_values,val_ci=self.get_CI(b)
            se2 = stats.sem(b)                
            color_b='lightcoral'
            color_b2='red'
            plt.axvline(x=np.mean(b), color=color_b2, linestyle='--')
            level_CI,eye_height=5,1
            x_values,pdf1=self.make_the_eye2(b)
            pdf1_rescaled1=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI+eye_height))
            pdf1_rescaled2=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI-eye_height))
            plt.plot( x_values,pdf1_rescaled1 ,color=color_b,alpha=0.3,linewidth=2)
            plt.plot( x_values,pdf1_rescaled2 ,color=color_b,alpha=0.3,linewidth=2)
            std_lim,_=self.get_Z(level_CI_as_perc)            
            low_b = np.mean(b)-std_lim*stdev(b)
            high_b = np.mean(b)+std_lim*stdev(b)
            plt.fill_between(x_values,pdf1_rescaled2, pdf1_rescaled1, where=(low_b < x_values) & (x_values < high_b),
                             interpolate=True,color=color_b,alpha=0.2)
            plt.plot(np.mean(b),level_CI, 'o',color=color_b2)
            plt.plot([low_b,high_b],[level_CI,level_CI], '|',color=color_b2, linestyle="--")
            
            
            #plt.plot([np.mean(a)-se1,np.mean(a)+se1],[6.8,6.8], color=color_a2, linestyle="-")
            #plt.plot([np.mean(b)-se2,np.mean(b)+se2],[7.2,7.2], color=color_b2, linestyle="-")
            
            plt.yticks([2,5],["Var.1","Var. 2"])        

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            
            the_alpha=round(1-(level_CI_as_perc/100) ,3)
            plt.title('Variables C.I. at '+str(level_CI_as_perc)+' %'+' [p '+str(the_alpha )+']')
            plt.xlabel('')
            outcome_sentence=self.control_of_CI_bounds([low_a,high_a,low_b,high_b])
            print('At ',the_alpha*100,'% probability : ',outcome_sentence)
            #plt.savefig('my_plot3.tif', dpi = 300, bbox_inches='tight')

    def single_cat_eye(self,a,level_CI=95):
        font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 10
        }
        if Cat_Eye_2var.is_normal:
            while( level_CI<1 or level_CI>99):
                print('C.I. valid range between 1 and 99')
                level_CI=input('Please insert a value between 1 and 99 : ')
                level_CI=float(level_CI)
            fig,ax = plt.subplots()
            p_y_values,val_ci=self.get_CI(a)            
            se = stdev(a /np.sqrt(len(a)))
            res = [[ i for i, j in val_ci ], [ j for i, j in val_ci ]]               
            plt.plot(res[0],p_y_values*100,color='r')
            plt.plot(res[1],p_y_values*100,color='r')
            plt.gca().invert_yaxis()
            plt.axvline(x=np.mean(a), color='grey', linestyle='--')
            eye_height=10
            x_values,pdf1=self.make_the_eye1(a)
            pdf1_rescaled1=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI+eye_height))
            pdf1_rescaled2=np.interp(pdf1, (pdf1.min(), pdf1.max()), (level_CI, level_CI-eye_height))
            plt.plot( x_values,pdf1_rescaled1 ,color='gray',alpha=0.3,linewidth=2)
            plt.plot( x_values,pdf1_rescaled2 ,color='gray',alpha=0.3,linewidth=2)
            std_lim,_=self.get_Z(level_CI)            
            low = np.mean(a)-std_lim*stdev(a)
            high = np.mean(a)+std_lim*stdev(a)
            plt.fill_between(x_values,pdf1_rescaled2, pdf1_rescaled1, where=(low < x_values) & (x_values < high),
                             interpolate=True,color='grey',alpha=0.2)
            plt.plot(np.mean(a),level_CI, 'o',color='black')
            plt.plot([low,high],[level_CI,level_CI], '|',color='black', linestyle="--")
            plt.xlabel('Values of the variable')
            plt.ylabel('%')
            plt.text(min(x_values), level_CI-(eye_height/2), str(level_CI) +' %', fontdict=font)
            plt.text(max(x_values), level_CI-(eye_height/2), 'p '+str(round(1-(level_CI/100) ,3) ), fontdict=font)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #plt.savefig('my_plot30.tif', dpi = 300, bbox_inches='tight')

    def single_cat_eye_unbiased(self,a,level_CI=95):
        font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 10
        }
        if Cat_Eye_2var.is_normal:   
            while( level_CI<1 or level_CI>99):
                print('C.I. valid range between 1 and 99')
                level_CI=input('Please insert a value between 1 and 99 : ')
                level_CI=float(level_CI)
            se = stdev(a /np.sqrt(len(a)))
            p_y_values,val_ci=self.by_bootstrap(a)            
            res = [[ i for i, j in val_ci ], [ j for i, j in val_ci ]]
            fig,ax = plt.subplots()
            plt.gca().invert_yaxis()
            plt.plot(res[0],p_y_values*100,color='r')
            plt.plot(res[1],p_y_values*100,color='r')
            plt.axvline(x=np.mean(a), color='grey', linestyle='--')            
            x_values,pdf2=self.make_the_eye2(a)
            eye_height=10
            pdf2_rescaled1=np.interp(pdf2, (pdf2.min(), pdf2.max()), (level_CI, level_CI+eye_height))  
            pdf2_rescaled2=np.interp(pdf2, (pdf2.min(), pdf2.max()), (level_CI, level_CI-eye_height))  
            plt.plot( x_values, pdf2_rescaled1 ,color='gray',alpha=0.3,linewidth=2)
            plt.plot( x_values, pdf2_rescaled2 ,color='gray',alpha=0.3,linewidth=2)
            std_lim,_=self.get_Z(level_CI)            
            low = np.mean(a)-std_lim*stdev(a)
            high = np.mean(a)+std_lim*stdev(a)
            plt.fill_between(x_values,pdf2_rescaled2, pdf2_rescaled1, where=(low < x_values) & (x_values < high),
                             interpolate=True,color='grey',alpha=0.2)
            plt.plot(np.mean(a),level_CI, 'o',color='black')
            #plt.plot(np.median(self.x),level_CI, 'x',color='black')
            plt.plot([low,high],[level_CI,level_CI], '|',color='black', linestyle="--")
            plt.xlabel('Values of the variable')
            plt.ylabel('%')
            plt.text(min(x_values), level_CI-(eye_height/2), str(level_CI) +' %', fontdict=font)
            plt.text(max(x_values), level_CI-(eye_height/2), 'p '+str(round(1-(level_CI/100) ,3) ), fontdict=font)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #plt.savefig('my_plot31.tif', dpi = 300, bbox_inches='tight')

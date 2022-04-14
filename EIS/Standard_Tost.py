import numpy as np
import scipy.stats as stats
from statistics import stdev

def qt(p1,df1,ncp=0):
       
    if ncp==0:
        result=stats.t.ppf(q=p1,df=df1,loc=0,scale=1)
    else:
        result=stats.nct.ppf(q=p1,df=df1,nc=ncp,loc=0,scale=1)
    return result
def pt(q1,df1,ncp=0):
      
    if ncp==0:
        result=stats.t.cdf(x=q1,df=df1,loc=0,scale=1)
    else:
        result=stats.nct.cdf(x=q1,df=df1,nc=ncp,loc=0,scale=1)
    return result
class EQU():
    is_normal=True
    same_var=True
    def __init__(self, x1, y1,low_margin,upper_margin):
        print('Testing hypothesis of equivalence')
        self.x = np.asarray(x1)
        self.y = np.asarray(y1)
        self.Lmargin=low_margin
        self.Umargin=upper_margin

        _, p = stats.levene(x1,y1)
        if p<0.05:
            print('Input measurements do not have equal variances')
            EQU.same_var=False
            self.test_type="Welch Modified Two-Sample t-Test"
        else:
            self.test_type="Standard Two-Sample t-Test"
        shapiro_test1 = stats.shapiro(x1)
        shapiro_test2 = stats.shapiro(y1)
        if shapiro_test1.pvalue>0.05 and shapiro_test2.pvalue>0.05:
            EQU.is_normal=True
            
        else:
            EQU.is_normal=False            
            

    def _indep_calc(self,conf_level = 0.95,mu=0,side='two-sided'):
        alpha = 1 - conf_level
        m1=np.mean(self.x)
        m2=np.mean(self.y)
        sd1=stdev(self.x)
        sd2=stdev(self.y)
        n1=len(self.x)
        n2=len(self.y)
        xbar=m1-m2
        var1=sd1**2
        var2=sd2**2
        if EQU.same_var==True:
            stderr=np.sqrt((((n1 - 1) * var1 + (n2 - 1) * var2) * (1/n1 + 1/n2))/(n1 + n2 - 2))
            dof=n1 + n2 - 2
        else:
            stderr=np.sqrt((var1/n1) + (var2/n2))
            part1=(var1/n1+var2/n2)**2
            part2=((var1/n1)**2)*(1/(n1-1))
            part3=((var2/n2)**2)*(1/(n2-1))
            dof=part1/(part2+part3) # Satterwaite
            dof=int(dof)
        statistic =(xbar-mu)/stderr
        if side=='two-sided':
            hw=qt((1 - alpha/2),dof)*stderr
            p_value=2*qt(-abs(statistic),dof)
            conf_int=[xbar-hw,xbar+hw]
        elif side=='greater':
            p_value=1-pt(statistic,dof)
            conf_int=[xbar-qt((1-alpha),dof)*stderr,np.nan]
        elif side=='less':
            p_value=pt(statistic,dof)
            conf_int=[np.nan,xbar+qt((1-alpha),dof)*stderr]
        return [statistic,p_value,conf_int]
    def run_Tost_indep(self,alpha=0.05 ):
        m1=np.mean(self.x)
        m2=np.mean(self.y)
        sd1=stdev(self.x)
        sd2=stdev(self.y)
        n1=len(self.x)
        n2=len(self.y)
        out_indep1=self._indep_calc(conf_level =1 - alpha * 2,mu=self.Lmargin,side="greater")
        out_indep2=self._indep_calc(conf_level =1 - alpha * 2,mu=self.Umargin,side="less")
        null_hyp = str(round(self.Lmargin, 2))+ " >= (Mean1 - Mean2) or (Mean1 - Mean2) >= "+str(   round(self.Umargin, 2))
        alt_hyp =str(round(self.Lmargin, 2))+ " < (Mean1 - Mean2) < "+str(   round(self.Umargin, 2))

        p=max(out_indep1[1],out_indep2[1])
        if abs(out_indep1[0])<abs(out_indep2[0]):
            t=out_indep1[0]
        else:
            t=out_indep2[0]
        
        print(self.test_type)
        if p<alpha:
            print("Equivalence was significant")
            print('[\u03B1 =',alpha,'] p value :',p,' t statistic:',t)
            print(alt_hyp)
        else:
            print("Equivalence was not significant")
            print('[\u03B1 =',alpha,'] p value :',p,' t statistic:',t)
            print(null_hyp)
        
        
             


    def _dep_calc(self,conf_level = 0.95,mu=0,side='two-sided'):
        alpha = 1 - conf_level
        m1=np.mean(self.x)
        m2=np.mean(self.y)
        sd1=stdev(self.x)
        sd2=stdev(self.y)
        n1=len(self.x)
        n2=len(self.y)
        xbar=m1-m2
        var1=sd1**2
        var2=sd2**2
        if n1 != n2:
            n1 = min(n1, n2)
        if EQU.is_normal==True:
            r12=stats.pearsonr(self.x,self.y)[0]
        else:
            r12=stats.spearmanr(self.x,self.y)[0]

        sd_diff = np.sqrt(sd1**2 + sd2**2 - 2 * r12 * sd1 * sd2)
        stderr=np.sqrt(sd_diff**2/n1)
        dof=n1-1 
        statistic =(xbar-mu)/stderr
        if side=='two-sided':
            hw=qt((1 - alpha/2),dof)*stderr
            p_value=2*qt(-abs(statistic),dof)
            conf_int=[xbar-hw,xbar+hw]
        elif side=='greater':
            p_value=1-pt(statistic,dof)
            conf_int=[xbar-qt((1-alpha),dof)*stderr,np.nan]
        elif side=='less':
            p_value=pt(statistic,dof)
            conf_int=[np.nan,xbar+qt((1-alpha),dof)*stderr]
        return [statistic,p_value,conf_int]
    def run_Tost_dep(self,alpha=0.05 ):
        m1=np.mean(self.x)
        m2=np.mean(self.y)
        sd1=stdev(self.x)
        sd2=stdev(self.y)
        n1=len(self.x)
        n2=len(self.y)
        out_dep1=self._dep_calc(conf_level =1 - alpha * 2,mu=self.Lmargin,side="greater")
        out_dep2=self._dep_calc(conf_level =1 - alpha * 2,mu=self.Umargin,side="less")
        null_hyp = str(round(self.Lmargin, 2))+ " >= (Mean1 - Mean2) or (Mean1 - Mean2) >= "+str(   round(self.Umargin, 2))
        alt_hyp =str(round(self.Lmargin, 2))+ " < (Mean1 - Mean2) < "+str(   round(self.Umargin, 2))

        p=max(out_dep1[1],out_dep2[1])
        if abs(out_dep1[0])<abs(out_dep2[0]):
            t=out_dep1[0]
        else:
            t=out_dep2[0]
        
        print("Paired samples")
        if p<alpha:
            print("Equivalence was significant")
            print('[\u03B1 =',alpha,'] p value :',p,' t statistic:',t)
            print(alt_hyp)
        else:
            print("Equivalence was not significant")
            print('[\u03B1 =',alpha,'] p value :',p,' t statistic:',t)
            print(null_hyp)        
            
                                                                                    

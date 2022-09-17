import numpy as np
from math import sqrt
import scipy.stats as stats
from statistics import stdev,mean

class WS_eq(): #Welch_Satterthwaite
    is_normal=True
    same_var=True
    def __init__(self, x1, y1,delta1):
        ''' 
        Alternative TOST in case of heteroskedasticity in input series
        Args:
        x1,y1   laboratory measurements
        delta1 regulatory boundary assumed symmetric around zero (aka [-𝛿;𝛿]
        '''
        self.x = np.asarray(x1)
        self.y = np.asarray(y1)
        self.margin=delta1
        self._alpha=0.05
        stat, p = stats.levene(x1,y1)
        if p<0.05:
            print('Input measurements do not have equal variances')
            WS_eq.same_var=False
        shapiro_test1 = stats.shapiro(x1)
        shapiro_test2 = stats.shapiro(y1)
        if shapiro_test1.pvalue>0.05 and shapiro_test2.pvalue>0.05:
            WS_eq.is_normal=True            
        else:
            WS_eq.is_normal=False
            print('Inputs are not normally distributed')
            print('The procedure is discouraged')
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, v):
        if not (v > 0) :
            raise ValueError(f'"Alpha" nominal Type I error rate must be a number above 0, "{v}" provided.')
        elif v>1:
            raise ValueError(f'"Alpha" nominal Type I error rate must be a number below 1, "{v}" provided.')
        self._alpha = v
    def run_TOST(self):
        
        n1=len(self.x)
        n2=len(self.y)
        alpha=self._alpha #type I error rate
        del1=self.margin #equivalence bound
        muh1=mean(self.x) #sample means
        muh2=mean(self.y)
        sigmah1=stdev(self.x) #sample standard deviations
        sigmah2=stdev(self.y)
        
        if WS_eq.is_normal:
            mudh=muh1-muh2
            sigsqh1=sigmah1**2
            sigsqh2=sigmah2**2
            sigsqdh=sigsqh1/n1+sigsqh2/n2
            sigmadh=sqrt(sigsqdh)
            df1=n1-1
            df2=n2-1
            w1=(muh1-muh2+del1)/sigmadh
            w2=(muh1-muh2-del1)/sigmadh
            dft=df1+df2
            t2ta=stats.t.ppf(1-alpha,dft)
            qh1=(sigsqh1/n1)/sigsqdh
            qh2=1-qh1
            dfv=1/(qh1*qh1/df1+qh2*qh2/df2)
            tdfv=stats.t.ppf(1-alpha,dfv)
            ws=(w1>tdfv)*(w2<(-tdfv))
            if (ws==1) :
                
                print('Reject the NULL hypothesis: CI lie inside the margins of equivalence')
            else :
                
                print('Keep the NULL hypothesis: CI lie outside the margins of equivalence')
        else:
            print('Violation of normalcy assumption')
    def power_TOST(self,prec=4):
        
        n1=len(self.x)
        n2=len(self.y)
        alpha=self._alpha #type I error rate
        del1=self.margin #equivalence bound
        mu1=mean(self.x) #sample means
        mu2=mean(self.y)
        sigma1=stdev(self.x) #sample standard deviations
        sigma2=stdev(self.y)
        
        if WS_eq.is_normal:
            mud=mu1-mu2
            sigsq1=sigma1**2
            sigsq2=sigma2**2
            numintb=100
            lb=numintb+1
            dd=1e-10
            c_array=np.array([4,2])
            temp=np.tile(c_array,int(numintb/2-1))        
            coevecb=np.concatenate([[1],temp,[4,1]])
            intlb=(1-dd-dd)/numintb
            bvec=dd+intlb*np.asarray(range(numintb+1))
            numintc=5000
            lc=numintc+1
            cl=1e-10
            temp=np.tile(c_array,int(numintc/2-1))
            coevecc=np.concatenate([[1],temp,[4,1]])
            sigsqd=sigsq1/n1+sigsq2/n2
            sigmad=sqrt(sigsqd)
            df1=n1-1
            df2=n2-1
            wbpdf=(intlb/3)*coevecb*stats.beta.pdf(bvec,df1/2,df2/2)
            dft=df1+df2
            sn1=sigsq1/n1
            sn2=sigsq2/n2
            g=(sn1/df1)*bvec+(sn2/df2)*(1-bvec)
            q1=(sn1/df1)*bvec/g
            q2=1-q1
            dfvvec=1/(q1*q1/df1+q2*q2/df2)
            wstavec=stats.t.ppf(1-alpha,dfvvec)
            epvec=np.zeros(lb)
            for i in np.arange(0,lb):
              cu=(del1**2)/(g[i]*wstavec[i]**2)
              intc=cu-cl
              intlc=intc/numintc
              cvec=cl+intlc*np.asarray(range(numintc+1))
              wcpdf=(intlc/3)*coevecc*stats.chi2.pdf(cvec,dft)
              st=np.sqrt(cvec*g[i])*wstavec[i]
              epvec[i]=np.sum(wcpdf*(stats.norm.cdf((-st+del1-mud)/sigmad)-stats.norm.cdf((st-del1 -mud)/sigmad)))
            
            wsepower=sum(wbpdf*epvec)
            ntotal=n1+n2

            print('Nominal power :',round(wsepower,prec) ,' with sample size of ',round(ntotal,prec))
        else:
            print('Violation of normalcy assumption')
    def opt_sample_size(self,power=0.8,prec=4):
        
        n1=len(self.x)
        n2=len(self.y)
        nr=n1/n2
        alpha=self._alpha #type I error rate
        del1=self.margin #equivalence bound
        mu1=mean(self.x) #sample means
        mu2=mean(self.y)
        sigma1=stdev(self.x) #sample standard deviations
        sigma2=stdev(self.y)
        
        if WS_eq.is_normal:
            mud=mu1-mu2
            sigsq1=sigma1**2
            sigsq2=sigma2**2
            numintb=100
            lb=numintb+1
            dd=1e-10
            c_array=np.array([4,2])
            temp=np.tile(c_array,int(numintb/2-1))        
            coevecb=np.concatenate([[1],temp,[4,1]])
            intlb=(1-dd-dd)/numintb
            bvec=dd+intlb*np.asarray(range(numintb+1))
            numintc=5000
            lc=numintc+1
            cl=1e-10
            temp=np.tile(c_array,int(numintc/2-1))
            coevecc=np.concatenate([[1],temp,[4,1]])
            n1=5
            epower=0.0
            while (epower<power and n1<1001):
                n1=n1+1
                n2=np.ceil(nr*n1)
                sigsqd=sigsq1/n1+sigsq2/n2
                sigmad=np.sqrt(sigsqd)
                df1=n1-1
                df2=n2-1
                wbpdf=(intlb/3)*coevecb*stats.beta.pdf(bvec,df1/2,df2/2)
                dft=df1+df2
                sn1=sigsq1/n1
                sn2=sigsq2/n2
                g=(sn1/df1)*bvec+(sn2/df2)*(1-bvec)
                q1=(sn1/df1)*bvec/g
                q2=1-q1
                dfvvec=1/(q1*q1/df1+q2*q2/df2)
                wstavec=stats.t.ppf(1-alpha,dfvvec)
                epvec=np.zeros(lb)
                for i in np.arange(0,lb) :
                    cu=(del1**2)/(g[i]*wstavec[i]**2)
                    intc=cu-cl
                    intlc=intc/numintc
                    cvec=cl+intlc*np.asarray(range(numintc+1))
                    wcpdf=(intlc/3)*coevecc*stats.chi2.pdf(cvec,dft)
                    st=np.sqrt(cvec*g[i])*wstavec[i]
                    tmp=(stats.norm.cdf((-st+del1-mud)/sigmad)-stats.norm.cdf((st-del1-mud)/sigmad))
                    epvec[i]=np.sum(np.multiply(wcpdf,tmp))
                    
                
                epower=np.sum(np.multiply(wbpdf,epvec))
                  
            ntotal=n1+n2

            print('Required sample size :',n1,' and ',n2 ,', Total :',ntotal)
            print('To obtain statistical power of ',round(epower,prec))
        else:
            print('Violation of normalcy assumption')
        
                

        

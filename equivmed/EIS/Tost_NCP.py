import numpy as np
import scipy.stats as stats
from math import sqrt
from statistics import stdev,mean
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
class Tost_paired():
    is_normal=True
    same_var=True
    def __init__(self, x1, y1,delta1):
        ''' TOST alternative implemented for paired mesuremenets'''
        print('Assuming two paired measurements')
        self.x = np.asarray(x1)
        self.y = np.asarray(y1)
        self.margin=delta1
        self._alpha=0.05
        self._Null_Central_Proportion=0.9
        stat, p = stats.levene(x1,y1)
        if p<0.05:
            print('Input measurements do not have equal variances')
            Tost_paired.same_var=False
        shapiro_test1 = stats.shapiro(x1)
        shapiro_test2 = stats.shapiro(y1)
        if shapiro_test1.pvalue>0.05 and shapiro_test2.pvalue>0.05:
            Tost_paired.is_normal=True            
        else:
            Tost_paired.is_normal=False
            print('Inputs are not normally distributed')
            print('The procedure is discouraged')
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, v):
        if not (v > 0) :
            raise ValueError(f'"Alpha" probability must be a number above 0, "{v}" provided.')
        elif v>1:
            raise ValueError(f'"Alpha" probability must be a number below 1, "{v}" provided.')
        self._alpha = v
    @property
    def Null_Central_Proportion(self):
        return self._Null_Central_Proportion
    @Null_Central_Proportion.setter
    def Null_Central_Proportion(self, v):
        if not (v < 1) & (v > 0):
            raise ValueError(f'"Null Central Proportion" must be a number in the range 0 to 1, "{v}" provided.')
        self._Null_Central_Proportion = v
    def remap_data(self, my_data, oMin, oMax, nMin, nMax ):
        #range check
        if oMin == oMax:
            print("Warning: Zero input range")
            return None

        if nMin == nMax:
            print("Warning: Zero output range")
            return None

        #check reversed input range
        reverseInput = False
        oldMin = min( oMin, oMax )
        oldMax = max( oMin, oMax )
        if not oldMin == oMin:
            reverseInput = True

        #check reversed output range
        reverseOutput = False   
        newMin = min( nMin, nMax )
        newMax = max( nMin, nMax )
        if not newMin == nMin :
            reverseOutput = True

        portion = (my_data-oldMin)*(newMax-newMin)/(oldMax-oldMin)
        if reverseInput:
            portion = (oldMax-my_data)*(newMax-newMin)/(oldMax-oldMin)

        result = portion + newMin
        if reverseOutput:
            result = newMax - portion

        return result
    def run_tost(self,do_plot=True):
        ''' TOST on paired measurements'''
        print('Agreement test of central proportion')
        delta=self.margin
        alpha=self._alpha
        prop0=self._Null_Central_Proportion #NULL CENTRAL PROPORTION
        diff=self.x-self.y
        xbar=mean(diff)
        s=stdev(diff)
        n=len(diff)
        pct=1-(1-prop0)/2
        zp=stats.norm.ppf(pct)
        df=n-1
        stdh=s/sqrt(n)
        numint=1000
        c_array=np.array([4,2])
        temp=np.tile(c_array,int(numint/2-1)) 
        coevec=np.concatenate([[1],temp,[4,1]])    
        cl=1e-6
        cu=stats.chi2.ppf(1-cl,df)
        #cu=mia_chi(df,cl)
        interv=cu-cl
        intl=interv/numint
        cvec=cl+(intl*np.arange(numint+1))
        wcpdf=(intl/3)*coevec*stats.chi2.pdf(cvec,df) 
        def fungam():
            gaml=0
            gamu=100
            loop=0
            dalpha=1
            while(abs(dalpha)>1e-8 or dalpha<0):
                gam=(gaml+gamu)/2            
                h=zp*sqrt(n)-gam*np.sqrt(np.true_divide(cvec,df)   )
                ht=h*(cvec<n*df*(zp/gam)**2)
                alphat=np.sum(np.multiply(wcpdf,(2*stats.norm.cdf(ht)-1) ) ) 
                if (alphat>alpha):
                    gaml=gam 
                else: 
                    gamu=gam
                loop=loop+1
                dalpha=alphat-alpha
            return gam
        gam=fungam()
        el=xbar-gam*stdh
        eu=xbar+gam*stdh
        rej=(-delta<el)*(eu<delta)
        if (rej==1):
            #test="reject H0"
            test="The "+str(self._Null_Central_Proportion)+" proportion lies inside the margins of equivalence" # ALTERNATIVE
        else:
            #test="don't reject H0"
            test="The "+str(self._Null_Central_Proportion)+" proportion lies outside the margins of equivalence" # NULL
        #print([gam, el, eu])
        prec=4
        #print([round(gam,prec),round(el,prec),round(eu,prec)])
        print('Lower confidence interval :',round(el,prec))
        print('Upper confidence interval :',round(eu,prec))
        print('TOST Result :',test)
        #print('Critical value :',round(gam,prec))
        if do_plot:
            fig, ax = plt.subplots()
            valore=abs(eu-el)*0.1
            gkde_obj = stats.gaussian_kde(diff)
            x_pts = np.linspace(el, eu, 100)
            estimated_pdf = gkde_obj.evaluate(x_pts)
            pdf_data=self.remap_data(estimated_pdf,min(estimated_pdf),max(estimated_pdf),0.95,1.05)
            ax.add_patch( Rectangle((el, 0.95),
                        abs(el-eu), 0.1,
                        fc='yellow',
                        color ='blue',
                        linewidth = 2,
                        linestyle="dotted",alpha=0.75,label='Actual\nspan') )
            plt.xlabel('Difference between measurements')
            plt.ylim(0.9,1.1)            
            plt.yticks([], [])            
            plt.axvline(x=0,linestyle="dashdot",color="black",label='Perfect\nequiv.')
            plt.axvline(x=-delta,linestyle="--",color="red",linewidth=2)
            plt.axvline(x=delta,linestyle="--",color="red",linewidth=2)
            if -delta-(delta*0.1)<el-valore and delta+(delta*0.1)>eu+valore:
                plt.xlim(-delta-(delta*0.1),delta+(delta*0.1))
            else:
                plt.xlim(el-valore,eu+valore)
            plt.vlines(xbar, ymin=0.95, ymax=1.05, color="blue", linestyles="--",linewidth=1.5)
            #plt.vlines(np.median(diff), ymin=0.95, ymax=1.05, color="gray", linestyles="--",linewidth=1,alpha=0.5)
            ax.axvspan(-delta, delta, alpha=0.25, color='grey',label='Equiv.\nregion')
            plt.plot(x_pts,pdf_data,color='goldenrod',linewidth=2,alpha=0.8)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                                  ncol=3, fancybox=True, shadow=True)
            
            plt.show()
    def stat_power(self):
        prec=4
        diff=self.x-self.y
        alpha=self._alpha #DESIGNATED ALPHA
        prop0=self._Null_Central_Proportion #NULL CENTRAL PROPORTION
        delta=self.margin 
        n=len(diff) 
        mu=np.mean(diff) 
        sigma=stdev(diff) 
        
        pct=1-(1-prop0)/2
        zp=stats.norm.ppf(pct)
        l=-delta
        u=delta
        prop1=stats.norm.cdf((u-mu)/sigma)-stats.norm.cdf((l-mu)/sigma)
        sigsq=sigma**2
        thetal=mu-zp*sigma
        thetau=mu+zp*sigma
        df=n-1
        std=sqrt(sigsq/n)
        numint=1000
        c_array=np.array([4,2])
        temp=np.tile(c_array,int(numint/2-1)) 
        coevec=np.concatenate([[1],temp,[4,1]]) 
        cl=1e-6
        cu=stats.chi2.ppf(1-cl,df)
        interv=cu-cl
        intl=interv/numint
        cvec=cl+intl*np.arange(numint+1)
        wcpdf=(intl/3)*coevec*stats.chi2.pdf(cvec,df)
        def fungam() :
            gaml=0
            gamu=100
            dalpha=1
            while(abs(dalpha)>1e-8 or dalpha<0):
                gam=(gaml+gamu)/2
                h=zp*np.sqrt(n)-gam*np.sqrt(cvec/df)
                ht=h*(cvec<n*df*(zp/gam)**2)
                alphat=np.sum(wcpdf*(2*stats.norm.cdf(ht)-1))
                if (alphat>alpha) :
                    gaml=gam 
                else:
                    gamu=gam
                dalpha=alphat-alpha
          
            return gam
        
        gam=fungam()
        hel=(l-mu)/std+gam*np.sqrt(cvec/df)
        heu=(u-mu)/std-gam*np.sqrt(cvec/df)
        ke=(n*df*(u-l)**2)/(4*gam**2*sigsq)
        helt=hel*(cvec<ke)
        heut=heu*(cvec<ke)
        gpower=np.sum(wcpdf*(stats.norm.cdf(heut)-stats.norm.cdf(helt)))
        print('Nominal power :',round(gpower,prec) ,' with critical value ',round(gam,prec))
        
    def opt_sample_size(self,power=0.8):
        prec=4
        diff=self.x-self.y
        alpha=self._alpha #DESIGNATED ALPHA
        prop0=self._Null_Central_Proportion #NULL CENTRAL PROPORTION
        delta=self.margin 
        
        mu=np.mean(diff) 
        sigma=stdev(diff) 
        print('It might take time...')
        pct=1-(1-prop0)/2
        zp=stats.norm.ppf(pct)
        l=-delta
        u=delta
        prop1=stats.norm.cdf((u-mu)/sigma)-stats.norm.cdf((l-mu)/sigma)
        sigsq=sigma**2
        thetal=mu-zp*sigma
        thetau=mu+zp*sigma
        numint=1000
        c_array=np.array([4,2])
        temp=np.tile(c_array,int(numint/2-1)) 
        coevec=np.concatenate([[1],temp,[4,1]])
        cl=1e-6
        def fungam() :
            gaml=0
            gamu=100
            loop=0
            dalpha=1
            while abs(dalpha)>1e-8 or dalpha<0:
                gam=(gaml+gamu)/2
                h=zp*np.sqrt(n)-gam*np.sqrt(cvec/df)
                ht=h*(cvec<n*df*(zp/gam)**2)
                alphat=np.sum(wcpdf*(2*stats.norm.cdf(ht)-1))
                if (alphat>alpha) :
                    gaml=gam 
                else: 
                    gamu=gam
                loop=loop+1
                dalpha=alphat-alpha
          
            return gam
        
        n=5
        gpower=0
        while (gpower<power and n<5000):
            n=n+1
            df=n-1
            std1=sqrt(sigsq/n)
            cu=stats.chi2.ppf(1-cl,df)
            interv=cu-cl
            intl=interv/numint
            cvec=cl+intl*np.arange(numint+1)
            wcpdf=(intl/3)*coevec*stats.chi2.pdf(cvec,df)
            gam=fungam()
            hel=np.true_divide((l-mu),std1)+gam*np.sqrt(np.true_divide(cvec,df))
            heu=np.true_divide((u-mu),std1)-gam*np.sqrt(np.true_divide(cvec,df))
            ke=np.true_divide((n*df*(u-l)**2),(4*gam**2*sigsq))
            helt=hel*(cvec<ke)
            heut=heu*(cvec<ke)
            gpower=np.sum(wcpdf*(stats.norm.cdf(heut)-stats.norm.cdf(helt)))
        print('Minimal sample size :',round(n,prec) ,'to obtain statistical power of ',round(gpower,prec))
        print('Critical value is ',round(gam,prec) ) 
        
        print([round(gam,prec),round(gpower,prec),round(n,prec)])

        
            

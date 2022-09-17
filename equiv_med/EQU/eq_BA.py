import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde	
from matplotlib.patches import Rectangle
from . import eq_base
import math
class BA_analysis(eq_base.init_eq):
    is_normal=True
    def __init__(self, x, y):
        ''' 
        Revisited BA analysis
        Args:
        x,y laboratory measurements assumed to be normally distributed
        
        NOTE: Limits of Agreement (approximated and exact) and 
        Confidence Intervals can be adjusted with separate function calls
        (default LoA 1.96 and C.I. 95%)
        '''
        super().__init__(x, y)
        self._limitOfAgreement=1.96
        self._confidenceInterval=95
    @property
    def limitOfAgreement(self):
        return self._limitOfAgreement
    @limitOfAgreement.setter
    def limitOfAgreement(self, v):
        if not (v > 0) :
            raise ValueError(f'"limitOfAgreement" must be a number above 0, "{v}" provided.')
        self._limitOfAgreement = v
    @property
    def confidenceInterval(self):
        return self._confidenceInterval
    @confidenceInterval.setter
    def confidenceInterval(self, v):
        if not (v < 99.9) & (v > 1):
            raise ValueError(f'"confidence_Interval" must be a number in the range 1 to 99, "{v}" provided.')
        self._confidenceInterval = v  
    def calculate_Confidence_Intervals(self,md_ci, sd_ci, n_ci, lim_OfAgreement, confidence_Interval):
        confidenceIntervals = dict()

        confidence_Interval = confidence_Interval / 100.
        confidenceIntervals['mean'] = stats.t.interval(confidence_Interval, n_ci-1, 
                                                       loc=md_ci, scale=sd_ci/np.sqrt(n_ci))
        seLoA = ((1/n_ci) + (lim_OfAgreement**2 / (2 * (n_ci - 1)))) * (sd_ci**2)
        loARange = np.sqrt(seLoA) * stats.t._ppf((1-confidence_Interval)/2., n_ci-1)
        confidenceIntervals['upperLoA'] = ((md_ci + lim_OfAgreement*sd_ci) + loARange,
  										   (md_ci + lim_OfAgreement*sd_ci) - loARange)
        confidenceIntervals['lowerLoA'] = ((md_ci - lim_OfAgreement*sd_ci) + loARange,
   										   (md_ci - lim_OfAgreement*sd_ci) - loARange)
        confidenceIntervals['CI'] = ((md_ci - lim_OfAgreement*sd_ci) ,(md_ci + lim_OfAgreement*sd_ci) )        
        return confidenceIntervals
    def Exact_CI(self,md_ci,sd_ci,n_ci,conf_Int,alpha):
        prop=conf_Int/100
        pct=(prop+1)/2
        zp=stats.norm.ppf(pct)
        coverp=1-alpha #alpha is 0.05
        df=n_ci-1
        numint=1000
        c_array=np.array([4,2])
        temp=np.tile(c_array,int(numint/2-1)) 
        coevec=np.concatenate([[1],temp,[4,1]]) 
        def gfun():
            cql=10e-8 #orig 10e-8
            cqu=stats.chi2.ppf(1-cql,df) 
            interv=cqu-cql
            intl=interv/numint
            cvec=cql+(intl*np.arange(numint+1))
            wcpdf=(intl/3)*coevec*stats.chi2.pdf(cvec,df)
            gl=0
            gu=100
            dd=1
            while(abs(dd)>10e-9 or dd<0): #orig 10e-9
                g=(gl+gu)/2
                b=np.sqrt(n_ci)*(-zp+g*np.sqrt(np.true_divide(cvec,df)))
                part1=2*stats.norm.cdf(b)-1
                part2=cvec>df*(zp/g)**2
                cpt=np.sum(wcpdf*(part1*part2) )
                if cpt>coverp: 
                    gu=g
                else:
                    gl=g
                dd=cpt-coverp
            return g
        g=gfun()
        el=md_ci-g*sd_ci
        eu=md_ci+g*sd_ci
        return [el,eu]
    def exact_Bound_sample_size(self,mu,sigma,n_ci,conf_Int,alpha,half_width=2.5):
        if not (conf_Int < 99.9) & (conf_Int > 1):
            raise ValueError(f'"confidence_Interval" must be a number in the range 1 to 99, "{conf_Int}" provided.')
        
        delta=half_width*sigma
        prop=conf_Int/100
        pct=(prop+1)/2
        zp=stats.norm.ppf(pct)
        sigsq=sigma**2
        thetal=mu-zp*sigma
        thetau=mu+zp*sigma
        coverp=1-alpha
        numint=1000
        c_array=np.array([4,2])
        temp=np.tile(c_array,int(numint/2-1)) 
        coevec=np.concatenate([[1],temp,[4,1]])
        def gfun():
            cql=10e-8 #orig 10e-8
            cqu=stats.chi2.ppf(1-cql,df) 
            interv=cqu-cql
            intl=interv/numint
            cvec=cql+(intl*np.arange(numint+1))
            wcpdf=(intl/3)*coevec*stats.chi2.pdf(cvec,df)
            gl=0
            gu=100
            dd=1
            while(abs(dd)>10e-9 or dd<0): #orig 10e-9
                g=(gl+gu)/2
                b=np.sqrt(n_ci)*(-zp+g*np.sqrt(np.true_divide(cvec,df)))
                part1=2*stats.norm.cdf(b)-1
                part2=cvec>df*(zp/g)**2
                cpt=np.sum(wcpdf*(part1*part2) )
                if cpt>coverp: 
                    gu=g
                else:
                    gl=g
                dd=cpt-coverp
            return g
        n_ci=4
        ehe=1000
        while ehe>delta and n_ci<10000:
            n_ci=n_ci+1
            df=n_ci-1
            logc=np.log(np.sqrt(df/2))+math.lgamma(df/2)-math.lgamma(n_ci/2)
            c=np.exp(logc)
            g=gfun()
            ehe=(g/c)*sigma
        print('Two-Sided, Equal-Tailed Tolerance Interval [',thetal,';',thetau,']')
        print('Upper Bound of Assurance Half-Width ',half_width)        
        print('Suggested sample size ',n_ci,' to produce a two-sided ',coverp*100,'% confidence interval of the range of agreement.')
        return [alpha,coverp,mu,sigma,pct,zp,thetal,thetau,ehe,delta,n_ci]
    def exact_Bound_assurance(self,mu,sigma,n_ci,conf_Int,alpha,ap,half_width=2.5):
        if not (conf_Int < 99.9) & (conf_Int > 1):
            raise ValueError(f'"confidence_Interval" must be a number in the range 1 to 99, "{conf_Int}" provided.')

        omega=half_width*sigma
        prop=conf_Int/100
        pct=(prop+1)/2
        zp=stats.norm.ppf(pct)
        sigsq=sigma**2
        thetal=mu-zp*sigma
        thetau=mu+zp*sigma
        coverp=1-alpha
        numint=1000
        c_array=np.array([4,2])
        temp=np.tile(c_array,int(numint/2-1)) 
        coevec=np.concatenate([[1],temp,[4,1]])
        def gfun():
            cql=10e-8 #orig 10e-8
            cqu=stats.chi2.ppf(1-cql,df) 
            interv=cqu-cql
            intl=interv/numint
            cvec=cql+(intl*np.arange(numint+1))
            wcpdf=(intl/3)*coevec*stats.chi2.pdf(cvec,df)
            gl=0
            gu=100
            dd=1
            while(abs(dd)>10e-9 or dd<0): #orig 10e-9
                g=(gl+gu)/2
                b=np.sqrt(n_ci)*(-zp+g*np.sqrt(np.true_divide(cvec,df)))
                part1=2*stats.norm.cdf(b)-1
                part2=cvec>df*(zp/g)**2
                cpt=np.sum(wcpdf*(part1*part2) )
                if cpt>coverp: 
                    gu=g
                else:
                    gl=g
                dd=cpt-coverp
            return g
        n_ci=4
        ape=0
        while ape<ap and n_ci<20000:
            n_ci=n_ci+1
            df=n_ci-1
            g=gfun()
            ape=stats.chi2.cdf((df/g**2)*(omega/sigma)**2,df=df)
        print('Two-Sided, Equal-Tailed Tolerance Interval [',thetal,';',thetau,']')
        print('Lower Bound of Assurance Probability [Target] ',ap)
        print('Lower Bound of Assurance Probability [Actual] ',ape)
        print('Required sample size ',n_ci)
        return [alpha,coverp,mu,sigma,pct,zp,thetal,thetau,omega,ape,ap,n_ci]
    def calc_perc_values(self,variable1,conf_Interv):
        tot=0
        outliers=[]            
        for my_ind,my_value in enumerate(variable1):
            if my_value>conf_Interv['CI'][0] and my_value<conf_Interv['CI'][1]:
                tot +=1
            else:
                outliers.append(my_value)
        print('The ',round(100*(tot/len(variable1)),2),'% of values fall inside the ',self._confidenceInterval,'% BA C.I.')   
        print('BA limits of agreement: [',len(outliers),'] ',round(100*(len(outliers)/len(variable1)),2),'% outliers')
        return tot,outliers
    def calc_exact_perc_values(self,variable1,conf_Interv,alpha):        
        tot2=0
        outliers2=[]                 
        for my_ind,my_value in enumerate(variable1):
            if my_value>conf_Interv[0] and my_value<conf_Interv[1]:
                tot2 +=1
            else:
                outliers2.append(my_value)            
        print('The ',round(100*(tot2/len(variable1)),2),'% of values fall inside the ',self._confidenceInterval,'% Exact C.I. at α:',alpha)                       
        print('Exact methodology: [',len(outliers2),'] ',round(100*(len(outliers2)/len(variable1)),2),'% outliers')     
        return tot2,outliers2
    def check_BA(self):
        var1 = self.x
        var2 = self.y
        diff=var1-var2
        md=np.mean(diff)
        sd=np.std(diff,axis=0)        
        BA_analysis.is_normal=self.normality_test(diff)
        if self.normality_test(diff):
            is_bias=stats.ttest_1samp(a=diff, popmean=0)
            if is_bias[1]<0.05:
                print('Presence of a fixed bias [difference differs significantly from 0]')
            else:
                print('No systematic difference between the measurements')
            confidenceIntervals = self.calculate_Confidence_Intervals(md, sd, len(diff), self._limitOfAgreement, 
                                                                 self._confidenceInterval)            
            alpha=0.05
            e_ci=self.Exact_CI(md,sd,len(diff),self._confidenceInterval,alpha)
            print("Difference is normally distributed")
            print('\n')
            print('Measurement method 1 (New method)')
            t1,o1=self.calc_perc_values(var1,confidenceIntervals)
            print('Measurement method 2 (Reference)')
            t1,o1=self.calc_perc_values(var2,confidenceIntervals)
            print('\n')
            print('Measurement method 1 (New method)')
            t1,o1=self.calc_exact_perc_values(var1,e_ci,alpha)
            print('Measurement method 2 (Reference)')
            t1,o1=self.calc_exact_perc_values(var2,e_ci,alpha)
            print('\n')
        else:
            print('Distribution of differential data is not normal')
            print('Suggested data trasformation')
    def run_analysis(self):
        font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 10
        }
        var1=self.x
        var2=self.y
        diff=var1-var2
        BA_analysis.is_normal=self.normality_test(diff)       
        if self.normality_test(diff):
            md_p=np.mean(diff)
            sd_p=np.std(diff,axis=0)
            ci_p=self.calculate_Confidence_Intervals(md_p,sd_p,len(diff), self._limitOfAgreement, 
                                                                 self._confidenceInterval)
            lower3=ci_p['lowerLoA'][0] # tier 3
            upper3=ci_p['upperLoA'][1]
            lower2=ci_p['CI'][0] # tier 1 lower
            upper2=ci_p['CI'][1]
            lower1=ci_p['lowerLoA'][1] # tier 2
            upper1=ci_p['upperLoA'][0]
            minimal_v=np.min(diff)
            maximal_v=np.max(diff)
            elev=1
            reg_res=self.run_regression()
            lower=md_p-sd_p
            upper=md_p+sd_p
            plt.plot([md_p-self.standard_error(),md_p+self.standard_error()],[0.7,0.7], color='black', linestyle="-")
            plt.plot([minimal_v,maximal_v],[elev+2.7,elev+2.7], 'o',color='black', linestyle="--")
            if len(diff)>100:
                hist, bin_edges = np.histogram(reg_res.resid, bins=int(len(reg_res.resid)/10) )
            else:
                hist, bin_edges = np.histogram(reg_res.resid, bins=10)
            #hist_neg_cumulative = [np.sum(hist[i:]) for i in range(len(hist))]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
            hist_rescaled=self.rescale_linear(hist,2.0,3.0)
            plt.step(bin_centers, hist_rescaled,color='maroon')
            plt.plot(md_p,elev+2.7, 'd',color='crimson') # croos of the mean
            plt.plot([md_p,md_p],[0,elev+2.7], color='crimson', linestyle='--') #line of the mean
            plt.plot([0.0,0.0],[0,elev+2.7], color='limegreen', linestyle='--')       
            plt.plot(0.0,elev+2.7, 's',color='limegreen') # suqare at zero
            plt.axvline(x=lower2, color='royalblue', linestyle='--')
            plt.axvline(x=upper2, color='royalblue', linestyle='--')
            plt.axvspan(lower1, lower3, facecolor='deepskyblue', alpha=0.25)
            plt.axvspan(upper1, upper3, facecolor='deepskyblue', alpha=0.25)
            plt.plot([lower,upper],[elev,elev], color='black', linestyle="-")  # +/- 1 sd
            tot=0            
            for my_ind,my_value in enumerate(diff):
                if my_value>lower2 and my_value<upper2:
                    tot +=1
            mid_point=(upper2+lower2)/2
            bar_ext=((100*(tot/len(diff)))*(upper2-lower2))/self._confidenceInterval
            #bar_ext=((100*(tot/len(diff)))*(maximal_v-minimal_v))/100
            # specify the location of (left,bottom),width,height
            ##rect=Rectangle((mid_point-(bar_ext/2), elev+4.4), bar_ext, 0.5, linewidth=1, edgecolor='navy', facecolor='dodgerblue')
            rect=Rectangle((md_p-(bar_ext/2), elev+4.4), bar_ext, 0.5, linewidth=1, edgecolor='navy', facecolor='dodgerblue')
            plt.gca().add_patch(rect)
            plt.text(mid_point, 5.6, str(round( (100*(tot/len(diff))),1) )+' %', fontdict=font)
            #plt.text(md_p, 5.6, str(round( (100*(tot/len(diff))),1) )+' %', fontdict=font)
            if len(diff)<1000:
                dist_space = np.linspace(min(diff), max(diff), 1000)
                x_values= np.linspace(min(diff), max(diff), len(diff))
                yinterp = np.interp(dist_space, x_values, diff)                
                kde = gaussian_kde( yinterp )                
            else:
                kde = gaussian_kde( diff )
                dist_space = np.linspace( min(diff), max(diff), len(diff) )
                kde(dist_space)
            evaluated = kde.evaluate(dist_space)
            diff2=np.interp(evaluated, (evaluated.min(), evaluated.max()), (0.3, 4.5))
            for x_diff in diff:
                plt.plot([x_diff,x_diff],[0.2,0.4],color='gray',alpha=0.45)
            plt.plot( dist_space, diff2 ,color='gray',alpha=0.3,linewidth=2)
            e_ci=self.Exact_CI(md_p,sd_p,len(diff),self._confidenceInterval,0.05)
            tot2=0
            for my_ind,my_value in enumerate(diff):
                if my_value>e_ci[0] and my_value<e_ci[1]:
                    tot2 +=1
            mid_point=(e_ci[0]+e_ci[1])/2
            bar_ext=((100*(tot2/len(diff)))*(e_ci[1]-e_ci[0]))/self._confidenceInterval
            #bar_ext=((100*(tot2/len(diff)))*(maximal_v-minimal_v))/100
            font['color']='red'
            rect=Rectangle((mid_point-(bar_ext/2), elev+3.8), bar_ext, 0.5, linewidth=1, edgecolor='r', facecolor='papayawhip')
            #rect=Rectangle((md_p-(bar_ext/2), elev+3.8), bar_ext, 0.5, linewidth=1, edgecolor='r', facecolor='papayawhip')
            plt.gca().add_patch(rect)
            plt.text(mid_point, 5, str(round( (100*(tot2/len(diff))),1) )+' %', fontdict=font)
            #plt.text(md_p, 5, str(round( (100*(tot2/len(diff))),1) )+' %', fontdict=font)
            plt.axvline(x=e_ci[0], color='orange', linestyle='--')
            plt.axvline(x=e_ci[1], color='orange', linestyle='--')
            plt.yticks([0,0.7,1,2.5,3.7,5.05,5.65],["",u"\u00B1SE",u"\u00B1SD","Resid","Range","Exact\nProp","BA\nProp"])
            plt.xlabel("Difference between two variables")
            plt.ylim(0,6)
            
            plt.show()
            self.check_BA()
        else:
            print('Data not normally distributed')
    def minimal_detectable_change(self):
        var1=self.x
        var2=self.y
        diff=var1-var2
        BA_analysis.is_normal=self.normality_test(diff)       
        if self.normality_test(diff):        
            md_p=np.mean(diff)
            sd_p=np.std(diff,axis=0)
            ci_p=self.calculate_Confidence_Intervals(md_p,sd_p,len(diff), self._limitOfAgreement, 
                                                                 self._confidenceInterval)
        
            lower2=ci_p['CI'][0] # tier 1 lower
            upper2=ci_p['CI'][1]        
            e_ci=self.Exact_CI(md_p,sd_p,len(diff),self._confidenceInterval,0.05)
            MDC1=(upper2-lower2)*0.5
            MDC2=(e_ci[1]-e_ci[0])*0.5
            print('Minimal Detectable Change [Approx. Lim. of Agreement] :',MDC1)
            print('Minimal Detectable Change [Exact Lim. of Agreement] :',MDC2)

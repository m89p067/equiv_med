import numpy as np
from math import pi
import matplotlib.pyplot as plt
import pandas as pd
class Radar_plots:      
    def __init__(self,indexes_list,print_abbr=False):
        ''' 
        Radar plots for series of indexes
        Args:
        indexes_list    a list with index names (follow nomenclature)
        print_abbr  prints a report with abbreviations used on the plots as reference
        '''
        if len(indexes_list)<5 or len(indexes_list)>9:
            raise ValueError("Please, length of the selected indexes should be between 5 and 9")
        self.select_list=indexes_list
        abbrev_list={'True positive fraction':'TPF',
                'False Negative Fraction':'FNF',
                'True Negative Fraction':'TNF',
                'False Positive Fraction':'FPF',
                'Positive Predictive Value':'PPV',
                'Negative Predictive Value':'NPV',
                'Positive Likelihood Ratio':'PLR',
                'Negative Likelihood Ratio': 'NLR',
                'Diagnostic Odds Ratio':'DOR',
                'Diagnostic effectiveness (accuracy)':'ACC',
                'Youden index':'YOUD' ,
                'Cohen kappa coefficient':'K',
                'Balanced accuracy':'BACC',
                'Error rate':'ER',
                'False discovery rate':'FDR',
                'Precision':'Prec',
                'Recall':'Rec',
                'F-measure (aka F1-score or simply F-score)':'F1',
                'G measure':'G',
                'Matthews correlation coefficient':'MCC',
                'Bookmaker informedness':'BM',
                'Markedness':'MK','False negative rate':'FNR','False positive rate':'FPR',
                'False omission rate':'FOR','Sensitivity':'TPF','Specificity':'TNF',
                'Number necessary to test':'NNT','Number necessary to diagnose':'NND',
                'Critical success index':'CSI','Prevalence threshold':'PT','True negative rate':'TNR',
                'Fowlkes–Mallows index':'FMI','Prevalence':'PR','Bias':'B'}
        self.abbreviations=abbrev_list
        if print_abbr==True:
            print(abbrev_list)
    def map_fields(self,init_dict, map_dict, res_dict=None):
            res_dict = res_dict or {}
            for k, v in init_dict.items():
                print("Key: ", k)
                if isinstance(v, dict):
                    print("value is a dict - recursing")
                    v = map_fields(v, map_dict[k])
                elif k in map_dict.keys():
                    print("Remapping:", k, str(map_dict[k]))
                    k = str(map_dict[k])
                res_dict[k] = v
            return res_dict
    def radar_plot(self,input_data,color_radar='red',title_string=''):  
        ''' Single radar plot'''
        res = dict((k, input_data[k]) for k in self.select_list if k in input_data)
        values_red=self.map_fields(res, self.abbreviations)
        
        labels, values = zip(*values_red.items())
        if any(n < 0 for n in values):            
            raise ValueError('Sorry, only non-negative indexes for radar plots')
        if any(n > 1 for n in values):            
            raise ValueError('Sorry, only indexes below 1 for radar plots')
        num_vars = len(labels)
        #angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles=[n/float(num_vars)*2*pi for n in range(num_vars)]
        values += values[:1]
        angles += angles[:1]        

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], labels, color='grey', size=9)
        ax.set_rlabel_position(0)
        plt.yticks([0.25,0.5,0.75], ["0.25","0.5","0.75"], color="grey", size=7)
        plt.ylim(0,1)
         
        
        ax.plot(angles, values, linewidth=2, linestyle='solid',color=color_radar)
         
        
        ax.fill(angles, values, color_radar, alpha=0.1)
        if len(title_string)>0:
            plt.title(title_string)
        
        plt.show()
    def radar_plots(self,input_data1,input_data2,color_radar1='red',color_radar2='blue',
                    title_string1='Method 1',title_string2='Method 2',overlapping=False,sup_title='Diagnostic data'):  
        label_format = '{:,.2f}'
        res1 = dict((k, input_data1[k]) for k in self.select_list if k in input_data1)
        res2 = dict((k, input_data2[k]) for k in self.select_list if k in input_data2)

        values_red1=self.map_fields(res1, self.abbreviations)
        values_red2=self.map_fields(res2, self.abbreviations)

        labels1, values1 = zip(*values_red1.items())
        labels2, values2 = zip(*values_red2.items())
        if any(n < 0 for n in values1):            
            raise ValueError('Sorry, only non-negative indexes for radar plots')
        if any(n < 0 for n in values2):            
            raise ValueError('Sorry, only non-negative indexes for radar plots')
        if any(n > 1 for n in values1):            
            raise ValueError('Sorry, only indexes below 1 for radar plots')
        if any(n > 1 for n in values2):            
            raise ValueError('Sorry, only indexes below 1 for radar plots')


        num_vars1 = len(labels1)
        num_vars2 = len(labels2)

        angles1=[n/float(num_vars1)*2*pi for n in range(num_vars1)]
        values1 += values1[:1]
        angles1 += angles1[:1]        

        angles2=[n/float(num_vars2)*2*pi for n in range(num_vars2)]
        values2 += values2[:1]
        angles2 += angles2[:1] 
        if overlapping==False:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), subplot_kw=dict(polar=True))
            
            ax[0].set_xticks(angles1[:-1])
            ax[0].set_xticklabels(labels1, color='grey', size=9)
            ax[0].set_rlabel_position(0)
            w0=ax[0].get_yticks()

                        
            ax[1].set_xticks(angles2[:-1])
            ax[1].set_xticklabels(labels2, color='grey', size=9)
            ax[1].set_rlabel_position(0)
            w1=ax[1].get_yticks()
            
            w0.extend(y for y in w1 if y not in w0)
          
            max_all = max(w0 )
            min_all = min(w0 )

            asse_y=np.linspace(min_all+(min_all*0.15),max_all, num=5,endpoint=False)

            ax[0].set_yticks(asse_y)
            
            ax[0].set_yticklabels([label_format.format(x) for x in asse_y], color="grey", size=7)
            ax[1].set_yticks(asse_y)
            
            ax[1].set_yticklabels([label_format.format(x) for x in asse_y], color="grey", size=7)                  
            
            ax[0].plot(angles1, values1, linewidth=2, linestyle='solid',color=color_radar1)
            ax[1].plot(angles2, values2, linewidth=2, linestyle='solid',color=color_radar2) 
            
            ax[0].fill(angles1, values1, color_radar1, alpha=0.1)
            ax[1].fill(angles2, values2, color_radar2, alpha=0.1)
            if len(title_string1)>0:
                ax[0].set_title(title_string1)
            if len(title_string2)>0:
                ax[1].set_title(title_string2)

            ax[0].set_ylim(min_all,max_all)
            ax[1].set_ylim(min_all,max_all)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="polar")

            
            l1, = ax.plot(angles1, values1, linewidth=2, linestyle='solid',color=color_radar1, marker="o", label=title_string1)
            l2, = ax.plot(angles2, values2, linewidth=2, linestyle='solid',color=color_radar2, marker="o", label=title_string2)

            def _closeline(line):
                x, y = line.get_data()
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
            [_closeline(l) for l in [l1,l2]]

            ax.set_xticks(angles2[:-1])
            ax.set_xticklabels(labels2, color='grey', size=9)

            asse_y=np.linspace(0,1, num=5,endpoint=True)
            ax.set_yticks(asse_y)
            ax.set_yticklabels([label_format.format(x) for x in asse_y], color="grey", size=7)

            ax.set_ylim(0,1)

            plt.legend(bbox_to_anchor=(1,1), loc="upper left")
            if len(sup_title)>0:
                plt.title(sup_title)
            
            
        
        plt.show()
    def polar_bars(self,input_data1,input_data2,color_radar1='orange',color_radar2='blue',
                    title_string1='Method 1',title_string2='Method 2',sup_title='Diagnostic data'):
        res1 = dict((k, input_data1[k]) for k in self.select_list if k in input_data1)
        res2 = dict((k, input_data2[k]) for k in self.select_list if k in input_data2)

        values_red1=self.map_fields(res1, self.abbreviations)
        values_red2=self.map_fields(res2, self.abbreviations)

        labels1, values1 = zip(*values_red1.items())
        labels2, values2 = zip(*values_red2.items()) 
        
        sections = np.linspace(0.0, 2 * np.pi, len(labels1), endpoint=False)
        
        width = 0.2 # width of bars
        fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,subplot_kw=dict(projection='polar'))        
        
        bars = ax.bar(sections + 0 * width, values1, width=width, bottom=0.0,label=title_string1,color=color_radar1)            
        bars = ax.bar(sections + 1 * width, values2, width=width, bottom=0.0,label=title_string2,color=color_radar2)
        
        ax.set_xticks(sections)
        ax.set_xticklabels(labels1)
        if len(sup_title)>0:
            plt.title(sup_title)
        plt.legend(loc="best")
        
        plt.show()

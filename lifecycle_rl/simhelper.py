'''

    simhelper.py

    implements methods that help in interpreting simulation results

'''

import h5py
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import norm
import locale
from tabulate import tabulate
import pandas as pd
from tqdm import tqdm_notebook as tqdm

class SimHelpher():
    def plot_stats(self,datafile,baseline=None,ref=None,xlabel=None,dire=None):
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,kiila,muut,tyollaste,tyotaste,tyossa=self.load_data(datafile)
        mean_rew,mean_verot,mean_ps,mean_htv,mean_muut=self.comp_means(datafile)

        if baseline is not None:
            baseline_tax,baseline_verot,baseline_rew,baseline_ps,baseline_htv,baseline_kiila,baseline_muut=self.load_data(baseline)
            self.plot_taxes(additional_income_tax,mean_rew,mean_verot,mean_ps,mean_htv,mean_muut,
                       ref_additional_tax=baseline_tax,ref_mean_verot=baseline_verot,ref_mean_ps=baseline_ps,
                       ref_mean_htv=baseline_htv,ref_mean_muut=baseline_muut,ref_mean_rew=baseline_rew,
                       xlabel=xlabel,dire=dire)
        elif ref is not None:
            ref_rew,ref_verot,ref_ps,ref_htv,ref_muut,ref_kiila=self.get_refstats(ref,scale=additional_income_tax)
            self.plot_taxes(additional_income_tax,mean_rew,mean_verot,mean_ps,mean_htv,mean_muut,
                       ref_additional_tax=additional_income_tax,ref_mean_verot=ref_verot,ref_mean_ps=ref_ps,
                       ref_mean_htv=ref_htv,ref_mean_muut=ref_muut,ref_mean_rew=ref_rew,xlabel=xlabel,dire=dire)
        else:
            self.plot_taxes(additional_income_tax,mean_rew,mean_verot,mean_ps,mean_htv,mean_muut,xlabel=xlabel,dire=dire)

        self.plot_total(additional_income_tax,total_rew,total_verot,total_htv,total_ps)


    def plot_taxes(self,additional_tax,mean_rew,mean_verot,mean_ps,mean_htv,mean_muut,
                   ref_additional_tax=None,ref_mean_verot=None,ref_mean_ps=None,ref_mean_htv=None,ref_mean_muut=None,
                   ref_mean_rew=None,percent_scale=False,
                   dire=None,label1=None,label2=None,xlabel='Muutos [%-yks]'):    
        #print(rew,ps,verot)
        if percent_scale:
            scale=100
        else:
            scale=1

        fig,ax=plt.subplots()
        print(additional_tax,mean_rew)
        ax.plot(scale*additional_tax,mean_rew,label=label1)
        plt.title('Reward')
        if ref_additional_tax is not None:
            ax.plot(scale*ref_additional_tax,ref_mean_rew.T,label=label2)
        ax.legend()
        ax.set_ylabel('Hyöty')
        ax.set_xlabel(xlabel)
        if dire is not None:
            plt.savefig(dire+'reward.eps', format='eps')
            plt.savefig(dire+'reward.png', format='png')
            print('dire',dire)
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(scale*additional_tax,mean_ps,label=label1)
        if ref_additional_tax is not None:
            ax.plot(scale*ref_additional_tax,ref_mean_ps.T,label=label2)
        #print(mean_ps)
        plt.title('Palkkasumma')
        ax.set_ylabel('Palkkasumma [euroa]')
        ax.set_xlabel(xlabel)
        ax.legend()
        if dire is not None:
            plt.savefig(dire+'palkkasumma.eps', format='eps')
            plt.savefig(dire+'palkkasumma.png', format='png')
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(scale*additional_tax,mean_verot,label=label1)
        if ref_additional_tax is not None:
            ax.plot(scale*ref_additional_tax,ref_mean_verot.T,label=label2)
        plt.title('Verot ja veronluonteiset maksut')
        ax.legend()
        ax.set_ylabel('Verot [euroa]')
        ax.set_xlabel(xlabel)
        if dire is not None:
            plt.savefig(dire+'verot.eps', format='eps')
            plt.savefig(dire+'verot.png', format='png')
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(scale*additional_tax,mean_htv,label=label1)
        if ref_additional_tax is not None:
            ax.plot(scale*ref_additional_tax,ref_mean_htv.T,label=label2)
        plt.title('Työnteko')
        ax.legend()
        ax.set_ylabel('Henkilötyövuodet')
        ax.set_xlabel(xlabel)
        if dire is not None:
            plt.savefig(dire+'htv.eps', format='eps')
            plt.savefig(dire+'htv.png', format='png')
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(scale*additional_tax,mean_muut,label=label1)
        if ref_additional_tax is not None:
            ax.plot(scale*ref_additional_tax,ref_mean_muut.T,label=label2)
        ax.legend()
        ax.set_ylabel('Muut tulot [euroa]')
        ax.set_xlabel(xlabel)
        if dire is not None:
            plt.savefig(dire+'muut.eps', format='eps')
            plt.savefig(dire+'muut.png', format='png')

        plt.title('Tarvittavat muut tulot')
        plt.show()

    def plot_total(self,additional_tax,total_rew,total_verot,total_htv,total_ps,dire=None,xlabel=None):
        plt.plot(100*additional_tax,total_rew)
        plt.title('reward')
        plt.show()

        plt.plot(100*additional_tax,total_verot)
        plt.title('verot')
        if dire is not None:
            plt.savefig(dire+'verot_kaikki.png', format='png')
        plt.show()

        plt.plot(100*additional_tax,total_htv)
        plt.title('htv')
        plt.show()


        plt.plot(100*additional_tax,total_ps)
        plt.title('palkkasumma')
        plt.show()

    def plot_taxes_prop(self,additional_tax,osuus_vero,dire=None,xlabel=None):
        plt.plot(100*additional_tax,osuus_vero[:,:,0])
        plt.title('proportion of taxes paid by the not employed')
        plt.show()

        plt.plot(100*additional_tax,osuus_vero[:,:,1])
        plt.title('proportion of taxes paid by the employed')
        plt.show()

        plt.plot(100*additional_tax,osuus_vero[:,:,2])
        plt.title('proportion of taxes paid by the retired')
        plt.show()

    def plot_verokiila(self,additional_tax,kiila,xlabel=None):
        fig,ax=plt.subplots()
        ax.plot(100*additional_tax,100*kiila)
        ax.set_ylabel('Verokiila [%]')
        ax.set_xlabel('Lisävero [%-yks]')
        plt.show()


    def save_data(self,filename,additional_income_tax,total_verot,total_rew,total_ps,total_htv,
                  total_kiila,total_muut,total_tyollaste,total_tyotaste,total_tyossa,
                  osuus_vero=None,osuus_kunnallisvero=None,osuus_valtionvero=None):
        f = h5py.File(filename, 'w')
        ftype='float64'

        _ = f.create_dataset('additional_income_tax', data=additional_income_tax, dtype=ftype)

        _ = f.create_dataset('total_verot', data=total_verot, dtype=ftype)
        _ = f.create_dataset('total_rew', data=total_rew, dtype=ftype)
        _ = f.create_dataset('total_ps', data=total_ps, dtype=ftype)
        _ = f.create_dataset('total_htv', data=total_htv, dtype=ftype)
        _ = f.create_dataset('total_kiila', data=total_kiila, dtype=ftype)
        _ = f.create_dataset('total_muut', data=total_muut, dtype=ftype)
        _ = f.create_dataset('osuus_vero', data=osuus_vero, dtype=ftype)
        _ = f.create_dataset('osuus_kunnallisvero', data=osuus_kunnallisvero, dtype=ftype)
        _ = f.create_dataset('osuus_valtionvero', data=osuus_valtionvero, dtype=ftype)
        _ = f.create_dataset('total_tyollaste', data=total_tyollaste, dtype=ftype)
        _ = f.create_dataset('total_tyotaste', data=total_tyotaste, dtype=ftype)
        _ = f.create_dataset('total_tyossa', data=total_tyossa, dtype=ftype)

        f.close()

    def load_data(self,filename):
        f = h5py.File(filename, 'r')

        additional_income_tax=f.get('additional_income_tax').value
        total_verot=f.get('total_verot').value
        total_rew=f.get('total_rew').value
        total_ps=f.get('total_ps').value
        total_htv=f.get('total_htv').value
        total_kiila=f.get('total_kiila').value
        total_muut=f.get('total_muut').value
        if 'total_tyollaste' in f:
            total_tyollaste=f.get('total_tyollaste').value
            total_tyotaste=f.get('total_tyotaste').value
            total_tyossa=f.get('total_tyossa').value
        else:
            total_tyollaste=additional_income_tax*0
            total_tyotaste=additional_income_tax*0
            total_tyossa=additional_income_tax*0
    
        #osuus_vero,osuus_kunnallisvero,osuus_valtionvero

        f.close()

        return additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,total_tyollaste,total_tyotaste,total_tyossa

    def comp_means(self,filename):
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,_,_,_=load_data(filename)
        mean_rew=np.mean(total_rew,axis=1).reshape(-1, 1)
        mean_verot=np.mean(total_verot,axis=1).reshape(-1, 1)
        mean_ps=np.mean(total_ps,axis=1).reshape(-1, 1)
        mean_htv=np.mean(total_htv,axis=1).reshape(-1, 1)
        mean_muut=np.mean(total_muut,axis=1).reshape(-1, 1)

        return mean_rew,mean_verot,mean_ps,mean_htv,mean_muut

    def get_refstats(self,filename,scale=None):
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,_,_,_=\
            load_data(filename)
        z=np.where(np.abs(additional_income_tax)<0.001)

        if scale is not None:
            scaling=np.squeeze(np.ones(scale.shape))
        else:
            scaling=1

        # baseline muut
        baseline_muut=total_muut[z]*scaling
        baseline_verot=total_verot[z]*scaling.T
        baseline_rew=total_rew[z]*scaling.T
        baseline_ps=total_ps[z]*scaling.T
        baseline_htv=total_htv[z]*scaling.T
        baseline_kiila=total_kiila[z]*scaling.T

        print('baseline',baseline_htv.shape)

        return baseline_rew,baseline_verot,baseline_ps,baseline_htv,baseline_muut,baseline_kiila


    def comp_stats(self,taxresults,additional,datafile,repeats):
        k=0
        total_verot=np.zeros(( additional.shape[0],repeats))
        total_rew=np.zeros(( additional.shape[0],repeats))
        total_ps=np.zeros(( additional.shape[0],repeats))
        total_htv=np.zeros(( additional.shape[0],repeats))
        total_tyollaste=np.zeros(( additional.shape[0],repeats))
        total_tyotaste=np.zeros(( additional.shape[0],repeats))
        total_muut_tulot=np.zeros(( additional.shape[0],repeats))
        total_kiila=np.zeros(( additional.shape[0],repeats))
        total_tyossa=np.zeros(( additional.shape[0],repeats))

        osuus_valtionvero=np.zeros(( additional.shape[0],repeats,3))
        osuus_kunnallisvero=np.zeros(( additional.shape[0],repeats,3))
        osuus_vero=np.zeros(( additional.shape[0],repeats,3))

        for tax in additional:
            cc=Lifecycle(env='unemployment-v2',minimal=False,mortality=mortality,perustulo=False,
                         randomness=randomness,pinkslip=pinkslip,plotdebug=plotdebug,
                         additional_income_tax=tax)
            rew=[]
            ps=[]
            verot=[]
            htv=[]
            muut_tulot=[]
            for repeat in range(0,repeats):
                num=int(np.round(100*tax))
                print('computing extra tax {} repeat {}'.format(tax,repeat))
                results=taxresults+'_{}_{}'.format(num,repeat)
                r,qps,qv,h,muut,kiila,tyoll,tyot,lkm=cc.render_laffer(load=results)
                vvosuus,kvosuus,vosuus=cc.comp_taxratios(grouped=True)
                rew.append(r)
                ps.append(qps)
                verot.append(qv)
                htv.append(h)
                muut_tulot.append(muut)
                total_verot[k,repeat]=qv
                total_rew[k,repeat]=r
                total_ps[k,repeat]=qps
                total_htv[k,repeat]=h
                total_muut_tulot[k,repeat]=muut
                total_kiila[k,repeat]=kiila
                total_tyotaste[k,repeat]=tyot
                total_tyollaste[k,repeat]=tyoll
                total_tyossa[k,repeat]=lkm
                osuus_vero[k,repeat,:]=vosuus
                osuus_kunnallisvero[k,repeat,:]=kvosuus
                osuus_valtionvero[k,repeat,:]=vvosuus
            k=k+1

        save_data(datafile,additional,total_verot,total_rew,total_ps,total_htv,
                  total_kiila,total_muut_tulot,total_tyollaste,total_tyotaste,total_tyossa,
                  osuus_vero,osuus_kunnallisvero,osuus_valtionvero)

    def comp_elasticity(self,x,y):
        xl=x.shape[0]
        yl=x.shape[0]    
        el=np.zeros(xl.shape[0])
    
        for k in range(1,xl):
            dx=(x[k+1]-x[k])/x[k]
            dy=(y[k+1]-y[k])/y[k]        
            el[k]=dy/dx
        
        return el    
    
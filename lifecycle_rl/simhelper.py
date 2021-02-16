'''

    simhelper.py

    implements methods that help in interpreting multiple simulation results with a variety of parameters

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
#import smoothfit
from tqdm import tqdm_notebook as tqdm
from . lifecycle import Lifecycle

class SimHelper():
    def plot_stats(self,datafile,baseline=None,ref=None,xlabel=None,dire=None,plot_kiila=True,percent_scale=False):
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,kiila,muut,tyollaste,tyotaste,tyossa,\
            osatyoratio,osuus_vero,osuus_kunnallisvero,osuus_valtionvero=self.load_data(datafile)
        mean_rew,mean_verot,mean_ps,mean_htv,mean_muut,mean_kiila,mean_tyossa,mean_osatyoratio,mean_tyotaste=self.comp_means(datafile)

        self.plot_tyossa(additional_income_tax,mean_tyossa,mean_htv,xlabel=xlabel,percent_scale=percent_scale,dire=dire)
        self.plot_tyoton(additional_income_tax,mean_tyotaste,xlabel=xlabel,percent_scale=percent_scale,dire=dire)

        if baseline is not None:
            baseline_tax,baseline_verot,baseline_rew,baseline_ps,baseline_htv,baseline_kiila,baseline_muut=self.load_data(baseline)
            if plot_kiila:
                self.plot_verokiila(additional_income_tax,kiila,ref_additional_tax=baseline_tax,ref_kiila=baseline_kiila)

            self.plot_taxes(additional_income_tax,mean_rew,mean_verot,mean_ps,mean_htv,mean_muut,
                       ref_additional_tax=baseline_tax,ref_mean_verot=baseline_verot,ref_mean_ps=baseline_ps,
                       ref_mean_htv=baseline_htv,ref_mean_muut=baseline_muut,ref_mean_rew=baseline_rew,
                       xlabel=xlabel,dire=dire,percent_scale=percent_scale)
        elif ref is not None:
            ref_rew,ref_verot,ref_ps,ref_htv,ref_muut,ref_kiila=self.get_refstats(ref,scale=additional_income_tax)
            if plot_kiila:
                self.plot_verokiila(additional_income_tax,kiila,ref_additional_tax=additional_income_tax,ref_kiila=ref_kiila)
            self.plot_taxes(additional_income_tax,mean_rew,mean_verot,mean_ps,mean_htv,mean_muut,
                       ref_additional_tax=additional_income_tax,ref_mean_verot=ref_verot,ref_mean_ps=ref_ps,
                       ref_mean_htv=ref_htv,ref_mean_muut=ref_muut,ref_mean_rew=ref_rew,xlabel=xlabel,dire=dire)
        else:
            self.plot_verokiila(additional_income_tax,kiila)
            self.plot_taxes(additional_income_tax,mean_rew,mean_verot,mean_ps,mean_htv,mean_muut,xlabel=xlabel,dire=dire,percent_scale=percent_scale)
            #self.plot_taxes_prop(additional_income_tax,TELosuus_vero,xlabel='Verokiila')
            
        self.plot_elasticity(mean_kiila,mean_htv,xlabel='Verokiila',ylabel='Työnmäärän jousto',dire=dire,percent_scale=percent_scale)
        self.plot_elasticity(mean_kiila,mean_tyossa,xlabel='Verokiila',ylabel='Työnteon jousto',dire=dire,percent_scale=percent_scale)
        self.plot_osatyo(additional_income_tax,mean_osatyoratio,xlabel=xlabel,ylabel='Osuus [%-yks]',dire=dire,percent_scale=percent_scale)
        self.plot_veroosuudet(additional_income_tax,osuus_vero,osuus_kunnallisvero,osuus_valtionvero)

        self.plot_total(additional_income_tax,total_rew,total_verot,total_htv,total_ps) #,percent_scale=percent_scale)

    def plot_veroosuudet(self,additional_income_tax,osuus_vero,osuus_kunnallisvero,osuus_valtionvero,dire=None,label1=None,label2=None,
                        xlabel='Muutos [%-yks]',percent_scale=False,ylabel='Eläkeläisten osuus veroista',fname='elas'):
        vero=np.squeeze(osuus_vero[:,:,2])/np.squeeze(np.sum(osuus_vero,axis=2))
        kunnallis=np.squeeze(osuus_kunnallisvero[:,:,2])/np.squeeze(np.sum(osuus_kunnallisvero,axis=2))
        valtio=np.squeeze(osuus_valtionvero[:,:,2])/np.squeeze(np.sum(osuus_valtionvero,axis=2))
        self.plot_osuus(additional_income_tax,vero,label1=label1,xlabel=xlabel,ylabel='Eläkeläisten osuus veroista',dire=dire,percent_scale=percent_scale)
        self.plot_osuus(additional_income_tax,kunnallis,label1=label1,xlabel=xlabel,ylabel='Eläkeläisten osuus kunnallisverosta',dire=dire,percent_scale=percent_scale)
        self.plot_osuus(additional_income_tax,valtio,label1=label1,xlabel=xlabel,ylabel='Eläkeläisten osuus ansiotuloverosta',dire=dire,percent_scale=percent_scale)
        vero=np.squeeze(osuus_vero[:,:,1])/np.squeeze(np.sum(osuus_vero,axis=2))
        kunnallis=np.squeeze(osuus_kunnallisvero[:,:,1])/np.squeeze(np.sum(osuus_kunnallisvero,axis=2))
        valtio=np.squeeze(osuus_valtionvero[:,:,1])/np.squeeze(np.sum(osuus_valtionvero,axis=2))
        self.plot_osuus(additional_income_tax,vero,label1=label1,xlabel=xlabel,ylabel='Työn osuus veroista',dire=dire,percent_scale=percent_scale)
        self.plot_osuus(additional_income_tax,kunnallis,label1=label1,xlabel=xlabel,ylabel='Työn osuus kunnallisverosta',dire=dire,percent_scale=percent_scale)
        self.plot_osuus(additional_income_tax,valtio,label1=label1,xlabel=xlabel,ylabel='Työn osuus ansiotuloverosta',dire=dire,percent_scale=percent_scale)
        vero=np.squeeze(osuus_vero[:,:,0])/np.squeeze(np.sum(osuus_vero,axis=2))
        kunnallis=np.squeeze(osuus_kunnallisvero[:,:,0])/np.squeeze(np.sum(osuus_kunnallisvero,axis=2))
        valtio=np.squeeze(osuus_valtionvero[:,:,0])/np.squeeze(np.sum(osuus_valtionvero,axis=2))
        self.plot_osuus(additional_income_tax,vero,label1=label1,xlabel=xlabel,ylabel='Etuudensaajien osuus veroista',dire=dire,percent_scale=percent_scale)
        self.plot_osuus(additional_income_tax,kunnallis,label1=label1,xlabel=xlabel,ylabel='Etuudensaajien osuus kunnallisverosta',dire=dire,percent_scale=percent_scale)
        self.plot_osuus(additional_income_tax,valtio,label1=label1,xlabel=xlabel,ylabel='Etuudensaajien osuus ansiotuloverosta',dire=dire,percent_scale=percent_scale)
    
    def plot_osuus(self,x,y,dire=None,label1=None,label2=None,xlabel='Muutos [%-yks]',percent_scale=False,ylabel='Elasticity',fname='elas'):
        if percent_scale:
            scale=100
        else:
            scale=1
    
        fig,ax=plt.subplots()
        ax.plot(scale*x,y,label=label1)
        #plt.title(fname)
        #if ref_additional_tax is not None:
        #    ax.plot(scale*ref_additional_tax,ref_mean_rew,label=label2)
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if dire is not None:
            plt.savefig(dire+fname+'_'+ylabel+'.eps', format='eps')
            plt.savefig(dire+fname+'_'+ylabel+'.png', format='png')
            print('dire',dire)
            
        plt.show()

    def plot_elasticity(self,additional_income_tax,htv,dire=None,label1=None,label2=None,xlabel='Muutos [%-yks]',percent_scale=False,ylabel='Elasticity'):
#         if percent_scale:
#             scale=100
#         else:
#             scale=1
    
        el=self.comp_elasticity(additional_income_tax,htv)
        self.plot_osuus(additional_income_tax,el,label1=label1,xlabel=xlabel,ylabel=ylabel,dire=dire,percent_scale=percent_scale)
#         fig,ax=plt.subplots()
#         ax.plot(scale*additional_income_tax,el,label=label1)
#         plt.title('Elasticity')
#         #if ref_additional_tax is not None:
#         #    ax.plot(scale*ref_additional_tax,ref_mean_rew,label=label2)
#         ax.legend()
#         ax.set_ylabel(ylabel)
#         ax.set_xlabel(xlabel)
#         if dire is not None:
#             plt.savefig(dire+'elasticity '+ylabel+'.eps', format='eps')
#             plt.savefig(dire+'elasticity '+ylabel+'.png', format='png')
#             print('dire',dire)
            
#         a=np.min(scale*additional_income_tax)
#         b=np.max(scale*additional_income_tax)
#         u = smoothfit.fit1d(scale*additional_income_tax,el, a, b, 1000, degree=1, lmbda=1.0e-6)
#         x = numpy.linspace(a, b, 201)
#         vals = [u(xx) for xx in x]
#         plt.plot(x, vals, "-", label="smooth fit")
#             
#        plt.show()

    def plot_tyoton(self,additional_income_tax,osuus,dire=None,label1=None,label2=None,xlabel='Tulovero [%-yks]',percent_scale=True,ylabel='Työttömien osuus väestöstä [%-yks]'):
        self.plot_osuus(additional_income_tax,osuus,label1=label1,xlabel=xlabel,ylabel=ylabel,dire=dire,percent_scale=percent_scale)

    def plot_osatyo(self,additional_income_tax,osuus,dire=None,label1=None,label2=None,xlabel='Tulovero [%-yks]',percent_scale=True,ylabel='Osatyön osuus [%-yks]'):
        self.plot_osuus(additional_income_tax,osuus,label1=label1,xlabel=xlabel,ylabel=ylabel,dire=dire,percent_scale=percent_scale)
#         if percent_scale:
#             scale=100
#         else:
#             scale=1
#     
#         fig,ax=plt.subplots()
#         ax.plot(scale*additional_income_tax,100*osuus,label=label1)
#         plt.title('Osatyössä olevien osuus')
#         #if ref_additional_tax is not None:
#         #    ax.plot(scale*ref_additional_tax,ref_mean_rew,label=label2)
#         ax.legend()
#         ax.set_ylabel(ylabel)
#         ax.set_xlabel(xlabel)
#         if dire is not None:
#             plt.savefig(dire+'osatyo '+ylabel+'.eps', format='eps')
#             plt.savefig(dire+'osatyo '+ylabel+'.png', format='png')
#             print('dire',dire)
#             
# #         a=np.min(scale*additional_income_tax)
# #         b=np.max(scale*additional_income_tax)
# #         u = smoothfit.fit1d(scale*additional_income_tax,el, a, b, 1000, degree=1, lmbda=1.0e-6)
# #         x = numpy.linspace(a, b, 201)
# #         vals = [u(xx) for xx in x]
# #         plt.plot(x, vals, "-", label="smooth fit")
# #             
#         plt.show()


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
            ax.plot(scale*ref_additional_tax,ref_mean_rew,label=label2)
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
            ax.plot(scale*ref_additional_tax,ref_mean_ps,label=label2)
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
            ax.plot(scale*ref_additional_tax,ref_mean_verot,label=label2)
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
            ax.plot(scale*ref_additional_tax,ref_mean_htv,label=label2)
        plt.title('Työnteko')
        ax.legend()
        ax.set_ylabel('Henkilöitä')
        ax.set_xlabel(xlabel)
        if dire is not None:
            plt.savefig(dire+'htv.eps', format='eps')
            plt.savefig(dire+'htv.png', format='png')
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(scale*additional_tax,mean_muut,label=label1)
        if ref_additional_tax is not None:
            ax.plot(scale*ref_additional_tax,ref_mean_muut,label=label2)
        ax.legend()
        ax.set_ylabel('Muut tulot [euroa]')
        ax.set_xlabel(xlabel)
        if dire is not None:
            plt.savefig(dire+'muut.eps', format='eps')
            plt.savefig(dire+'muut.png', format='png')

        plt.title('Tarvittavat muut tulot')
        plt.show()

    def plot_tyossa(self,additional_tax,mean_tyossa,mean_htv,ref_mean_htv=None,ref_additional_tax=None,label1='tyossa',label2='',xlabel='',dire=None,percent_scale=True):
        if percent_scale:
            scale=100
        else:
            scale=1

        fig,ax=plt.subplots()
        ax.plot(scale*additional_tax,mean_htv,label='htv')
        ax.plot(scale*additional_tax,mean_tyossa,label='tyossa')
        if ref_additional_tax is not None:
            ax.plot(scale*ref_additional_tax,ref_mean_htv,label=label2)
        plt.title('Työnteko')
        ax.legend()
        ax.set_ylabel('Henkilötyövuodet')
        ax.set_xlabel(xlabel)
        if dire is not None:
            plt.savefig(dire+'tyossa.eps', format='eps')
            plt.savefig(dire+'tyossa.png', format='png')
        plt.show()
    

    def plot_total(self,additional_tax,total_rew,total_verot,total_htv,total_ps,dire=None,xlabel=None,percent_scale=False):
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

    def plot_verokiila(self,additional_tax,kiila,xlabel=None,dire=None,ref_additional_tax=None,ref_kiila=None):
        fig,ax=plt.subplots()
        ax.plot(100*additional_tax,100*kiila)
        if ref_additional_tax is not None:
            ax.plot(scale*ref_additional_tax,ref_kiila,label=label2)
        ax.set_ylabel('Verokiila [%]')
        ax.set_xlabel('Lisävero [%-yks]')
        if dire is not None:
            plt.savefig(dire+'verot_kaikki.png', format='png')
        plt.show()


    def save_data(self,filename,additional_income_tax,total_verot,total_rew,total_ps,total_htv,
                  total_kiila,total_muut,total_tyollaste,total_tyotaste,total_tyossa,ratio_osatyo,
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
        _ = f.create_dataset('ratio_osatyo', data=ratio_osatyo, dtype=ftype)

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
        if 'ratio_osatyo' in f:
            ratio_osatyo=f.get('ratio_osatyo').value
        else:
            ratio_osatyo=additional_income_tax*0
    
        osuus_vero=f.get('osuus_vero').value
        osuus_kunnallisvero=f.get('osuus_kunnallisvero').value
        osuus_valtionvero=f.get('osuus_valtionvero').value

        f.close()

        return additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,\
               total_tyollaste,total_tyotaste,total_tyossa,ratio_osatyo,osuus_vero,osuus_kunnallisvero,osuus_valtionvero

    def comp_means(self,filename):
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,total_tyollaste,total_tyotaste,total_tyossa,ratio_osatyo,_,_,_=self.load_data(filename)
        mean_rew=np.mean(total_rew,axis=1).reshape(-1, 1)
        mean_verot=np.mean(total_verot,axis=1).reshape(-1, 1)
        mean_ps=np.mean(total_ps,axis=1).reshape(-1, 1)
        mean_htv=np.mean(total_htv,axis=1).reshape(-1, 1)
        mean_muut=np.mean(total_muut,axis=1).reshape(-1, 1)
        mean_kiila=np.mean(total_kiila,axis=1).reshape(-1, 1)
        mean_tyossa=np.mean(total_tyossa,axis=1).reshape(-1, 1)
        mean_ratio_osatyo=np.mean(ratio_osatyo,axis=1).reshape(-1, 1)
        mean_tyotaste=np.mean(total_tyotaste,axis=1).reshape(-1, 1)

        return mean_rew,mean_verot,mean_ps,mean_htv,mean_muut,mean_kiila,mean_tyossa,mean_ratio_osatyo,mean_tyotaste

    def get_refstats(self,filename,scale=None):
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,_,_,_=\
            self.load_data(filename)
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


    def comp_stats(self,taxresults,additional,datafile,repeats,minimal=False,mortality=False,perustulo=False,
                    randomness=True,pinkslip=True,plotdebug=False,start_repeat=0):
        k=0
        total_verot=np.zeros((additional.shape[0],repeats))
        total_rew=np.zeros((additional.shape[0],repeats))
        total_ps=np.zeros((additional.shape[0],repeats))
        total_htv=np.zeros((additional.shape[0],repeats))
        total_tyollaste=np.zeros((additional.shape[0],repeats))
        total_tyotaste=np.zeros((additional.shape[0],repeats))
        total_muut_tulot=np.zeros((additional.shape[0],repeats))
        total_kiila=np.zeros((additional.shape[0],repeats))
        total_tyossa=np.zeros((additional.shape[0],repeats))
        ratio_osatyo=np.zeros((additional.shape[0],repeats))

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
            for repeat in range(start_repeat,repeats):
                num=int(np.round(100*tax))
                print('computing extra tax {} repeat {}'.format(tax,repeat))
                results=taxresults+'_{}_{}'.format(num,repeat)
                print(results)
                r,qps,qv,h,muut,kiila,tyoll,tyot,lkm,osatyo_osuus=cc.render_laffer(load=results)
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
                ratio_osatyo[k,repeat]=osatyo_osuus
                osuus_vero[k,repeat,:]=vosuus
                osuus_kunnallisvero[k,repeat,:]=kvosuus
                osuus_valtionvero[k,repeat,:]=vvosuus
            k=k+1
            
        #self.comp_elasticity(additional,total_htv)

        self.save_data(datafile,additional,total_verot,total_rew,total_ps,total_htv,
                  total_kiila,total_muut_tulot,total_tyollaste,total_tyotaste,total_tyossa,
                  ratio_osatyo,osuus_vero,osuus_kunnallisvero,osuus_valtionvero)

    def comp_elasticity(self,x,y):
        xl=x.shape[0]
        yl=x.shape[0]    
        el=np.zeros((xl,1))
        #print(x,y)
    
        for k in range(1,xl-2):
            #dx=(-x[k+1]+x[k-1])/(2*(1-x[k]))
            dx=(-x[k+1]+x[k-1])/(2*x[k])
            dy=(y[k]-y[k-1])/y[k]        
            el[k]=dy/dx
            
            print('{}: {} vs {}'.format(k,(x[k]-x[k-1]),dy))
        
        return el    
    
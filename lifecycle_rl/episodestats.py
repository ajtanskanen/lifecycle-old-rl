'''

    episodestats.py

    implements statistic that are used in producing employment statistics for the
    lifecycle model

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

locale.setlocale(locale.LC_ALL, 'fi_FI')

class EpisodeStats():
    def __init__(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year=2018,version=2):
        self.version=version
        self.reset(timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year)
        print('version',version)

    def reset(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year,version=None):
        self.min_age=min_age
        self.max_age=max_age
        self.min_retirementage=min_retirementage
        self.minimal=minimal
        
        if version is not None:
            self.version=version
            
        self.n_employment=n_emps
        self.n_time=n_time
        self.timestep=timestep # 0.25 = 3kk askel
        self.inv_timestep=int(np.round(1/self.timestep)) # pitää olla kokonaisluku
        self.n_pop=n_pop
        self.year=year
        self.env=env
        self.reaalinen_palkkojenkasvu=0.016
        
        if self.minimal:
            self.version=0
        
        if self.version==0:
            self.n_groups=1
        else:
            self.n_groups=6
            
        n_emps=self.n_employment
        self.empstate=np.zeros((self.n_time,n_emps))
        self.gempstate=np.zeros((self.n_time,n_emps,self.n_groups))
        self.deceiced=np.zeros((self.n_time,1))
        self.alive=np.zeros((self.n_time,1))
        self.galive=np.zeros((self.n_time,self.n_groups))
        self.rewstate=np.zeros((self.n_time,n_emps))
        self.poprewstate=np.zeros((self.n_time,self.n_pop))
        self.salaries_emp=np.zeros((self.n_time,n_emps))
        self.actions=np.zeros((self.n_time,self.n_pop))
        self.popempstate=np.zeros((self.n_time,self.n_pop))
        self.popunemprightleft=np.zeros((self.n_time,self.n_pop))
        self.popunemprightused=np.zeros((self.n_time,self.n_pop))
        self.tyoll_distrib_bu=np.zeros((self.n_time,self.n_pop))
        self.unemp_distrib_bu=np.zeros((self.n_time,self.n_pop))
        self.siirtyneet=np.zeros((self.n_time,n_emps))
        self.pysyneet=np.zeros((self.n_time,n_emps))
        self.salaries=np.zeros((self.n_time,self.n_pop))
        self.aveV=np.zeros((self.n_time,self.n_pop))
        self.time_in_state=np.zeros((self.n_time,n_emps))
        self.stat_tyoura=np.zeros((self.n_time,n_emps))
        self.stat_toe=np.zeros((self.n_time,n_emps))
        self.stat_pension=np.zeros((self.n_time,n_emps))
        self.stat_paidpension=np.zeros((self.n_time,n_emps))
        self.out_of_work=np.zeros((self.n_time,n_emps))
        self.stat_unemp_len=np.zeros((self.n_time,self.n_pop))
        self.stat_wage_reduction=np.zeros((self.n_time,n_emps))
        self.infostats_taxes=np.zeros((self.n_time,1))
        self.infostats_etuustulo=np.zeros((self.n_time,1))
        self.infostats_perustulo=np.zeros((self.n_time,1))
        self.infostats_palkkatulo=np.zeros((self.n_time,1))
        self.infostats_ansiopvraha=np.zeros((self.n_time,1))
        self.infostats_asumistuki=np.zeros((self.n_time,1))
        self.infostats_valtionvero=np.zeros((self.n_time,1))
        self.infostats_kunnallisvero=np.zeros((self.n_time,1))
        self.infostats_ptel=np.zeros((self.n_time,1))
        self.infostats_tyotvakmaksu=np.zeros((self.n_time,1))
        self.infostats_tyoelake=np.zeros((self.n_time,1))
        self.infostats_kokoelake=np.zeros((self.n_time,1))
        self.infostats_opintotuki=np.zeros((self.n_time,1))
        self.infostats_isyyspaivaraha=np.zeros((self.n_time,1))
        self.infostats_aitiyspaivaraha=np.zeros((self.n_time,1))
        self.infostats_kotihoidontuki=np.zeros((self.n_time,1))
        self.infostats_sairauspaivaraha=np.zeros((self.n_time,1))
        self.infostats_toimeentulotuki=np.zeros((self.n_time,1))
        self.infostats_tulot_netto=np.zeros((self.n_time,1))
        self.infostats_pinkslip=np.zeros((self.n_time,n_emps))
        self.infostats_chilren18_emp=np.zeros((self.n_time,n_emps))
        self.infostats_chilren7_emp=np.zeros((self.n_time,n_emps))
        self.infostats_chilren18=np.zeros((self.n_time,1))
        self.infostats_chilren7=np.zeros((self.n_time,1))
        self.infostats_tyelpremium=np.zeros((self.n_time,self.n_pop))
        self.infostats_paid_tyel_pension=np.zeros((self.n_time,self.n_pop))
        self.infostats_irr=np.zeros((self.n_pop,1))
        self.infostats_npv0=np.zeros((self.n_pop,1))
        

    def add(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None): 
        
        if self.version==0:
            emp,_,_,a,_,_=self.env.state_decode(state) # current employment state
            newemp,_,newsal,a2,tis,next_wage=self.env.state_decode(newstate)
            g=0
            bu=0
        elif self.version==1:
            # v1
            emp,_,_,_,a,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
            newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,ura,oof,bu,wr,p=self.env.state_decode(newstate)
        else: 
            # v2
            emp,_,_,_,a,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
            newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,ura,bu,wr,pr,upr,uw,uwr,c3,c7,c18,unemp_left,aa=self.env.state_decode(newstate)
    
        t=int(np.round((a2-self.min_age)*self.inv_timestep))-1
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            if a2>self.min_retirementage and newemp==3:
                newemp=2
            if self.version>0:
                self.empstate[t,newemp]+=1
                self.alive[t]+=1
                self.rewstate[t,newemp]+=r
            
                self.poprewstate[t,n]=r
                self.actions[t,n]=act
                self.popempstate[t,n]=newemp
                self.salaries[t,n]=newsal
                self.salaries_emp[t,newemp]+=newsal
                self.time_in_state[t,newemp]+=tis
                self.infostats_pinkslip[t,newemp]+=pink
                self.gempstate[t,newemp,g]+=1
                self.stat_wage_reduction[t,newemp]+=wr
                self.galive[t,g]+=1
                self.stat_tyoura[t,newemp]+=ura
                self.stat_toe[t,newemp]+=toe
                self.stat_pension[t,newemp]+=newpen
                self.stat_paidpension[t,newemp]+=paidpens
                self.stat_unemp_len[t,n]=tis
                self.popunemprightleft[t,n]=-self.env.unempright_left(newemp,tis,bu,a2,ura)
                self.popunemprightused[t,n]=bu
                if q is not None:
                    self.infostats_taxes[t]+=q['verot']*self.timestep*12
                    self.infostats_etuustulo[t]+=q['etuustulo_brutto']*self.timestep*12
                    self.infostats_perustulo[t]+=q['perustulo']*self.timestep*12
                    self.infostats_palkkatulo[t]+=q['palkkatulot']*self.timestep*12
                    self.infostats_ansiopvraha[t]+=q['ansiopvraha']*self.timestep*12
                    self.infostats_asumistuki[t]+=q['asumistuki']*self.timestep*12
                    self.infostats_valtionvero[t]+=q['valtionvero']*self.timestep*12
                    self.infostats_kunnallisvero[t]+=q['kunnallisvero']*self.timestep*12
                    self.infostats_ptel[t]+=q['ptel']*self.timestep*12
                    self.infostats_tyotvakmaksu[t]+=q['tyotvakmaksu']*self.timestep*12
                    self.infostats_tyoelake[t]+=q['elake_maksussa']*self.timestep*12
                    self.infostats_kokoelake[t]+=q['kokoelake']*self.timestep*12
                    self.infostats_opintotuki[t]+=q['opintotuki']*self.timestep*12
                    self.infostats_isyyspaivaraha[t]+=q['isyyspaivaraha']*self.timestep*12
                    self.infostats_aitiyspaivaraha[t]+=q['aitiyspaivaraha']*self.timestep*12
                    self.infostats_kotihoidontuki[t]+=q['kotihoidontuki']*self.timestep*12
                    self.infostats_sairauspaivaraha[t]+=q['sairauspaivaraha']*self.timestep*12
                    self.infostats_toimeentulotuki[t]+=q['toimtuki']*self.timestep*12
                    self.infostats_tulot_netto[t]+=q['kateen']*self.timestep*12
                    self.infostats_tyelpremium[t,n]=q['tyel_kokomaksu']*self.timestep*12
                    self.infostats_paid_tyel_pension[t,n]=q['puhdas_tyoelake']*self.timestep*12
                    self.infostats_npv0[n]=q['multiplier']
            else:
                self.empstate[t,newemp]+=1
                self.alive[t]+=1
                self.rewstate[t,newemp]+=r
                
                self.poprewstate[t,n]=r
                self.actions[t,n]=act
                self.popempstate[t,n]=newemp
                self.salaries[t,n]=newsal
                self.salaries_emp[t,newemp]+=newsal
                self.time_in_state[t,newemp]+=tis
                if self.version>0:
                    self.gempstate[t,newemp,g]+=1
                    self.stat_wage_reduction[t,newemp]+=wr
                    self.galive[t,g]+=1
                    self.stat_tyoura[t,newemp]+=ura
                    self.stat_toe[t,newemp]+=toe
                    self.stat_pension[t,newemp]+=newpen
                    self.stat_paidpension[t,newemp]+=paidpens
                    self.stat_unemp_len[t,n]=tis
                    self.popunemprightleft[t,n]=0
                    self.popunemprightused[t,n]=0
            
            if aveV is not None:
                self.aveV[t,n]=aveV

            if not emp==newemp:
                self.siirtyneet[t,emp]+=1
            else:
                self.pysyneet[t,emp]+=1
        elif newemp<0:
            self.deceiced[t]+=1
    
    def map_age(self,age,start_zero=False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)
    
    def episodestats_exit(self):
        plt.close(self.episode_fig)
        
    def comp_gini(self):
        '''
        Laske Gini-kerroin populaatiolle
        '''
        income=np.sort(self.infostats_tulot_netto,axis=None)
        n=len(income)
        L=np.arange(n,0,-1)
        A=np.sum(L*income)/np.sum(income)
        G=(n+1-2*A)/2
        return G

    def comp_annual_irr(self,npv,premium,pension,empstate,doprint=False):
        k=0
        ext=int(np.ceil(npv))
        cashflow=-premium+pension
        x=np.zeros(cashflow.shape[0]+ext)
        
        #print(npv,tyel_premium,paid_pension)
        
        # indeksointi puuttuu
        x[:cashflow.shape[0]]=cashflow
        if npv>0:
            x[cashflow.shape[0]-1:]=(cashflow[-2])
            
        y=np.zeros(int(np.ceil(x.shape[0]/4)))
        for k in range(y.shape[0]):
            y[k]=np.sum(x[4*k:4*k+4])
        irri=npf.irr(y)*100
        
        #if np.isnan(irri):
        #    if np.sum(pension)<0.1 and np.sum(empstate[0:self.map_age(63)]==15)>0: # vain maksuja, joista ei saa tuottoja, joten tappio 100%
        #        irri=-100
        
        if irri<0.01 and doprint:
            print('---------\nirri {}\nnpv {}\nx {}\ny {}\nprem {}\npens {}\nemps {}\n---------\n'.format(irri,npv,x,y,premium,pension,empstate))

        if irri>100 and doprint:
            print('---------\nirri {}\nnpv {}\nx {}\ny {}\nprem {}\npens {}\nemps {}\n---------\n'.format(irri,npv,x,y,premium,pension,empstate))
            
        if np.isnan(irri) and doprint:
            print('---------\nirri {}\nnpv {}\nx {}\ny {}\nprem {}\npens {}\nemps {}\n---------\n'.format(irri,npv,x,y,premium,np.sum(pension),empstate))
            #print('---------\nirri {}\nnpv {}\n\ny {}\nprem {}\npens {}\nemps {}\n---------\n'.format(irri,npv,x,y,premium,np.sum(pension),np.sum(empstate==15)))
            
        if irri<-50 and doprint:
            print('---------\nirri {}\nnpv {}\nx {}\ny {}\nprem {}\npens {}\nemps {}\n---------\n'.format(irri,npv,x,y,premium,pension,empstate))
            
        return irri
        
    def comp_irr(self):
        '''
        Laskee sisäisen tuottoasteen (IRR)
        Indeksointi puuttuu npv:n osalta
        Tuloksiin lisättävä inflaatio+palkkojen reaalikasvu = palkkojen nimellinen kasvu
        '''
        for k in range(self.n_pop):
            self.infostats_irr[k]=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(self.infostats_npv0[k,0],self.infostats_tyelpremium[:,k],self.infostats_paid_tyel_pension[:,k],self.popempstate[:,k])

    def comp_aggirr(self):
        '''
        Laskee aggregoidun sisäisen tuottoasteen (IRR)
        Indeksointi puuttuu npv:n osalta
        Tuloksiin lisättävä inflaatio+palkkojen reaalikasvu = palkkojen nimellinen kasvu
        '''
        maxnpv=np.max(self.infostats_npv0)
        agg_premium=np.sum(self.infostats_tyelpremium,axis=1)
        agg_pensions=np.sum(self.infostats_paid_tyel_pension,axis=1)
        agg_irr=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(maxnpv,agg_premium,agg_pensions,self.popempstate[:,0])
        x=np.zeros(self.infostats_paid_tyel_pension.shape[0]+int(np.ceil(maxnpv)))
        
        for k in range(self.n_pop):
            if np.sum(self.popempstate[0:self.map_age(63),k]==15)<1: # ilman kuolleita
                n=int(np.ceil(self.infostats_npv0[k,0]))
                cashflow=-self.infostats_tyelpremium[:,k]+self.infostats_paid_tyel_pension[:,k]
        
                # indeksointi puuttuu
                cfn=cashflow.shape[0]
                x[:cfn]+=cashflow            
                if n>0:
                    x[cfn-1:cfn+n-1]+=(cashflow[-2])
            
        y=np.zeros(int(np.ceil(x.shape[0]/4)))
        for k in range(y.shape[0]):
            y[k]=np.sum(x[4*k:4*k+4])
        irri=npf.irr(y)*100        
    
        
        print('aggregate irr {}'.format(agg_irr))

    def plot_ratiostats(self,t):
        '''
        Tee kuvia tuloksista
        '''
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.set_xlabel('palkat')
        ax.set_ylabel('freq')
        ax.hist(self.salaries[t,:])
        plt.show()
        fig,ax=plt.subplots()
        ax.set_xlabel('aika')
        ax.set_ylabel('palkat')
        meansal=np.mean(self.salaries,axis=1)
        stdsal=np.std(self.salaries,axis=1)
        ax.plot(x,meansal)
        ax.plot(x,meansal+stdsal)
        ax.plot(x,meansal-stdsal)
        plt.show()
        
    def comp_unemp_durations(self,popempstate=None,popunemprightused=None,putki=True,\
            tmtuki=False,laaja=False,outsider=False,ansiosid=True,tyott=False,kaikki=False,\
            return_q=True,max_age=100):
        '''
        Poikkileikkaushetken työttömyyskestot
        '''
        unempset=[]
        
        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]
        if laaja:
            unempset=[0,4,11,13]
        if kaikki:
            unempset=[0,2,3,4,5,6,7,8,9,11,12,13,14]
            
        unempset=set(unempset)
        
        if popempstate is None:
            popempstate=self.popempstate
            
        if popunemprightused is None:
            popunemprightused=self.popunemprightused

        keskikesto=np.zeros((5,5)) # 20-29, 30-39, 40-49, 50-59, 60-69, vastaa TYJin tilastoa
        n=np.zeros(5)
        
        for k in range(self.n_pop):
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if popempstate[t,k] in unempset:
                        if age<29:
                            l=0
                        elif age<39:
                            l=1
                        elif age<49:
                            l=2
                        elif age<59:
                            l=3
                        else:
                            l=4
                            
                        n[l]+=1
                        if self.popunemprightused[t,k]<=0.51:
                            keskikesto[l,0]+=1
                        elif self.popunemprightused[t,k]<=1.01:
                            keskikesto[l,1]+=1
                        elif self.popunemprightused[t,k]<=1.51:
                            keskikesto[l,2]+=1
                        elif self.popunemprightused[t,k]<=2.01:
                            keskikesto[l,3]+=1
                        else:
                            keskikesto[l,4]+=1

        for k in range(5):
            keskikesto[k,:] /= n[k]
            
        if return_q:
            return self.empdur_to_dict(keskikesto)
        else:
            return keskikesto

    def empdur_to_dict(self,empdur):
        q={}
        q['20-29']=empdur[0,:]
        q['30-39']=empdur[1,:]
        q['40-49']=empdur[2,:]
        q['50-59']=empdur[3,:]
        q['60-65']=empdur[4,:]
        return q
            
    def comp_unemp_durations_v2(self,popempstate=None,putki=True,tmtuki=False,laaja=False,\
            outsider=False,ansiosid=True,tyott=False,kaikki=False,\
            return_q=True,max_age=100):
        '''
        Poikkileikkaushetken työttömyyskestot
        Tässä lasketaan tulos tiladatasta, jolloin kyse on viimeisimmän jakson kestosta
        '''
        unempset=[]
        
        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]
        if laaja:
            unempset=[0,4,11,13]
        if kaikki:
            unempset=[0,2,3,4,5,6,7,8,9,11,12,13,14]
            
        unempset=set(unempset)
        
        if popempstate is None:
            popempstate=self.popempstate

        keskikesto=np.zeros((5,5)) # 20-29, 30-39, 40-49, 50-59, 60-69, vastaa TYJin tilastoa
        n=np.zeros(5)
        
        for k in range(self.n_pop):
            prev_state=popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if popempstate[t,k]!=prev_state:
                        if prev_state in unempset and popempstate[t,k] not in unempset:
                            prev_state=popempstate[t,k]
                            duration=(t-prev_trans)*self.timestep
                            prev_trans=t
                            
                            if age<29:
                                l=0
                            elif age<39:
                                l=1
                            elif age<49:
                                l=2
                            elif age<59:
                                l=3
                            else:
                                l=4
                            
                            n[l]+=1
                            if duration<=0.51:
                                keskikesto[l,0]+=1
                            elif duration<=1.01:
                                keskikesto[l,1]+=1
                            elif duration<=1.51:
                                keskikesto[l,2]+=1
                            elif duration<=2.01:
                                keskikesto[l,3]+=1
                            else:
                                keskikesto[l,4]+=1                            
                        elif prev_state not in unempset and popempstate[t,k] in unempset:
                            prev_trans=t
                            prev_state=popempstate[t,k]
                        else: # some other state
                            prev_state=popempstate[t,k]
                            prev_trans=t
                
        for k in range(5):
            keskikesto[k,:] /= n[k]
            
        if return_q:
            return self.empdur_to_dict(keskikesto)
        else:
            return keskikesto
        
    def comp_virrat(self,popempstate=None,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,kaikki=False,max_age=100):
        tyoll_virta=np.zeros((self.n_time,1))
        tyot_virta=np.zeros((self.n_time,1))
        unempset=[]
        empset=[]
        
        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]
        if laaja:
            unempset=[0,4,11,13]
        if kaikki:
            unempset=[0,2,3,4,5,6,7,8,9,11,12,13,14]
            
        empset=set([1,10])
        unempset=set(unempset)
        
        if popempstate is None:
            popempstate=self.popempstate
        
        for k in range(self.n_pop):
            prev_state=popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if popempstate[t,k]!=prev_state:
                        if prev_state in unempset and popempstate[t,k] in empset:
                            tyoll_virta[t]+=1
                            prev_state=popempstate[t,k]
                        elif prev_state in empset and popempstate[t,k] in unempset:
                            tyot_virta[t]+=1
                            prev_state=popempstate[t,k]
                        else: # some other state
                            prev_state=popempstate[t,k]

        return tyoll_virta,tyot_virta
        
    def comp_tyollistymisdistribs(self,popempstate=None,popunemprightleft=None,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,max_age=100):
        tyoll_distrib=[]
        tyoll_distrib_bu=[]
        unempset=[]
        
        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]
            
        if laaja:
            unempset=[0,4,11,13]
            
        empset=set([1,10])
        unempset=set(unempset)
        
        if popempstate is None or popunemprightleft is None:
            popempstate=self.popempstate
            popunemprightleft=self.popunemprightleft
        
        for k in range(self.n_pop):
            prev_state=popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if popempstate[t,k]!=prev_state:
                        if prev_state in unempset and popempstate[t,k] in empset:
                            tyoll_distrib.append((t-prev_trans)*self.timestep)
                            tyoll_distrib_bu.append(popunemprightleft[t,k])
                            prev_state=popempstate[t,k]
                            prev_trans=t
                        else: # some other state
                            prev_state=popempstate[t,k]
                            prev_trans=t
                    
        return tyoll_distrib,tyoll_distrib_bu

    def comp_empdistribs(self,popempstate=None,popunemprightleft=None,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,max_age=100):
        unemp_distrib=[]
        unemp_distrib_bu=[]
        emp_distrib=[]
        unempset=[]
        
        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]
            
        if laaja:
            unempset=[0,4,11,13]
            
        if popempstate is None or popunemprightleft is None:
            popempstate=self.popempstate
            popunemprightleft=self.popunemprightleft
        
        empset=set([1,10])
        unempset=set(unempset)
        
        for k in range(self.n_pop):
            prev_state=popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if self.popempstate[t,k]!=prev_state:
                        if prev_state in unempset and popempstate[t,k] not in unempset:
                            unemp_distrib.append((t-prev_trans)*self.timestep)
                            unemp_distrib_bu.append(popunemprightleft[t,k])
                            
                            prev_state=popempstate[t,k]
                            prev_trans=t
                        elif prev_state in empset and popempstate[t,k] not in unempset:
                            emp_distrib.append((t-prev_trans)*self.timestep)
                            prev_state=popempstate[t,k]
                            prev_trans=t
                        else: # some other state
                            prev_state=popempstate[t,k]
                            prev_trans=t
                    
        return unemp_distrib,emp_distrib,unemp_distrib_bu
        
    def empdist_stat(self):
        ratio=np.array([1,0.287024901703801,0.115508955875928,0.0681083442551332,0.0339886413280909,0.0339886413280909,0.0114460463084316,0.0114460463084316,0.0114460463084316,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206])
        
        return ratio
        
    def plot_empdistribs(self,emp_distrib):
        fig,ax=plt.subplots()
        ax.set_xlabel('työsuhteen pituus [v]')
        ax.set_ylabel('freq')
        ax.set_yscale('log')
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1        
        x=np.linspace(0,max_time,nn_time)
        scaled,x2=np.histogram(emp_distrib,x)
        scaled=scaled/np.sum(emp_distrib)
        #ax.hist(emp_distrib)
        ax.bar(x2[1:-1],scaled[1:],align='center')
        plt.show()
        
    def plot_compare_empdistribs(self,emp_distrib,emp_distrib2,label='vaihtoehto'):
        fig,ax=plt.subplots()
        ax.set_xlabel('työsuhteen pituus [v]')
        ax.set_ylabel('skaalattu taajuus')
        ax.set_yscale('log')
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1        
        x=np.linspace(0,max_time,nn_time)
        scaled,x2=np.histogram(emp_distrib,x)
        scaled=scaled/np.sum(emp_distrib)
        x=np.linspace(0,max_time,nn_time)
        scaled3,x3=np.histogram(emp_distrib2,x)
        scaled3=scaled3/np.sum(emp_distrib2)
        
        ax.plot(x3[:-1],scaled3,label='perus')
        ax.plot(x2[:-1],scaled,label=label)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
        plt.show()
        
    def plot_vlines_unemp(self,point=0):
        axvcolor='gray'
        lstyle='--'
        plt.axvline(x=300/(12*21.5),ls=lstyle,color=axvcolor)
        plt.text(310/(12*21.5),point,'300',rotation=90)        
        plt.axvline(x=400/(12*21.5),ls=lstyle,color=axvcolor)
        plt.text(410/(12*21.5),point,'400',rotation=90)        
        plt.axvline(x=500/(12*21.5),ls=lstyle,color=axvcolor)
        plt.text(510/(12*21.5),point,'500',rotation=90)                    
        
    def plot_tyolldistribs(self,emp_distrib,tyoll_distrib,tyollistyneet=True,max=10,figname=None):
        max_time=55
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        if not tyollistyneet:
            scaled=scaled0
            x2=x0
        else:
            scaled,x2=np.histogram(tyoll_distrib,x)
        jaljella=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien kumulatiivinen summa
        scaled=scaled/jaljella
        
        fig,ax=plt.subplots()
        ax.set_xlabel('työttömyysjakson pituus [v]')
        if tyollistyneet:
            ax.set_ylabel('työllistyneiden osuus')
            point=0.5
        else:
            ax.set_ylabel('pois siirtyneiden osuus')
            point=0.9
        self.plot_vlines_unemp(point)
        ax.plot(x2[1:-1],scaled[1:])
        #ax.bar(x2[1:-1],scaled[1:],align='center',width=self.timestep)
        plt.xlim(0,max)
        if figname is not None:
            plt.savefig(figname+'tyollistyneetdistrib.eps', format='eps')
            
        plt.show()        

    def plot_tyolldistribs_both(self,emp_distrib,tyoll_distrib,max=10,figname=None):
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        scaled=scaled0
        scaled_tyoll,x2=np.histogram(tyoll_distrib,x)
            
        jaljella=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien summa
        scaled=scaled/jaljella
        jaljella_tyoll=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien summa
        scaled_tyoll=scaled_tyoll/jaljella_tyoll
        fig,ax=plt.subplots()
        ax.set_xlabel('työttömyysjakson pituus [v]')
        point=0.6
        self.plot_vlines_unemp(point)
        ax.plot(x2[1:-1],scaled[1:],label='pois siirtyneiden osuus')
        ax.plot(x2[1:-1],scaled_tyoll[1:],label='työllistyneiden osuus')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.legend()
        ax.set_ylabel('pois siirtyneiden osuus')

        plt.xlim(0,max)
        plt.ylim(0,0.8)
        if figname is not None:
            plt.savefig(figname+'tyolldistribs.eps', format='eps')
        plt.show()        

    def plot_tyolldistribs_both_bu(self,emp_distrib,tyoll_distrib,max=2):
        max_time=4
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(-max_time,0,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        scaled=scaled0
        scaled_tyoll,x2=np.histogram(tyoll_distrib,x)
            
        jaljella=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien summa
        #jaljella=np.cumsum(scaled0)
        scaled=scaled/jaljella
        jaljella_tyoll=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien summa
        #jaljella_tyoll=np.cumsum(scaled0)
        scaled_tyoll=scaled_tyoll/jaljella_tyoll
        fig,ax=plt.subplots()
        ax.set_xlabel('aika ennen ansiopäivärahaoikeuden loppua [v]')
        point=0.6
        #self.plot_vlines_unemp(point)
        ax.plot(x2[1:-1],scaled[1:],label='pois siirtyneiden osuus')
        ax.plot(x2[1:-1],scaled_tyoll[1:],label='työllistyneiden osuus')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_ylabel('pois siirtyneiden osuus')

        plt.xlim(-max,0)
        #plt.ylim(0,0.8)
        plt.show()        

    def plot_compare_tyolldistribs(self,emp_distrib1,tyoll_distrib1,emp_distrib2,
                tyoll_distrib2,tyollistyneet=True,max=4,label1='perus',label2='vaihtoehto',
                figname=None):
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)

        # data1
        scaled01,x0=np.histogram(emp_distrib1,x)
        if not tyollistyneet:
            scaled1=scaled01
            x1=x0
        else:
            scaled1,x1=np.histogram(tyoll_distrib1,x)
        jaljella1=np.cumsum(scaled01[::-1])[::-1] # jäljellä olevien summa
        scaled1=scaled1/jaljella1
        
        # data2
        scaled02,x0=np.histogram(emp_distrib2,x)
        if not tyollistyneet:
            scaled2=scaled02
            x2=x0
        else:
            scaled2,x2=np.histogram(tyoll_distrib2,x)
        jaljella2=np.cumsum(scaled02[::-1])[::-1] # jäljellä olevien summa
        scaled2=scaled2/jaljella2
        
        fig,ax=plt.subplots()
        ax.set_xlabel('työttömyysjakson pituus [v]')
        if tyollistyneet:
            ax.set_ylabel('työllistyneiden osuus')
        else:
            ax.set_ylabel('pois siirtyneiden osuus')
        self.plot_vlines_unemp()
        ax.plot(x2[1:-1],scaled2[1:],label=label2)
        ax.plot(x1[1:-1],scaled1[1:],label=label1)
        #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.legend()
        plt.xlim(0,max)
        if figname is not None:
            plt.savefig(figname+'comp_tyollistyneetdistrib.eps', format='eps')
        
        plt.show()        
                
    def plot_unempdistribs(self,unemp_distrib,max=10,figname=None,miny=None,maxy=None):
        #fig,ax=plt.subplots()
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled,x2=np.histogram(unemp_distrib,x)
        scaled=scaled/np.sum(unemp_distrib)
        fig,ax=plt.subplots()
        self.plot_vlines_unemp(0.6)
        ax.set_xlabel('työttömyysjakson pituus [v]')
        ax.set_ylabel('skaalattu taajuus')
        ax.plot(x[:-1],scaled)
        ax.set_yscale('log')
        plt.xlim(0,max)
        if miny is not None:
            plt.ylim(miny,maxy)
        if figname is not None:
            plt.savefig(figname+'unempdistribs.eps', format='eps')
        
        plt.show()   

    def plot_saldist(self,t=0,sum=False,all=False,n=10):
        if all:
            fig,ax=plt.subplots()
            for t in range(1,self.n_time-1,5):
                scaled,x=np.histogram(self.salaries[t,:])
                x2=0.5*(x[1:]+x[0:-1])
                ax.plot(x2,scaled,label=t)
            plt.legend()
            plt.show()
        else:
            if sum:
                scaled,x=np.histogram(np.sum(self.salaries,axis=0))
                x2=0.5*(x[1:]+x[0:-1])
                plt.plot(x2,scaled)
            else:
                fig,ax=plt.subplots()
                for t in range(t,t+n,1):
                    scaled,x=np.histogram(self.salaries[t,:])
                    x2=0.5*(x[1:]+x[0:-1])
                    ax.plot(x2,scaled,label=t)
                plt.legend()
                plt.show()

    def plot_rewdist(self,t=0,sum=False,all=False):
        if all:
            fig,ax=plt.subplots()
            for t in range(1,self.n_time-1,5):
                scaled,x=np.histogram(self.poprewstate[t,:])
                x2=0.5*(x[1:]+x[0:-1])
                ax.plot(x2,scaled,label=t)
            plt.legend()
            plt.show()
        else:
            if sum:
                scaled,x=np.histogram(np.sum(self.poprewstate,axis=0))
                x2=0.5*(x[1:]+x[0:-1])
                plt.plot(x2,scaled)
            else:
                fig,ax=plt.subplots()
                for t in range(t,t+10,1):
                    scaled,x=np.histogram(self.poprewstate[t,:])
                    x2=0.5*(x[1:]+x[0:-1])
                    ax.plot(x2,scaled,label=t)
                plt.legend()
                plt.show()

    def plot_unempdistribs_bu(self,unemp_distrib,max=2):
        #fig,ax=plt.subplots()
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(-max_time,0,nn_time)
        scaled,x2=np.histogram(unemp_distrib,x)
        scaled=scaled/np.abs(np.sum(unemp_distrib))
        fig,ax=plt.subplots()
        #self.plot_vlines_unemp(0.6)
        ax.set_xlabel('työttömyysjakson pituus [v]')
        ax.set_ylabel('skaalattu taajuus')
        #x3=np.flip(x[:-1])
        #ax.plot(x3,scaled)
        ax.plot(x[:-1],scaled)
        #ax.set_yscale('log')
        plt.xlim(-max,0)
        plt.show()   

    def plot_compare_unempdistribs(self,unemp_distrib1,unemp_distrib2,max=4,
            label2='none',label1='none',logy=True,diff=False,figname=None):
        #fig,ax=plt.subplots()
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(self.timestep,max_time,nn_time)
        scaled1,x1=np.histogram(unemp_distrib1,x)
        print('{} keskikesto {} v {} keskikesto {} v'.format(label1,np.mean(unemp_distrib1),label2,np.mean(unemp_distrib2)))
        print('Skaalaamaton {} lkm {} v {} lkm {} v'.format(label1,len(unemp_distrib1),label2,len(unemp_distrib2)))
        print('Skaalaamaton {} työtpäiviä yht {} v {} työtpäiviä yht {} v'.format(label1,np.sum(unemp_distrib1),label2,np.sum(unemp_distrib2)))
        #scaled=scaled/np.sum(unemp_distrib)
        scaled1=scaled1/np.sum(scaled1)
        
        scaled2,x1=np.histogram(unemp_distrib2,x)
        scaled2=scaled2/np.sum(scaled2)
        fig,ax=plt.subplots()
        if not diff:
            self.plot_vlines_unemp(0.5)
        ax.set_xlabel('Työttömyysjakson pituus [v]')
        ax.set_ylabel('Osuus')
        if diff:
            ax.plot(x[:-1],scaled1-scaled2,label=label1+'-'+label2)
        else:
            ax.plot(x[:-1],scaled2,label=label2)
            ax.plot(x[:-1],scaled1,label=label1)
        if logy and not diff:
            ax.set_yscale('log')
        if not diff:
            plt.ylim(1e-4,1.0)
        #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.legend()
        plt.xlim(0,max)
        if figname is not None:
            plt.savefig(figname+'comp_unempdistrib.eps', format='eps')
        
        plt.show()   

    def plot_compare_virrat(self,virta1,virta2,min_time=25,max_time=65,label1='perus',label2='vaihtoehto',virta_label='työllisyys',ymin=None,ymax=None):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        
        demog,demog2=self.get_demog()
        
        scaled1=virta1*demog2/self.n_pop #/self.alive
        scaled2=virta2*demog2/self.n_pop #/self.alive
        
        fig,ax=plt.subplots()
        plt.xlim(min_time,max_time)
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel(virta_label+'virta')
        ax.plot(x,scaled1,label=label1)
        ax.plot(x,scaled2,label=label2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if ymin is not None and ymax is not None:
            plt.ylim(ymin,ymax)
            
        plt.show()   

    def comp_gempratios(self,unempratio=True,gender='men'):
        if gender=='men': # men
            gempstate=np.sum(self.gempstate[:,:,0:3],axis=2)
            alive=np.zeros((self.galive.shape[0],1))
            alive[:,0]=np.sum(self.galive[:,0:3],1)
        else: # women
            gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
            alive=np.zeros((self.galive.shape[0],1))
            alive[:,0]=np.sum(self.galive[:,3:6],1)
    
        tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive,unempratio=unempratio)
        
        return tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste


    def comp_empratios(self,emp,alive,unempratio=True):
        employed=emp[:,1]
        retired=emp[:,2]
        unemployed=emp[:,0]
        
        if self.version>0:
            disabled=emp[:,3]
            piped=emp[:,4]
            mother=emp[:,5]
            dad=emp[:,6]
            kotihoidontuki=emp[:,7]
            vetyo=emp[:,9]
            veosatyo=emp[:,8]
            osatyo=emp[:,10]
            outsider=emp[:,11]
            student=emp[:,12]
            tyomarkkinatuki=emp[:,13]
            tyollisyysaste=100*(employed+osatyo+veosatyo+vetyo)/alive[:,0]
            osatyoaste=100*(osatyo+veosatyo)/(employed+osatyo+veosatyo+vetyo)
            if unempratio:
                tyottomyysaste=100*(unemployed+piped+tyomarkkinatuki)/(tyomarkkinatuki+unemployed+employed+piped+osatyo+veosatyo+vetyo)
                ka_tyottomyysaste=100*np.sum(unemployed+tyomarkkinatuki+piped)/np.sum(tyomarkkinatuki+unemployed+employed+piped+osatyo+veosatyo+vetyo)
            else:
                tyottomyysaste=100*(unemployed+piped+tyomarkkinatuki)/alive[:,0]
                ka_tyottomyysaste=100*np.sum(unemployed+tyomarkkinatuki+piped)/np.sum(alive[:,0])
        else:
            tyollisyysaste=100*(employed)/alive[:,0]
            osatyoaste=np.zeros(employed.shape)
            if unempratio:
                tyottomyysaste=100*(unemployed)/(unemployed+employed)
                ka_tyottomyysaste=100*np.sum(unemployed)/np.sum(unemployed+employed)
            else:
                tyottomyysaste=100*(unemployed)/alive[:,0]
                ka_tyottomyysaste=100*np.sum(unemployed)/np.sum(alive[:,0])

        return tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste

    def comp_L1error(self):
        
        tyollisyysaste_m,osatyoaste_m,tyottomyysaste_m,ka_tyottomyysaste=self.comp_gempratios(gender='men',unempratio=False)
        tyollisyysaste_w,osatyoaste_w,tyottomyysaste_w,ka_tyottomyysaste=self.comp_gempratios(gender='women',unempratio=False)
        emp_statsratio_m=self.emp_stats(g=1)[:-1]*100
        emp_statsratio_w=self.emp_stats(g=2)[:-1]*100
        unemp_statsratio_m=self.unemp_stats(g=1)[:-1]*100
        unemp_statsratio_w=self.unemp_stats(g=2)[:-1]*100
        
        w1=1.0
        w2=3.0
        
        L2= w1*np.sum(np.abs(emp_statsratio_m-tyollisyysaste_m[:-1])**2)+\
            w1*np.sum(np.abs(emp_statsratio_w-tyollisyysaste_w[:-1])**2)+\
            w2*np.sum(np.abs(unemp_statsratio_m-tyottomyysaste_m[:-1])**2)+\
            w2*np.sum(np.abs(unemp_statsratio_w-tyottomyysaste_w[:-1])**2)
        L2=L2/self.n_pop
            
        #print(L1,emp_statsratio_m,tyollisyysaste_m,tyollisyysaste_w,unemp_statsratio_m,tyottomyysaste_m,tyottomyysaste_w)

        print('L2 error {}'.format(L2))

        return L2

    def plot_outsider(self,printtaa=True):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*(self.empstate[:,11]+self.empstate[:,5]+self.empstate[:,6]+self.empstate[:,7])/self.alive[:,0],label='työvoiman ulkopuolella, ei opiskelija, sis. vanh.vapaat')
        emp_statsratio=100*self.outsider_stats()    
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend()
        plt.show()
        if printtaa:
            #print('yht',100*(self.empstate[:,11]+self.empstate[:,5]+self.empstate[:,6]+self.empstate[:,7])/self.alive[:,0])
            nn=np.sum(self.galive[:,3:5],1,keepdims=True)
            n=np.sum(100*(self.gempstate[:,5,3:5]+self.gempstate[:,6,3:5]+self.gempstate[:,7,3:5]),1,keepdims=True)/nn
            mn=np.sum(self.galive[:,0:2],1,keepdims=True)
            m=np.sum(100*(self.gempstate[:,5,0:2]+self.gempstate[:,6,0:2]+self.gempstate[:,7,0:2]),1,keepdims=True)/mn
            #print('naiset vv',n[1::4,0])
            #print('miehet vv',m[1::4,0])

    def plot_pinkslip(self):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*self.infostats_pinkslip[:,0]/self.empstate[:,0],label='ansiosidonnaisella')
        ax.plot(x,100*self.infostats_pinkslip[:,4]/self.empstate[:,4],label='putkessa')
        ax.plot(x,100*self.infostats_pinkslip[:,13]/self.empstate[:,13],label='työmarkkinatuella')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Irtisanottujen osuus tilassa [%]')
        ax.legend()
        plt.show()
        
    def plot_student(self):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*self.empstate[:,12]/self.alive[:,0],label='opiskelijat')
        emp_statsratio=100*self.student_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend()
        plt.show()
        
    def plot_group_student(self):
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='Opiskelijat Miehet'
                opiskelijat=np.sum(self.gempstate[:,12,0:3],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
            else:
                leg='Opiskelijat Naiset'
                opiskelijat=np.sum(self.gempstate[:,12,3:6],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
        
            opiskelijat=np.reshape(opiskelijat,(self.galive.shape[0],1))
            osuus=100*opiskelijat/alive
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,osuus,label=leg)
            
        emp_statsratio=100*self.student_stats(g=1)
        ax.plot(x,emp_statsratio,label='havainto, naiset')
        emp_statsratio=100*self.student_stats(g=2)
        ax.plot(x,emp_statsratio,label='havainto, miehet')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        
    def plot_group_disab(self):
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='TK Miehet'
                opiskelijat=np.sum(self.gempstate[:,3,0:3],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
            else:
                leg='TK Naiset'
                opiskelijat=np.sum(self.gempstate[:,3,3:6],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
        
            opiskelijat=np.reshape(opiskelijat,(self.galive.shape[0],1))
            osuus=100*opiskelijat/alive
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,osuus,label=leg)
            
        emp_statsratio=100*self.disab_stat(g=1)
        ax.plot(x,emp_statsratio,label='havainto, naiset')
        emp_statsratio=100*self.disab_stat(g=2)
        ax.plot(x,emp_statsratio,label='havainto, miehet')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def plot_emp(self,figname=None):

        tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(self.empstate,self.alive,unempratio=False)

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,tyollisyysaste,label='työllisyysaste')
        ax.plot(x,tyottomyysaste,label='työttömien osuus')
        emp_statsratio=100*self.emp_stats()
        ax.plot(x,emp_statsratio,ls='--',label='havainto')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyollisyysaste.eps', format='eps')
        plt.show()

        if self.version>0:
            fig,ax=plt.subplots()
            ax.stackplot(x,osatyoaste,100-osatyoaste,
                        labels=('osatyössä','kokoaikaisessa työssä')) #, colors=pal) pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
            ax.legend()
            plt.show()

        empstate_ratio=100*self.empstate/self.alive
        if figname is not None:
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',stack=True,figname=figname+'_stack')
        else:
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',stack=True)

        if self.version>0:
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',ylimit=20,stack=False)
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',parent=True,stack=False)
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',unemp=True,stack=False)

        if figname is not None:
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',start_from=60,stack=True,figname=figname+'_stack60')
        else:
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',start_from=60,stack=True)

    def plot_unemp(self,unempratio=True,figname=None,grayscale=False):
        '''
        Plottaa työttömyysaste (unempratio=True) tai työttömien osuus väestöstö (False)
        '''
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        if unempratio:
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(self.empstate,self.alive,unempratio=True)
            unempratio_stat=100*self.unempratio_stats()
            labeli='keskimääräinen työttömyysaste '+str(ka_tyottomyysaste)      
            ylabeli='Työttömyysaste [%]'
            labeli2='työttömyysaste'
        else:
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(self.empstate,self.alive,unempratio=False)
            unempratio_stat=100*self.unemp_stats()  
            labeli='keskimääräinen työttömien osuus väestöstö '+str(ka_tyottomyysaste)
            ylabeli='Työttömien osuus väestöstö [%]'
            labeli2='työttömien osuus väestöstö'

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel(ylabeli)
        ax.plot(x,unempratio_stat,ls='--',label='havainto')
        ax.plot(x,tyottomyysaste)
        if figname is not None:
            plt.savefig(figname+'tyottomyysaste.eps', format='eps')
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel(ylabeli)
        ax.plot(x,unempratio_stat,label='havainto')
        pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
        ax.stackplot(x,tyottomyysaste,colors=pal)
        #ax.plot(x,tyottomyysaste)
        plt.show()
        
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='Miehet'
                gempstate=np.sum(self.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
                color='darkgray'
            else:
                gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
                leg='Naiset'
                color='black'
        
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive,unempratio=unempratio)
        
            ax.plot(x,tyottomyysaste,color=color,label='{} {}'.format(labeli2,leg))
            
        if grayscale:
            lstyle='--'
        else:
            lstyle='--'            
            
        if unempratio:
            ax.plot(x,100*self.unempratio_stats(g=1),ls=lstyle,label='havainto, naiset')
            ax.plot(x,100*self.unempratio_stats(g=2),ls=lstyle,label='havainto, miehet')
            labeli='keskimääräinen työttömyysaste '+str(ka_tyottomyysaste)      
            ylabeli='Työttömyysaste [%]'
        else:
            ax.plot(x,100*self.unemp_stats(g=1),ls=lstyle,label='havainto, naiset')
            ax.plot(x,100*self.unemp_stats(g=2),ls=lstyle,label='havainto, miehet')
            labeli='keskimääräinen työttömien osuus väestöstö '+str(ka_tyottomyysaste)
            ylabeli='Työttömien osuus väestöstö [%]'
            
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel(ylabeli)
        if False:
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyysaste_spk.eps', format='eps')
        plt.show()        

    def plot_parttime_ratio(self,greyscale=True,figname=None):
        '''
        Plottaa osatyötä tekevien osuus väestöstö
        '''
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        labeli2='Osatyötä tekevien osuus'
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='Miehet'
                g='men'
                pstyle='-'
            else:
                g='women'
                leg='Naiset'
                pstyle=''
        
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_gempratios(gender=g,unempratio=False)
        
            ax.plot(x,osatyoaste,'{}'.format(pstyle),label='{} {}'.format(labeli2,leg))
            
            
        o_x=np.array([20,30,40,50,60,70])
        f_osatyo=np.array([55,21,16,12,18,71])
        m_osatyo=np.array([32,8,5,4,9,65])
        if greyscale:
            ax.plot(o_x,f_osatyo,ls='--',label='havainto, naiset')
            ax.plot(o_x,m_osatyo,ls='--',label='havainto, miehet')
        else:
            ax.plot(o_x,f_osatyo,label='havainto, naiset')
            ax.plot(o_x,m_osatyo,label='havainto, miehet')
        labeli='osatyöaste '#+str(ka_tyottomyysaste)      
        ylabeli='Osatyön osuus työnteosta [%]'
            
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel(ylabeli)
        if False:
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'osatyoaste_spk.eps', format='eps')
        plt.show()

        
    def plot_unemp_shares(self):
        empstate_ratio=100*self.empstate/self.alive
        self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',onlyunemp=True,stack=True)

    def plot_group_emp(self,grayscale=False,figname=None):
        fig,ax=plt.subplots()
        if grayscale:
            lstyle='--'
        else:
            lstyle='--'
        
        for gender in range(2):
            if gender==0:
                leg='Miehet'
                gempstate=np.sum(self.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
                color='darkgray'
            else:
                gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
                leg='Naiset'
                color='black'
        
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive)
        
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,tyollisyysaste,color=color,label='työllisyysaste {}'.format(leg))
            #ax.plot(x,tyottomyysaste,label='työttömyys {}'.format(leg))
            
        emp_statsratio=100*self.emp_stats(g=2)
        ax.plot(x,emp_statsratio,ls=lstyle,color='darkgray',label='havainto, miehet')
        emp_statsratio=100*self.emp_stats(g=1)
        ax.plot(x,emp_statsratio,ls=lstyle,color='black',label='havainto, naiset')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        if False:
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyollisyysaste_spk.eps', format='eps')
                
        plt.show()

    def plot_pensions(self):
        if self.version>0:
            self.plot_ratiostates(self.stat_pension,ylabel='Tuleva eläke [e/v]',stack=False)

    def plot_career(self):    
        if self.version>0:
            self.plot_ratiostates(self.stat_tyoura,ylabel='Työuran pituus [v]',stack=False)

    def plot_ratiostates(self,statistic,ylabel='',ylimit=None, show_legend=True, parent=False,\
                         unemp=False,start_from=None,stack=False,no_ve=False,figname=None):
        self.plot_states(statistic/self.empstate,ylabel=ylabel,ylimit=ylimit,no_ve=no_ve,\
                    show_legend=show_legend,parent=parent,unemp=unemp,start_from=start_from,\
                    stack=stack,figname=figname)

    def count_putki(self,emps=None):
        if emps is None:
            piped=np.reshape(self.empstate[:,4],(self.empstate[:,4].shape[0],1))
            demog,demog2=self.get_demog()
            putkessa=self.timestep*np.nansum(piped[1:]/self.alive[1:]*demog2[1:])
            return putkessa
        else:
            piped=np.reshape(emps[:,4],(emps[:,4].shape[0],1))
            demog,demog2=self.get_demog()
            alive=np.sum(emps,axis=1,keepdims=True)
            putkessa=self.timestep*np.nansum(piped[1:]/alive[1:]*demog2[1:])
            return putkessa

    def plot_states(self,statistic,ylabel='',ylimit=None,show_legend=True,parent=False,unemp=False,no_ve=False,
                    start_from=None,stack=True,figname=None,yminlim=None,ymaxlim=None,
                    onlyunemp=False,reverse=False,grayscale=False):
        if start_from is None:
            x=np.linspace(self.min_age,self.max_age,self.n_time)
        else:
            x_n = self.max_age-60+1
            x_t = int(np.round((x_n-1)*self.inv_timestep))+2
            x=np.linspace(start_from,self.max_age,x_t)
            #x=np.linspace(start_from,self.max_age,self.n_time)
            statistic=statistic[self.map_age(start_from):]
    
        ura_emp=statistic[:,1]
        ura_ret=statistic[:,2]
        ura_unemp=statistic[:,0]
        if self.version>0:
            ura_disab=statistic[:,3]
            ura_pipe=statistic[:,4]
            ura_mother=statistic[:,5]
            ura_dad=statistic[:,6]
            ura_kht=statistic[:,7]
            ura_vetyo=statistic[:,9]
            ura_veosatyo=statistic[:,8]
            ura_osatyo=statistic[:,10]
            ura_outsider=statistic[:,11]
            ura_student=statistic[:,12]
            ura_tyomarkkinatuki=statistic[:,13]
            ura_army=statistic[:,14]

        if no_ve:
            ura_ret[-2:-1]=None

        fig,ax=plt.subplots()
        if stack:
            if grayscale:
                pal=sns.light_palette("black", 8, reverse=True)
            else:
                pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
            reverse=True
            
            if parent:
                if self.version>0:
                    ax.stackplot(x,ura_mother,ura_dad,ura_kht,
                        labels=('äitiysvapaa','isyysvapaa','khtuki'), colors=pal)
            elif unemp:
                if self.version>0:
                    ax.stackplot(x,ura_unemp,ura_pipe,ura_student,ura_outsider,ura_tyomarkkinatuki,
                        labels=('tyött','putki','opiskelija','ulkona','tm-tuki'), colors=pal)
                else:
                    ax.stackplot(x,ura_unemp,labels=('tyött'), colors=pal)
            elif onlyunemp:
                if self.version>0:
                    #urasum=np.nansum(statistic[:,[0,4,11,13]],axis=1)/100
                    urasum=np.nansum(statistic[:,[0,4,13]],axis=1)/100
                    osuus=(1.0-np.array([0.84,0.68,0.62,0.58,0.57,0.55,0.53,0.50,0.29]))*100
                    xx=np.array([22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5])
                    #ax.stackplot(x,ura_unemp/urasum,ura_pipe/urasum,ura_outsider/urasum,ura_tyomarkkinatuki/urasum,
                    #    labels=('ansiosidonnainen','lisäpäivät','työvoiman ulkopuolella','tm-tuki'), colors=pal)
                    ax.stackplot(x,ura_unemp/urasum,ura_pipe/urasum,ura_tyomarkkinatuki/urasum,
                        labels=('ansiosidonnainen','lisäpäivät','tm-tuki'), colors=pal)
                    ax.plot(xx,osuus,color='k')
                else:
                    ax.stackplot(x,ura_unemp,labels=('tyött'), colors=pal)
            else:
                if self.version>0:
                    #ax.stackplot(x,ura_emp,ura_osatyo,ura_vetyo,ura_veosatyo,ura_unemp,ura_tyomarkkinatuki,ura_pipe,ura_disab,ura_mother,ura_dad,ura_kht,ura_ret,ura_student,ura_outsider,ura_army,
                    #    labels=('työssä','osatyö','ve+työ','ve+osatyö','työtön','tm-tuki','työttömyysputki','tk-eläke','äitiysvapaa','isyysvapaa','kh-tuki','vanhuuseläke','opiskelija','työvoiman ulkop.','armeijassa'), 
                    #    colors=pal)
                    ax.stackplot(x,ura_emp,ura_osatyo,ura_vetyo,ura_veosatyo,ura_unemp,ura_tyomarkkinatuki,ura_pipe,ura_ret,ura_disab,ura_mother,ura_dad,ura_kht,ura_student,ura_outsider,ura_army,
                        labels=('työssä','osatyö','ve+työ','ve+osatyö','työtön','tm-tuki','työttömyysputki','vanhuuseläke','tk-eläke','äitiysvapaa','isyysvapaa','kh-tuki','opiskelija','työvoiman ulkop.','armeijassa'), 
                        colors=pal)
                else:
                    ax.stackplot(x,ura_emp,ura_unemp,ura_ret,
                        labels=('työssä','työtön','vanhuuseläke'), colors=pal)
            if start_from is None:
                ax.set_xlim(self.min_age,self.max_age)
            else:
                ax.set_xlim(60,self.max_age)
        
            if ymaxlim is None:
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(yminlim,ymaxlim)
        else:
            if parent:
                if self.version>0:
                    ax.plot(x,ura_mother,label='äitiysvapaa')
                    ax.plot(x,ura_dad,label='isyysvapaa')
                    ax.plot(x,ura_kht,label='khtuki')
            elif unemp:
                ax.plot(x,ura_unemp,label='tyött')
                if self.version>0:
                    ax.plot(x,ura_tyomarkkinatuki,label='tm-tuki')
                    ax.plot(x,ura_student,label='student')
                    ax.plot(x,ura_outsider,label='outsider')
                    ax.plot(x,ura_pipe,label='putki')
            else:
                ax.plot(x,ura_unemp,label='tyött')
                ax.plot(x,ura_ret,label='eläke')
                ax.plot(x,ura_emp,label='työ')
                if self.version>0:
                    ax.plot(x,ura_disab,label='tk')
                    ax.plot(x,ura_pipe,label='putki')
                    ax.plot(x,ura_tyomarkkinatuki,label='tm-tuki')
                    ax.plot(x,ura_mother,label='äitiysvapaa')
                    ax.plot(x,ura_dad,label='isyysvapaa')
                    ax.plot(x,ura_kht,label='khtuki')
                    ax.plot(x,ura_vetyo,label='ve+työ')
                    ax.plot(x,ura_veosatyo,label='ve+osatyö')
                    ax.plot(x,ura_osatyo,label='osatyö')
                    ax.plot(x,ura_student,label='student')
                    ax.plot(x,ura_outsider,label='outsider')
                    ax.plot(x,ura_army,label='armeijassa')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel(ylabel)
        if show_legend:
            if not reverse:
                lgd=ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            else:
                handles, labels = ax.get_legend_handles_labels()
                lgd=ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
        if ylimit is not None:
            ax.set_ylim([0,ylimit])  
        #fig.tight_layout()
        if figname is not None:
            if show_legend:
                plt.savefig(figname,bbox_inches='tight',bbox_extra_artists=(lgd,), format='eps')
            else:
                plt.savefig(figname,bbox_inches='tight', format='eps')
        plt.show()

    def plot_toe(self):    
        if self.version>0:
            self.plot_ratiostates(self.stat_toe,'työssäolo-ehdon pituus 28 kk aikana [v]',stack=False)
            
    def plot_sal(self):
        self.plot_ratiostates(self.salaries_emp,'Keskipalkka [e/v]',stack=False)

    def plot_moved(self):
        siirtyneet_ratio=self.siirtyneet/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        pysyneet_ratio=self.pysyneet/self.alive*100
        self.plot_states(pysyneet_ratio,ylabel='Pysyneet tilassa',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(pysyneet_ratio,1))))

    def plot_army(self):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*self.empstate[:,14]/self.alive[:,0],label='armeijassa ja siviilipalveluksessa olevat')
        emp_statsratio=100*self.army_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend()
        plt.show()

    def plot_group_army(self):
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='Armeija Miehet'
                opiskelijat=np.sum(self.gempstate[:,14,0:3],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
            else:
                leg='Armeija Naiset'
                opiskelijat=np.sum(self.gempstate[:,14,3:6],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
        
            opiskelijat=np.reshape(opiskelijat,(self.galive.shape[0],1))
            osuus=100*opiskelijat/alive
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,osuus,label=leg)
            
        emp_statsratio=100*self.army_stats(g=1)
        ax.plot(x,emp_statsratio,label='havainto, naiset')
        emp_statsratio=100*self.army_stats(g=2)
        ax.plot(x,emp_statsratio,label='havainto, miehet')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        

    def plot_ave_stay(self):
        self.plot_ratiostates(self.time_in_state,ylabel='Ka kesto tilassa',stack=False)
        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        plt.plot(x,self.time_in_state[:,1]/self.empstate[:,1])
        ax.set_xlabel('Aika')
        ax.set_ylabel('Ka kesto työssä')
        plt.show()
        fig,ax=plt.subplots()
        ax.set_xlabel('Aika')
        ax.set_ylabel('ka kesto työttömänä')
        plt.plot(x,self.time_in_state[:,0]/self.empstate[:,0])
        plt.show()

    def plot_reward(self):
        self.plot_ratiostates(self.rewstate,ylabel='Keskireward tilassa',stack=False)
        self.plot_ratiostates(self.rewstate,ylabel='Keskireward tilassa',stack=False,no_ve=True)
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        total_reward=np.sum(self.rewstate,axis=1)
        fig,ax=plt.subplots()
        ax.plot(x,total_reward)
        ax.set_xlabel('Aika')
        ax.set_ylabel('Koko reward tilassa')
        ax.legend()
        plt.show() 
        
    def comp_total_reward(self): 
        total_reward=np.sum(self.rewstate)
        rr=total_reward/self.n_pop
        #print('total rew1 {} rew2 {}'.format(total_reward,np.sum(self.poprewstate)))
        #print('ave rew1 {} rew2 {}'.format(rr,np.mean(np.sum(self.poprewstate,axis=0))))
        #print('shape rew2 {} pop {} alive {}'.format(self.poprewstate.shape,self.n_pop,self.alive[0]))
        print('Ave reward {}'.format(rr))
        
        return rr

    def plot_wage_reduction(self):
        self.plot_ratiostates(self.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False)
        self.plot_ratiostates(self.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False,unemp=True)
        #self.plot_ratiostates(np.log(1.0+self.stat_wage_reduction),ylabel='log 5wage-reduction tilassa',stack=False)

    def plot_distrib(self,label='',plot_emp=False,plot_bu=False,ansiosid=False,tmtuki=False,putki=False,outsider=False,max_age=500,laaja=False,max=4,figname=None):
        unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=ansiosid,tmtuki=tmtuki,putki=putki,outsider=outsider,max_age=max_age,laaja=laaja)
        tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=ansiosid,tmtuki=tmtuki,putki=putki,outsider=outsider,max_age=max_age,laaja=laaja)
        if plot_emp:
            self.plot_empdistribs(emp_distrib)
        if plot_bu:
            self.plot_unempdistribs_bu(unemp_distrib_bu)
        else:
            self.plot_unempdistribs(unemp_distrib,figname=figname)
        #self.plot_tyolldistribs(unemp_distrib,tyoll_distrib,tyollistyneet=False)     
        if plot_bu:
            self.plot_tyolldistribs_both_bu(unemp_distrib_bu,tyoll_distrib_bu,max=max)
        else:
            self.plot_tyolldistribs_both(unemp_distrib,tyoll_distrib,max=max,figname=figname)

    def plot_irr(self,figname=''):
        self.comp_aggirr()
        self.comp_irr()
        self.plot_irrdistrib(self.infostats_irr,figname=figname)

    def plot_irrdistrib(self,irr_distrib,grayscale=True,figname=''):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        print('Nans {} out of {}'.format(np.sum(np.isnan(irr_distrib)),irr_distrib.shape[0]))
        fig,ax=plt.subplots()
        ax.set_xlabel('Sisäinen tuottoaste [%]')
        lbl=ax.set_ylabel('Taajuus')
        
        #ax.set_yscale('log')
        #max_time=50
        #nn_time = int(np.round((max_time)*self.inv_timestep))+1        
        x=np.linspace(-5,5,100)
        scaled,x2=np.histogram(irr_distrib,x)
        #scaled=scaled/np.nansum(np.abs(irr_distrib))
        ax.bar(x2[1:-1],scaled[1:],align='center')
        #print(x2,scaled)
        if figname is not None:
            plt.savefig(figname+'irrdistrib.eps', bbox_inches='tight', format='eps')
        plt.show()
        fig,ax=plt.subplots()
        ax.hist(irr_distrib,bins=40)
        plt.show()
        print('Keskimääräinen irr {:.3f} %'.format(np.nanmean(irr_distrib)))
        print('Mediaani irr {:.3f} %'.format(np.nanmedian(irr_distrib)))
        count = (irr_distrib < 0).sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus irr<0 {} %:lla'.format(100*percent))
        count = (irr_distrib <=-50).sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus irr<-50 {} %:lla'.format(100*percent))
        count = (np.sum(self.infostats_paid_tyel_pension,axis=0)<0.1).sum()
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus eläke ei maksussa {} %:lla'.format(100*percent))
        count1 = np.sum(self.popempstate[0:self.map_age(63),:]==15)
        count = (np.sum(self.infostats_paid_tyel_pension,axis=0)<0.1).sum()-count1
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus eläke ei maksussa, ei kuollut {} %:lla'.format(100*percent))
        count = np.sum(self.popempstate==15)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus kuolleet {} %:lla'.format(100*percent))

    def plot_stats(self,greyscale=False,figname=None):
        
        if greyscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        self.comp_total_reward()
        #self.plot_rewdist()

        self.plot_emp(figname=figname)

        if self.version>0:            
            q=self.comp_budget(scale=True)
            q_stat=self.stat_budget_2018()
            df1 = pd.DataFrame.from_dict(q,orient='index',columns=['e/v'])
            df2 = pd.DataFrame.from_dict(q_stat,orient='index',columns=['toteuma'])
            df=df1.copy()
            df['toteuma']=df2['toteuma']
            df['ero']=df1['e/v']-df2['toteuma']
                           
            print('Rahavirrat skaalattuna väestötasolle')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

            q=self.comp_participants(scale=True)
            q_stat=self.stat_participants_2018()
            q_days=self.stat_days_2018()
            df1 = pd.DataFrame.from_dict(q,orient='index',columns=['arvio (htv)'])
            df2 = pd.DataFrame.from_dict(q_stat,orient='index',columns=['toteuma'])
            df3 = pd.DataFrame.from_dict(q_days,orient='index',columns=['htv_tot'])

            df=df1.copy()
            df['toteuma (kpl)']=df2['toteuma']
            df['toteuma (htv)']=df3['htv_tot']
            df['ero (htv)']=df['arvio (htv)']-df['toteuma (htv)']
                           
            print('Henkilöitä tiloissa skaalattuna väestötasolle')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.0f"))
        else:
            q=self.comp_participants(scale=True)
            q_stat=self.stat_participants_2018()
            q_days=self.stat_days_2018()
            df1 = pd.DataFrame.from_dict(q,orient='index',columns=['arvio (htv)'])
            df2 = pd.DataFrame.from_dict(q_stat,orient='index',columns=['toteuma'])
            df3 = pd.DataFrame.from_dict(q_days,orient='index',columns=['htv_tot'])

            df=df1.copy()
            df['toteuma (kpl)']=df2['toteuma']
            df['toteuma (htv)']=df3['htv_tot']
            df['ero (htv)']=df['arvio (htv)']-df['toteuma (htv)']
                           
            print('Henkilöitä tiloissa skaalattuna väestötasolle')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.0f"))
        
        G=self.comp_gini()
        
        print('Gini coefficient is {}'.format(G))
        
        if self.version>0:
            self.plot_outsider()
        
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        keskikesto=self.comp_unemp_durations()
        df = pd.DataFrame.from_dict(keskikesto,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))
        
        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        keskikesto=self.comp_unemp_durations_v2()
        df = pd.DataFrame.from_dict(keskikesto,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))
        
        self.plot_emp(figname=figname)
        if self.version>0:
            print('Lisäpäivillä on {:.0f} henkilöä'.format(self.count_putki()))
        self.plot_unemp(unempratio=True,figname=figname)
        self.plot_unemp(unempratio=False)
        self.plot_unemp_shares()
        if self.version>0:
            self.plot_group_emp(figname=figname)
            self.plot_parttime_ratio(figname=figname)
            
        self.plot_sal()
        if self.version>0:
            self.plot_pinkslip()
            self.plot_outsider()
            self.plot_student()
            self.plot_army()
            self.plot_group_student()
            self.plot_group_army()
            self.plot_group_disab()
            self.plot_moved()
            self.plot_ave_stay()
            self.plot_reward()
            self.plot_pensions()
            self.plot_career()
            self.plot_toe()
            self.plot_wage_reduction()
        
        self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, no max age',ansiosid=True,tmtuki=True,putki=True,outsider=False)
        self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää',ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=50,figname=figname)
        
        if self.version>0:
            #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu=True,ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=50)
            self.plot_distrib(label='Jakauma ansiosidonnainen+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu=False,ansiosid=True,tmtuki=False,putki=True,outsider=False,max_age=50)
            #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki ilman putkea',ansiosid=True,tmtuki=True,putki=False,outsider=False)
            #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki ilman putkea, max Ikä 50v',ansiosid=True,tmtuki=True,putki=False,outsider=False,max_age=50)
            self.plot_distrib(label='Jakauma tmtuki',ansiosid=False,tmtuki=True,putki=False,outsider=False)
            #self.plot_distrib(label='Jakauma työvoiman ulkopuoliset',ansiosid=False,tmtuki=False,putki=False,outsider=True)
            #self.plot_distrib(label='Jakauma laaja (ansiosidonnainen+tmtuki+putki+ulkopuoliset)',laaja=True)

            
    def plot_final(self):
        
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        keskikesto=self.comp_unemp_durations()
        df = pd.DataFrame.from_dict(keskikesto,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))
        
        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        keskikesto=self.comp_unemp_durations_v2()
        df = pd.DataFrame.from_dict(keskikesto,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))
        
        self.plot_emp()
        if self.version>0:
            print('Lisäpäivillä on {:.0f} henkilöä'.format(self.count_putki()))
        self.plot_unemp(unempratio=True)
        self.plot_unemp(unempratio=False)
        self.plot_unemp_shares()
        if self.version>0:
            self.plot_group_emp()
        self.plot_parttime_ratio()
        self.plot_sal()
        
        self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, no max age',ansiosid=True,tmtuki=True,putki=True,outsider=False)
        self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää',ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=50)
        #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu=True,ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=50)
        self.plot_distrib(label='Jakauma ansiosidonnainen+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu=False,ansiosid=True,tmtuki=False,putki=True,outsider=False,max_age=50)
        #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki ilman putkea',ansiosid=True,tmtuki=True,putki=False,outsider=False)
        #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki ilman putkea, max Ikä 50v',ansiosid=True,tmtuki=True,putki=False,outsider=False,max_age=50)
        self.plot_distrib(label='Jakauma tmtuki',ansiosid=False,tmtuki=True,putki=False,outsider=False)
        #self.plot_distrib(label='Jakauma työvoiman ulkopuoliset',ansiosid=False,tmtuki=False,putki=False,outsider=True)
        #self.plot_distrib(label='Jakauma laaja (ansiosidonnainen+tmtuki+putki+ulkopuoliset)',laaja=True)

        if self.version>0:
            self.plot_outsider()
            self.plot_student()
            self.plot_army()
            self.plot_group_student()
            self.plot_group_army()
            self.plot_group_disab()
            self.plot_moved()
            self.plot_ave_stay()
            self.plot_reward()
            self.plot_pensions()
            self.plot_career()
            self.plot_toe()
            self.plot_wage_reduction()            

    def plot_img(self,img,xlabel="eläke",ylabel="Palkka",title="Employed"):
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        heatmap = plt.pcolor(img) 
        plt.colorbar(heatmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.show()
        
    # _stats functions return actual statistics (From Statistics Finland, 2018)
    def map_ratios(self,ratio):
        emp_ratio=np.zeros(self.n_time)
        for a in range(self.min_age,self.max_age):
            emp_ratio[self.map_age(a):self.map_age(a+1)]=ratio[a-self.min_age]
            
        return emp_ratio
        
    def emp_stats(self,g=0):
        '''
        työssä olevien osuus väestöstö
        Lähde: Tilastokeskus
        '''
        if self.year==2018:
            if g==0: # kaikki
                emp=np.array([0.461,0.545,0.567,0.599,0.645,0.678,0.706,0.728,0.740,0.752,0.758,0.769,0.776,0.781,0.787,0.795,0.801,0.807,0.809,0.820,0.820,0.829,0.831,0.833,0.832,0.828,0.827,0.824,0.822,0.817,0.815,0.813,0.807,0.802,0.796,0.788,0.772,0.763,0.752,0.728,0.686,0.630,0.568,0.382,0.217,0.142,0.106,0.086,0.011,0.003,0.002])
            elif g==1: # naiset
                emp=np.array([ 0.554,0.567,0.584,0.621,0.666,0.685,0.702,0.723,0.727,0.734,0.741,0.749,0.753,0.762,0.768,0.777,0.788,0.793,0.798,0.813,0.816,0.827,0.832,0.834,0.835,0.835,0.833,0.836,0.833,0.831,0.828,0.827,0.824,0.821,0.815,0.812,0.798,0.791,0.779,0.758,0.715,0.664,0.596,0.400,0.220,0.136,0.098,0.079,0.011,0.004,0.002 ])
            else: # miehet
                emp=np.array([ 0.374,0.524,0.550,0.579,0.626,0.671,0.710,0.733,0.752,0.769,0.774,0.788,0.798,0.800,0.805,0.812,0.814,0.820,0.819,0.826,0.824,0.831,0.831,0.831,0.828,0.821,0.822,0.813,0.811,0.803,0.803,0.798,0.790,0.783,0.776,0.764,0.746,0.735,0.723,0.696,0.656,0.596,0.539,0.362,0.214,0.148,0.115,0.094,0.012,0.002,0.002 ])
        else: # 2019, päivitä
            if g==1:
                ratio=np.array([0.014412417,0.017866162,0.019956194,0.019219484,0.022074491,0.022873602,0.024247334,0.025981477,0.025087389,0.023162522,0.025013013,0.023496079,0.025713399,0.025633996,0.028251301,0.028930719,0.028930188,0.030955287,0.031211716,0.030980726,0.035395247,0.03522291,0.035834422,0.036878386,0.040316277,0.044732619,0.046460599,0.050652725,0.054797849,0.057018324,0.0627497,0.067904263,0.072840649,0.079978222,0.083953327,0.092811744,0.106671337,0.119490669,0.129239815,0.149503982,0.179130081,0.20749958,0.22768029,0.142296259,0.135142865,0.010457403,0,0,0,0,0])
            else:
                ratio=np.array([0.0121151,0.0152247,0.0170189,0.0200570,0.0196213,0.0208018,0.0215082,0.0223155,0.0220908,0.0213913,0.0214263,0.0242843,0.0240043,0.0240721,0.0259648,0.0263371,0.0284309,0.0270143,0.0286249,0.0305952,0.0318945,0.0331264,0.0350743,0.0368707,0.0401613,0.0431067,0.0463718,0.0487914,0.0523801,0.0569297,0.0596571,0.0669273,0.0713361,0.0758116,0.0825295,0.0892805,0.1047429,0.1155854,0.1336167,0.1551418,0.1782882,0.2106220,0.2291799,0.1434176,0.1301574,0.0110726,0,0,0,0,0])

        return self.map_ratios(emp)
        
        
    def disab_stat(self,g):
        '''
        Työkyvyttömyyselökkeellö olevien osuus väestöstö
        Lähde: ETK
        '''
        if self.year==2018:
            if g==1:
                ratio=np.array([0.014412417,0.017866162,0.019956194,0.019219484,0.022074491,0.022873602,0.024247334,0.025981477,0.025087389,0.023162522,0.025013013,0.023496079,0.025713399,0.025633996,0.028251301,0.028930719,0.028930188,0.030955287,0.031211716,0.030980726,0.035395247,0.03522291,0.035834422,0.036878386,0.040316277,0.044732619,0.046460599,0.050652725,0.054797849,0.057018324,0.0627497,0.067904263,0.072840649,0.079978222,0.083953327,0.092811744,0.106671337,0.119490669,0.129239815,0.149503982,0.179130081,0.20749958,0.22768029,0.142296259,0.135142865,0.010457403,0,0,0,0,0])
            else:
                ratio=np.array([0.0121151,0.0152247,0.0170189,0.0200570,0.0196213,0.0208018,0.0215082,0.0223155,0.0220908,0.0213913,0.0214263,0.0242843,0.0240043,0.0240721,0.0259648,0.0263371,0.0284309,0.0270143,0.0286249,0.0305952,0.0318945,0.0331264,0.0350743,0.0368707,0.0401613,0.0431067,0.0463718,0.0487914,0.0523801,0.0569297,0.0596571,0.0669273,0.0713361,0.0758116,0.0825295,0.0892805,0.1047429,0.1155854,0.1336167,0.1551418,0.1782882,0.2106220,0.2291799,0.1434176,0.1301574,0.0110726,0,0,0,0,0])
        else: # 2019, päivitä
            if g==1:
                ratio=np.array([0.014412417,0.017866162,0.019956194,0.019219484,0.022074491,0.022873602,0.024247334,0.025981477,0.025087389,0.023162522,0.025013013,0.023496079,0.025713399,0.025633996,0.028251301,0.028930719,0.028930188,0.030955287,0.031211716,0.030980726,0.035395247,0.03522291,0.035834422,0.036878386,0.040316277,0.044732619,0.046460599,0.050652725,0.054797849,0.057018324,0.0627497,0.067904263,0.072840649,0.079978222,0.083953327,0.092811744,0.106671337,0.119490669,0.129239815,0.149503982,0.179130081,0.20749958,0.22768029,0.142296259,0.135142865,0.010457403,0,0,0,0,0])
            else:
                ratio=np.array([0.0121151,0.0152247,0.0170189,0.0200570,0.0196213,0.0208018,0.0215082,0.0223155,0.0220908,0.0213913,0.0214263,0.0242843,0.0240043,0.0240721,0.0259648,0.0263371,0.0284309,0.0270143,0.0286249,0.0305952,0.0318945,0.0331264,0.0350743,0.0368707,0.0401613,0.0431067,0.0463718,0.0487914,0.0523801,0.0569297,0.0596571,0.0669273,0.0713361,0.0758116,0.0825295,0.0892805,0.1047429,0.1155854,0.1336167,0.1551418,0.1782882,0.2106220,0.2291799,0.1434176,0.1301574,0.0110726,0,0,0,0,0])
            
        return self.map_ratios(ratio)
        
    def student_stats(self,g=0):
        '''
        Opiskelijoiden osuus väestöstö
        Lähde: Tilastokeskus
        '''
        if self.year==2018:
            if g==0: # kaikki
                emp_ratio=np.array([0.261,0.279,0.272,0.242,0.195,0.155,0.123,0.098,0.082,0.070,0.062,0.054,0.048,0.045,0.041,0.039,0.035,0.033,0.031,0.027,0.025,0.024,0.022,0.019,0.018,0.017,0.017,0.016,0.015,0.014,0.013,0.011,0.010,0.009,0.009,0.008,0.008,0.006,0.005,0.004,0.003,0.003,0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001])
            elif g==1: # naiset
                emp_ratio=np.array([0.283,0.290,0.271,0.231,0.184,0.151,0.124,0.100,0.089,0.079,0.069,0.062,0.058,0.055,0.052,0.050,0.044,0.040,0.038,0.034,0.031,0.029,0.027,0.024,0.023,0.021,0.021,0.019,0.017,0.016,0.015,0.012,0.011,0.011,0.011,0.009,0.008,0.007,0.005,0.005,0.003,0.002,0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001 ])
            else: # miehet
                emp_ratio=np.array([0.240,0.269,0.273,0.252,0.207,0.159,0.122,0.096,0.076,0.062,0.056,0.047,0.037,0.035,0.031,0.029,0.027,0.026,0.023,0.021,0.019,0.019,0.017,0.015,0.015,0.014,0.013,0.013,0.013,0.011,0.010,0.010,0.009,0.007,0.008,0.007,0.007,0.006,0.005,0.004,0.003,0.003,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.002,0.001 ])
        else: # 2019, päivitä
            if g==0: # kaikki
                emp_ratio=np.array([0.261,0.279,0.272,0.242,0.195,0.155,0.123,0.098,0.082,0.070,0.062,0.054,0.048,0.045,0.041,0.039,0.035,0.033,0.031,0.027,0.025,0.024,0.022,0.019,0.018,0.017,0.017,0.016,0.015,0.014,0.013,0.011,0.010,0.009,0.009,0.008,0.008,0.006,0.005,0.004,0.003,0.003,0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001])
            elif g==1: # naiset
                emp_ratio=np.array([0.283,0.290,0.271,0.231,0.184,0.151,0.124,0.100,0.089,0.079,0.069,0.062,0.058,0.055,0.052,0.050,0.044,0.040,0.038,0.034,0.031,0.029,0.027,0.024,0.023,0.021,0.021,0.019,0.017,0.016,0.015,0.012,0.011,0.011,0.011,0.009,0.008,0.007,0.005,0.005,0.003,0.002,0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001 ])
            else: # miehet
                emp_ratio=np.array([0.240,0.269,0.273,0.252,0.207,0.159,0.122,0.096,0.076,0.062,0.056,0.047,0.037,0.035,0.031,0.029,0.027,0.026,0.023,0.021,0.019,0.019,0.017,0.015,0.015,0.014,0.013,0.013,0.013,0.011,0.010,0.010,0.009,0.007,0.008,0.007,0.007,0.006,0.005,0.004,0.003,0.003,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.002,0.001 ])

        return self.map_ratios(emp_ratio)
        
    def army_stats(self,g=0):
        '''
        Armeijassa olevien osuus väestöstö
        Lähde: Tilastokeskus
        '''
        if self.year==2018:
            if g==0: # kaikki
                emp_ratio=np.array([0.048,0.009,0.004,0.002,0.001,0.001,0.001,0.001,0.000,0.000,0.000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   ])
            elif g==1: # naiset
                emp_ratio=np.array([0.004,0.002,0.001,0.001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   ])
            else: # miehet
                emp_ratio=np.array([0.089,0.015,0.006,0.004,0.002,0.002,0.001,0.001,0.001,0.001,0.000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   ])
        else: # 2019, päivitä
            if g==0: # kaikki
                emp_ratio=np.array([0.048,0.009,0.004,0.002,0.001,0.001,0.001,0.001,0.000,0.000,0.000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   ])
            elif g==1: # naiset
                emp_ratio=np.array([0.004,0.002,0.001,0.001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   ])
            else: # miehet
                emp_ratio=np.array([0.089,0.015,0.006,0.004,0.002,0.002,0.001,0.001,0.001,0.001,0.000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   ])

        return self.map_ratios(emp_ratio)
        
    def outsider_stats(self,g=0):
        '''
        Työelömön ulkopuolella olevien osuus väestöstö
        Lähde: Tilastokeskus
        '''
        if self.year==2018:
            if g==0: # kaikki
                emp_ratio=np.array([ 0.115,0.070,0.065,0.066,0.066,0.069,0.073,0.075,0.077,0.079,0.079,0.079,0.076,0.075,0.072,0.067,0.065,0.063,0.062,0.057,0.055,0.050,0.048,0.048,0.047,0.046,0.045,0.044,0.042,0.044,0.042,0.043,0.042,0.043,0.043,0.044,0.045,0.045,0.045,0.044,0.044,0.040,0.038,0.022,0.010,0.007,0.004,0.004,0.004,0.004,0.004 ])
            elif g==1: # naiset
                emp_ratio=np.array([ 0.077,0.066,0.068,0.072,0.074,0.082,0.089,0.092,0.098,0.099,0.101,0.098,0.098,0.093,0.088,0.082,0.076,0.074,0.071,0.065,0.061,0.053,0.049,0.049,0.046,0.045,0.044,0.041,0.042,0.041,0.040,0.039,0.040,0.040,0.041,0.041,0.042,0.042,0.042,0.043,0.044,0.041,0.039,0.023,0.012,0.007,0.005,0.004,0.004,0.004,0.004 ])
            else: # miehet
                emp_ratio=np.array([ 0.151,0.074,0.063,0.060,0.057,0.057,0.057,0.059,0.058,0.059,0.059,0.061,0.056,0.057,0.058,0.053,0.054,0.052,0.053,0.051,0.050,0.046,0.047,0.046,0.047,0.046,0.047,0.046,0.042,0.046,0.044,0.046,0.045,0.047,0.044,0.046,0.047,0.048,0.048,0.045,0.044,0.039,0.037,0.021,0.009,0.007,0.004,0.004,0.005,0.003,0.004 ])
        else: # 2019, päivitä
            if g==0: # kaikki
                emp_ratio=np.array([ 0.115,0.070,0.065,0.066,0.066,0.069,0.073,0.075,0.077,0.079,0.079,0.079,0.076,0.075,0.072,0.067,0.065,0.063,0.062,0.057,0.055,0.050,0.048,0.048,0.047,0.046,0.045,0.044,0.042,0.044,0.042,0.043,0.042,0.043,0.043,0.044,0.045,0.045,0.045,0.044,0.044,0.040,0.038,0.022,0.010,0.007,0.004,0.004,0.004,0.004,0.004 ])
            elif g==1: # naiset
                emp_ratio=np.array([ 0.077,0.066,0.068,0.072,0.074,0.082,0.089,0.092,0.098,0.099,0.101,0.098,0.098,0.093,0.088,0.082,0.076,0.074,0.071,0.065,0.061,0.053,0.049,0.049,0.046,0.045,0.044,0.041,0.042,0.041,0.040,0.039,0.040,0.040,0.041,0.041,0.042,0.042,0.042,0.043,0.044,0.041,0.039,0.023,0.012,0.007,0.005,0.004,0.004,0.004,0.004 ])
            else: # miehet
                emp_ratio=np.array([ 0.151,0.074,0.063,0.060,0.057,0.057,0.057,0.059,0.058,0.059,0.059,0.061,0.056,0.057,0.058,0.053,0.054,0.052,0.053,0.051,0.050,0.046,0.047,0.046,0.047,0.046,0.047,0.046,0.042,0.046,0.044,0.046,0.045,0.047,0.044,0.046,0.047,0.048,0.048,0.045,0.044,0.039,0.037,0.021,0.009,0.007,0.004,0.004,0.005,0.003,0.004 ])
        
        return self.map_ratios(emp_ratio)
        
    def tyotjakso_stats(self):
        '''
        Päättyneiden työttömyysjaksojen jakauma
        '''
        d=[]
        
    def pensioner_stats(self,g=0):
        '''
        Eläkkeellä olevien osuus väestöstö
        Lähde: Tilastokeskus
        '''
        if self.year==2018:
            if g==0: # kaikki
                emp_ratio=np.array([ 0.011,0.014,0.016,0.016,0.018,0.019,0.019,0.021,0.020,0.019,0.020,0.021,0.022,0.022,0.024,0.024,0.025,0.025,0.026,0.026,0.029,0.029,0.030,0.031,0.034,0.037,0.039,0.042,0.045,0.047,0.052,0.056,0.060,0.065,0.070,0.076,0.088,0.098,0.110,0.128,0.159,0.196,0.277,0.533,0.741,0.849,0.888,0.908,0.983,0.992,0.993 ])
            elif g==1: # naiset
                emp_ratio=np.array([ 0.010,0.013,0.014,0.016,0.016,0.017,0.018,0.018,0.018,0.018,0.017,0.020,0.020,0.020,0.022,0.021,0.023,0.022,0.023,0.024,0.025,0.026,0.027,0.029,0.031,0.034,0.036,0.038,0.041,0.044,0.047,0.052,0.055,0.058,0.062,0.066,0.077,0.085,0.097,0.111,0.142,0.177,0.258,0.518,0.735,0.855,0.896,0.916,0.983,0.991,0.992 ])
            else: # miehet
                emp_ratio=np.array([ 0.012,0.016,0.018,0.017,0.019,0.020,0.021,0.023,0.022,0.021,0.022,0.021,0.024,0.024,0.026,0.026,0.026,0.028,0.028,0.027,0.032,0.032,0.033,0.033,0.036,0.041,0.042,0.045,0.049,0.051,0.056,0.060,0.065,0.072,0.078,0.086,0.099,0.112,0.123,0.144,0.178,0.216,0.297,0.549,0.748,0.843,0.880,0.901,0.982,0.993,0.993 ])
        else: # 2019, päivitä
            if g==0: # kaikki
                emp_ratio=np.array([ 0.011,0.014,0.016,0.016,0.018,0.019,0.019,0.021,0.020,0.019,0.020,0.021,0.022,0.022,0.024,0.024,0.025,0.025,0.026,0.026,0.029,0.029,0.030,0.031,0.034,0.037,0.039,0.042,0.045,0.047,0.052,0.056,0.060,0.065,0.070,0.076,0.088,0.098,0.110,0.128,0.159,0.196,0.277,0.533,0.741,0.849,0.888,0.908,0.983,0.992,0.993 ])
            elif g==1: # naiset
                emp_ratio=np.array([ 0.010,0.013,0.014,0.016,0.016,0.017,0.018,0.018,0.018,0.018,0.017,0.020,0.020,0.020,0.022,0.021,0.023,0.022,0.023,0.024,0.025,0.026,0.027,0.029,0.031,0.034,0.036,0.038,0.041,0.044,0.047,0.052,0.055,0.058,0.062,0.066,0.077,0.085,0.097,0.111,0.142,0.177,0.258,0.518,0.735,0.855,0.896,0.916,0.983,0.991,0.992 ])
            else: # miehet
                emp_ratio=np.array([ 0.012,0.016,0.018,0.017,0.019,0.020,0.021,0.023,0.022,0.021,0.022,0.021,0.024,0.024,0.026,0.026,0.026,0.028,0.028,0.027,0.032,0.032,0.033,0.033,0.036,0.041,0.042,0.045,0.049,0.051,0.056,0.060,0.065,0.072,0.078,0.086,0.099,0.112,0.123,0.144,0.178,0.216,0.297,0.549,0.748,0.843,0.880,0.901,0.982,0.993,0.993 ])
        

        return self.map_ratios(emp_ratio)
        
    def unemp_stats(self,g=0):
        '''
        Työttömien osuus väestöstö
        Lähde: Tilastokeskus
        '''
        emp_ratio=np.zeros(self.n_time)
        if self.year==2018:
            if g==0:
                emp_ratio=np.array([0.104,0.083,0.077,0.075,0.075,0.078,0.078,0.078,0.080,0.080,0.081,0.077,0.079,0.078,0.076,0.075,0.074,0.072,0.074,0.070,0.071,0.068,0.069,0.069,0.069,0.072,0.072,0.074,0.076,0.078,0.078,0.078,0.081,0.081,0.082,0.084,0.087,0.087,0.089,0.097,0.108,0.131,0.115,0.062,0.030,0,0,0,0,0,0])
            elif g==1: # naiset
                emp_ratio=np.array([ 0.073,0.063,0.062,0.060,0.060,0.065,0.066,0.066,0.069,0.070,0.072,0.071,0.071,0.070,0.071,0.070,0.069,0.070,0.070,0.065,0.066,0.064,0.065,0.064,0.065,0.065,0.066,0.066,0.068,0.067,0.070,0.070,0.069,0.071,0.071,0.072,0.074,0.076,0.076,0.083,0.097,0.116,0.105,0.057,0.031,0  ,0  ,0  ,0  ,0  ,0   ])
            else: # miehet
                emp_ratio=np.array([ 0.133,0.102,0.091,0.089,0.089,0.091,0.089,0.089,0.091,0.089,0.089,0.083,0.085,0.085,0.080,0.081,0.080,0.075,0.077,0.075,0.074,0.072,0.073,0.074,0.074,0.078,0.076,0.083,0.085,0.089,0.087,0.086,0.092,0.091,0.094,0.096,0.101,0.099,0.102,0.110,0.120,0.146,0.125,0.066,0.028,0  ,0  ,0  ,0  ,0  ,0   ])
        else: # 2019, päivitä
            if g==0:
                emp_ratio=np.array([0.104,0.083,0.077,0.075,0.075,0.078,0.078,0.078,0.080,0.080,0.081,0.077,0.079,0.078,0.076,0.075,0.074,0.072,0.074,0.070,0.071,0.068,0.069,0.069,0.069,0.072,0.072,0.074,0.076,0.078,0.078,0.078,0.081,0.081,0.082,0.084,0.087,0.087,0.089,0.097,0.108,0.131,0.115,0.062,0.030,0,0,0,0,0,0])
            elif g==1: # naiset
                emp_ratio=np.array([ 0.073,0.063,0.062,0.060,0.060,0.065,0.066,0.066,0.069,0.070,0.072,0.071,0.071,0.070,0.071,0.070,0.069,0.070,0.070,0.065,0.066,0.064,0.065,0.064,0.065,0.065,0.066,0.066,0.068,0.067,0.070,0.070,0.069,0.071,0.071,0.072,0.074,0.076,0.076,0.083,0.097,0.116,0.105,0.057,0.031,0  ,0  ,0  ,0  ,0  ,0   ])
            else: # miehet
                emp_ratio=np.array([ 0.133,0.102,0.091,0.089,0.089,0.091,0.089,0.089,0.091,0.089,0.089,0.083,0.085,0.085,0.080,0.081,0.080,0.075,0.077,0.075,0.074,0.072,0.073,0.074,0.074,0.078,0.076,0.083,0.085,0.089,0.087,0.086,0.092,0.091,0.094,0.096,0.101,0.099,0.102,0.110,0.120,0.146,0.125,0.066,0.028,0  ,0  ,0  ,0  ,0  ,0   ])
        
        return self.map_ratios(emp_ratio)

    def unempratio_stats(self,g=0):
        '''
        Työttömien osuus väestöstö
        Lähde: Tilastokeskus
        '''
        emp_ratio=self.emp_stats(g=g)
        unemp_ratio=self.unemp_stats(g=g)
        ratio=unemp_ratio/(emp_ratio+unemp_ratio)
        return ratio # ei mapata, on jo tehty!

    def save_sim(self,filename):
        f = h5py.File(filename, 'w')
        ftype='float64'
        _ = f.create_dataset('version', data=self.version, dtype=ftype)
        _ = f.create_dataset('n_pop', data=self.n_pop, dtype=ftype)
        _ = f.create_dataset('empstate', data=self.empstate, dtype=ftype)
        _ = f.create_dataset('gempstate', data=self.gempstate, dtype=ftype)
        _ = f.create_dataset('deceiced', data=self.deceiced, dtype=ftype)
        _ = f.create_dataset('rewstate', data=self.rewstate, dtype=ftype)
        _ = f.create_dataset('salaries_emp', data=self.salaries_emp, dtype=ftype)
        _ = f.create_dataset('actions', data=self.actions, dtype=ftype)
        _ = f.create_dataset('alive', data=self.alive, dtype=ftype)
        _ = f.create_dataset('galive', data=self.galive, dtype=ftype)
        _ = f.create_dataset('siirtyneet', data=self.siirtyneet, dtype=ftype)
        _ = f.create_dataset('pysyneet', data=self.pysyneet, dtype=ftype)
        _ = f.create_dataset('salaries', data=self.salaries, dtype=ftype)
        _ = f.create_dataset('aveV', data=self.aveV, dtype=ftype)
        _ = f.create_dataset('time_in_state', data=self.time_in_state, dtype=ftype)
        _ = f.create_dataset('stat_tyoura', data=self.stat_tyoura, dtype=ftype)
        _ = f.create_dataset('stat_toe', data=self.stat_toe, dtype=ftype)
        _ = f.create_dataset('stat_pension', data=self.stat_pension, dtype=ftype)
        _ = f.create_dataset('stat_paidpension', data=self.stat_paidpension, dtype=ftype)
        _ = f.create_dataset('stat_unemp_len', data=self.stat_unemp_len, dtype=ftype)
        _ = f.create_dataset('popempstate', data=self.popempstate, dtype=ftype)
        _ = f.create_dataset('stat_wage_reduction', data=self.stat_wage_reduction, dtype=ftype)
        _ = f.create_dataset('popunemprightleft', data=self.popunemprightleft, dtype=ftype)
        _ = f.create_dataset('popunemprightused', data=self.popunemprightused, dtype=ftype)
        _ = f.create_dataset('infostats_taxes', data=self.infostats_taxes, dtype=ftype)
        _ = f.create_dataset('infostats_etuustulo', data=self.infostats_etuustulo, dtype=ftype)
        _ = f.create_dataset('infostats_perustulo', data=self.infostats_perustulo, dtype=ftype)
        _ = f.create_dataset('infostats_palkkatulo', data=self.infostats_palkkatulo, dtype=ftype)
        _ = f.create_dataset('infostats_ansiopvraha', data=self.infostats_ansiopvraha, dtype=ftype)
        _ = f.create_dataset('infostats_asumistuki', data=self.infostats_asumistuki, dtype=ftype)
        _ = f.create_dataset('infostats_valtionvero', data=self.infostats_valtionvero, dtype=ftype)
        _ = f.create_dataset('infostats_kunnallisvero', data=self.infostats_kunnallisvero, dtype=ftype)
        _ = f.create_dataset('infostats_ptel', data=self.infostats_ptel, dtype=ftype)
        _ = f.create_dataset('infostats_tyotvakmaksu', data=self.infostats_tyotvakmaksu, dtype=ftype)
        _ = f.create_dataset('infostats_tyoelake', data=self.infostats_tyoelake, dtype=ftype)
        _ = f.create_dataset('infostats_kokoelake', data=self.infostats_kokoelake, dtype=ftype)
        _ = f.create_dataset('infostats_opintotuki', data=self.infostats_opintotuki, dtype=ftype)
        _ = f.create_dataset('infostats_isyyspaivaraha', data=self.infostats_isyyspaivaraha, dtype=ftype)
        _ = f.create_dataset('infostats_aitiyspaivaraha', data=self.infostats_aitiyspaivaraha, dtype=ftype)
        _ = f.create_dataset('infostats_kotihoidontuki', data=self.infostats_kotihoidontuki, dtype=ftype)
        _ = f.create_dataset('infostats_sairauspaivaraha', data=self.infostats_sairauspaivaraha, dtype=ftype)
        _ = f.create_dataset('infostats_toimeentulotuki', data=self.infostats_toimeentulotuki, dtype=ftype)
        _ = f.create_dataset('infostats_tulot_netto', data=self.infostats_tulot_netto, dtype=ftype)
        _ = f.create_dataset('poprewstate', data=self.poprewstate, dtype=ftype)
        _ = f.create_dataset('infostats_pinkslip', data=self.infostats_pinkslip, dtype=ftype)
        _ = f.create_dataset('infostats_paid_tyel_pension', data=self.infostats_paid_tyel_pension, dtype=ftype)
        _ = f.create_dataset('infostats_tyelpremium', data=self.infostats_tyelpremium, dtype=ftype)
        _ = f.create_dataset('infostats_npv0', data=self.infostats_npv0, dtype=ftype)
        _ = f.create_dataset('infostats_irr', data=self.infostats_irr, dtype=ftype)
        
        f.close()

    def save_to_hdf(self,filename,nimi,arr,dtype):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset(nimi, data=arr, dtype=dtype)
        f.close()

    def load_hdf(self,filename,nimi):
        f = h5py.File(filename, 'r')
        val=f.get(nimi).value
        f.close()
        return val
        
    def load_sim(self,filename,n_pop=None):
        f = h5py.File(filename, 'r')
        
        if 'version' in f.keys():
            self.version=f.get('version').value
        else:
            self.version=1

        self.empstate=f.get('empstate').value
        self.gempstate=f.get('gempstate').value
        self.deceiced=f.get('deceiced').value
        self.rewstate=f.get('rewstate').value
        if 'poprewstate' in f.keys():
            self.poprewstate=f.get('poprewstate').value
        
        self.salaries_emp=f.get('salaries_emp').value
        self.actions=f.get('actions').value
        self.alive=f.get('alive').value
        self.galive=f.get('galive').value
        self.siirtyneet=f.get('siirtyneet').value
        self.pysyneet=f.get('pysyneet').value
        self.salaries=f.get('salaries').value
        self.aveV=f.get('aveV').value
        self.time_in_state=f.get('time_in_state').value
        self.stat_tyoura=f.get('stat_tyoura').value
        self.stat_toe=f.get('stat_toe').value
        self.stat_pension=f.get('stat_pension').value
        self.stat_paidpension=f.get('stat_paidpension').value
        self.stat_unemp_len=f.get('stat_unemp_len').value
        self.popempstate=f.get('popempstate').value
        self.stat_wage_reduction=f.get('stat_wage_reduction').value
        self.popunemprightleft=f.get('popunemprightleft').value
        self.popunemprightused=f.get('popunemprightused').value
        
        if 'infostats_taxes' in f.keys(): # older runs do not have these
            self.infostats_taxes=f.get('infostats_taxes').value
            self.infostats_etuustulo=f.get('infostats_etuustulo').value
            self.infostats_perustulo=f.get('infostats_perustulo').value
            self.infostats_palkkatulo=f.get('infostats_palkkatulo').value
            self.infostats_ansiopvraha=f.get('infostats_ansiopvraha').value
            self.infostats_asumistuki=f.get('infostats_asumistuki').value
            self.infostats_valtionvero=f.get('infostats_valtionvero').value
            self.infostats_kunnallisvero=f.get('infostats_kunnallisvero').value
            self.infostats_ptel=f.get('infostats_ptel').value
            self.infostats_tyotvakmaksu=f.get('infostats_tyotvakmaksu').value
            self.infostats_tyoelake=f.get('infostats_tyoelake').value
            self.infostats_kokoelake=f.get('infostats_kokoelake').value
            self.infostats_opintotuki=f.get('infostats_opintotuki').value
            self.infostats_isyyspaivaraha=f.get('infostats_isyyspaivaraha').value
            self.infostats_aitiyspaivaraha=f.get('infostats_aitiyspaivaraha').value
            self.infostats_kotihoidontuki=f.get('infostats_kotihoidontuki').value
            self.infostats_sairauspaivaraha=f.get('infostats_sairauspaivaraha').value
            self.infostats_toimeentulotuki=f.get('infostats_toimeentulotuki').value
            self.infostats_tulot_netto=f.get('infostats_tulot_netto').value

        if 'infostats_pinkslip' in f.keys(): # older runs do not have these
            self.infostats_pinkslip=f.get('infostats_pinkslip').value      
            
        if 'infostats_paid_tyel_pension' in f.keys(): # older runs do not have these
            self.infostats_paid_tyel_pension=f.get('infostats_paid_tyel_pension').value      
            self.infostats_tyelpremium=f.get('infostats_tyelpremium').value                  

        if 'infostats_npv0' in f.keys(): # older runs do not have these
            self.infostats_npv0=f.get('infostats_npv0').value      
            self.infostats_irr=f.get('infostats_irr').value                  
            
        if 'infostats_chilren7' in f.keys(): # older runs do not have these
            self.infostats_chilren7=f.get('infostats_chilren7').value      
        if 'infostats_chilren18' in f.keys(): # older runs do not have these
            self.infostats_chilren18=f.get('infostats_chilren18').value      
        
        if 'n_pop' in f:
            self.n_pop=int(f.get('n_pop').value)
        else:
            self.n_pop=np.sum(self.empstate[0,:])
        if n_pop is not None:
            self.n_pop=n_pop

        print('n_pop {}'.format(self.n_pop))

            
        f.close()
        
    def render(self,load=None,figname=None):
        if load is not None:
            self.load_sim(load)

        #self.plot_stats(5)
        self.plot_stats(figname=figname)
        self.plot_reward()   

    def stat_budget_2018(self,scale=False):
        q={}
        q['palkkatulo']=89_134_200_000 # lähde: ETK
        q['etuusmeno']=0
        q['verot+maksut']=30_763_000_000 
        q['valtionvero']=5_542_000_000
        q['kunnallisvero']=18_991_000_000
        q['ptel']=5_560_000_000
        #q['elakemaksut']=22_085_700_000 # Lähde: ETK
        q['tyotvakmaksu']=0.019*q['palkkatulo']
        q['ansiopvraha']=3_895_333_045 # 1_930_846_464+1_964_486_581 # ansioturva + perusturva = 3 895 333 045
        q['asumistuki']=1_488_900_000 + 600_100_000 # yleinen plus eläkkeensaajan
        q['tyoelake']=27_865_000_000
        q['kokoelake']=q['tyoelake']+2_357_000_000
        q['opintotuki']=417_404_073+54_057
        q['isyyspaivaraha']=104_212_164
        q['aitiyspaivaraha']=341_304_991+462_228_789
        q['kotihoidontuki']=245_768_701
        q['sairauspaivaraha']=786_659_783
        q['toimeentulotuki']=715_950_847
        q['perustulo']=0
        q['etuusmeno']=q['ansiopvraha']+q['kokoelake']+q['opintotuki']+q['isyyspaivaraha']+\
            q['aitiyspaivaraha']+q['sairauspaivaraha']+q['toimeentulotuki']+q['perustulo']
        q['tulot_netto']=q['palkkatulo']+q['etuusmeno']-q['verot+maksut']
        q['muut tulot']=q['etuusmeno']-q['verot+maksut']

        return q
        
    def comp_budget(self,scale=False):
        demog,demog2=self.get_demog()

        scalex=demog2/self.n_pop
        
        q={}
        q['palkkatulo']=np.sum(self.infostats_palkkatulo*scalex)
        q['etuusmeno']=np.sum(self.infostats_etuustulo*scalex)
        q['verot+maksut']=np.sum(self.infostats_taxes*scalex)
        q['muut tulot']=q['etuusmeno']-q['verot+maksut']
        q['valtionvero']=np.sum(self.infostats_valtionvero*scalex)
        q['kunnallisvero']=np.sum(self.infostats_kunnallisvero*scalex)
        q['ptel']=np.sum(self.infostats_ptel*scalex)
        q['tyotvakmaksu']=np.sum(self.infostats_tyotvakmaksu*scalex)
        q['ansiopvraha']=np.sum(self.infostats_ansiopvraha*scalex)
        q['asumistuki']=np.sum(self.infostats_asumistuki*scalex)
        q['tyoelake']=np.sum(self.infostats_tyoelake*scalex)
        q['kokoelake']=np.sum(self.infostats_kokoelake*scalex)
        q['opintotuki']=np.sum(self.infostats_opintotuki*scalex)
        q['isyyspaivaraha']=np.sum(self.infostats_isyyspaivaraha*scalex)
        q['aitiyspaivaraha']=np.sum(self.infostats_aitiyspaivaraha*scalex)
        q['kotihoidontuki']=np.sum(self.infostats_kotihoidontuki*scalex)
        q['sairauspaivaraha']=np.sum(self.infostats_sairauspaivaraha*scalex)
        q['toimeentulotuki']=np.sum(self.infostats_toimeentulotuki*scalex)
        q['perustulo']=np.sum(self.infostats_perustulo*scalex)
        q['tulot_netto']=np.sum(self.infostats_tulot_netto*scalex)

        return q

    def comp_participants(self,scale=False):
        demog,demog2=self.get_demog()

        scalex=np.squeeze(demog2/self.n_pop*self.timestep)
        
        q={}
#         q['palkansaajia']=np.sum(self.empstate[:,1])
#         q['ansiosidonnaisella']=np.sum(self.empstate[:,0]+self.empstate[:,4])
#         q['tmtuella']=np.sum(self.empstate[:,13])
#         q['isyysvapaalla']=np.sum(self.empstate[:,6])
#         q['kotihoidontuella']=np.sum(self.empstate[:,7])
#         q['vanhempainvapaalla']=np.sum(self.empstate[:,5])
        if self.version>0:
            q['yhteensä']=np.sum(np.sum(self.empstate[:,:],1)*scalex)
            q['palkansaajia']=np.sum((self.empstate[:,1]+self.empstate[:,10]+self.empstate[:,8]+self.empstate[:,9])*scalex)
            q['ansiosidonnaisella']=np.sum((self.empstate[:,0]+self.empstate[:,4])*scalex)
            q['tmtuella']=np.sum(self.empstate[:,13]*scalex)
            q['isyysvapaalla']=np.sum(self.empstate[:,6]*scalex)
            q['kotihoidontuella']=np.sum(self.empstate[:,7]*scalex)
            q['vanhempainvapaalla']=np.sum(self.empstate[:,5]*scalex)
        else:
            q['yhteensä']=np.sum(np.sum(self.empstate[:,:],1)*scalex)
            q['palkansaajia']=np.sum((self.empstate[:,1])*scalex)
            q['ansiosidonnaisella']=np.sum((self.empstate[:,0])*scalex)
            q['tmtuella']=np.sum(self.empstate[:,1]*0)
            q['isyysvapaalla']=np.sum(self.empstate[:,1]*0)
            q['kotihoidontuella']=np.sum(self.empstate[:,1]*0)
            q['vanhempainvapaalla']=np.sum(self.empstate[:,1]*0)
        
        return q

    def stat_participants_2018(self,scale=False):
        '''
        Lukumäärätiedot (EI HTV!)
        '''
        demog,demog2=self.get_demog()

        scalex=demog2/self.n_pop
        
        q={}
        q['yhteensä']=np.sum(demog2)
        q['palkansaajia']=2_204_000 # TK
        q['ansiosidonnaisella']=116_972+27_157  # Kelan tilasto 31.12.2018
        q['tmtuella']=189_780  # Kelan tilasto 31.12.2018
        q['isyysvapaalla']=59_640 # Kelan tilasto 2018
        q['kotihoidontuella']=42_042 # saajia Kelan tilasto 2018
        q['vanhempainvapaalla']=84_387 # Kelan tilasto 2018

        return q

    def stat_days_2018(self,scale=False):
        '''
        HTV-tiedot
        '''
        demog,demog2=self.get_demog()

        scalex=demog2/self.n_pop
        htv=6*52
        htv_tt=21.5*12
        
        q={}
        q['yhteensä']=np.sum(np.sum(demog2))
        q['palkansaajia']=2_204_000 # TK
        q['ansiosidonnaisella']=(30_676_200+7_553_200)/htv_tt  # Kelan tilasto 31.12.2018
        q['tmtuella']=49_880_300/htv_tt   # Kelan tilasto 31.12.2018
        q['isyysvapaalla']=1_424_000/htv # Kelan tilasto 2018
        q['kotihoidontuella']=42_042  # saajia Kelan tilasto 2018
        q['vanhempainvapaalla']=12_571_400/htv  # Kelan tilasto 2018, äideille

        return q

    def compare_with(self,cc2,label2='perus',label1='vaihtoehto',grayscale=True,figname=None,dash=False):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        diff_emp=self.empstate/self.n_pop-cc2.empstate/cc2.n_pop
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        #x=range(self.age_min,self.age_min+self.n_time)

        if self.minimal>0:
            s=20
            e=70
        else:
            s=21
            e=63.5

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1=self.comp_employed(self.empstate)
        tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2=self.comp_employed(cc2.empstate)
        htv1,tyoll1,haj1,tyollaste1,tyolliset1,osatyolliset1,kokotyolliset1,osata1,kokota1=self.comp_tyollisyys_stats(self.empstate/self.n_pop,scale_time=True,start=s,end=e,full=True)
        htv2,tyoll2,haj2,tyollaste2,tyolliset2,osatyolliset2,kokotyolliset2,osata2,kokota2=self.comp_tyollisyys_stats(cc2.empstate/cc2.n_pop,scale_time=True,start=s,end=e,full=True)
        ansiosid_osuus1,tm_osuus1=self.comp_employed_detailed(self.empstate)
        ansiosid_osuus2,tm_osuus2=self.comp_employed_detailed(cc2.empstate)
        
        q1=self.comp_budget(scale=True)
        q2=cc2.comp_budget(scale=True)
        
        df1 = pd.DataFrame.from_dict(q1,orient='index',columns=[label1])
        df2 = pd.DataFrame.from_dict(q2,orient='index',columns=['one'])
        df=df1.copy()
        df[label2]=df2['one']
        df['ero']=df1[label1]-df2['one']

        if self.version>0:
            print('Rahavirrat skaalattuna väestötasolle')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))
        
        if dash:
            ls='--'
        else:
            ls=None
        
        fig,ax=plt.subplots()
        ax.set_xlabel('Age [y]')
        ax.set_ylabel('Employment rate [%]')
        ax.plot(x,100*tyolliset1,label=label1)
        ax.plot(x,100*tyolliset2,ls=ls,label=label2)
        ax.set_ylim([0,100])  
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'emp.eps', format='eps')
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Ero osuuksissa [%]')
        diff_emp=diff_emp*100
        ax.plot(x,100*(tyot_osuus1-tyot_osuus2),label='unemployment')
        ax.plot(x,100*(kokotyo_osuus1-kokotyo_osuus2),label='fulltime work')
        if self.version>0:
            ax.plot(x,100*(osatyo_osuus1-osatyo_osuus2),label='osa-aikatyö')
            ax.plot(x,100*(tyolliset1-tyolliset2),label='työ yhteensä')
            ax.plot(x,100*(htv_osuus1-htv_osuus2),label='htv yhteensä')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Age [y]')
        ax.set_ylabel('Unemployment rate [%]')
        diff_emp=diff_emp*100
        ax.plot(x,100*tyot_osuus1,label=label1)
        ax.plot(x,100*tyot_osuus2,label=label2)
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'unemp.eps', format='eps')
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Ero osuuksissa [%]')
        diff_emp=diff_emp*100
        ax.plot(x,100*ansiosid_osuus2,label='ansiosid. työttömyys, '+label2)
        ax.plot(x,100*ansiosid_osuus1,label='ansiosid. työttömyys, '+label1)
        ax.plot(x,100*tm_osuus2,label='tm-tuki, '+label2)
        ax.plot(x,100*tm_osuus1,label='tm-tuki, '+label1)
        ax.legend()
        plt.show()
        
        print('Työllisyysvaikutus {:.0f}-{:.0f}-vuotiaisiin noin {:.0f} htv ja {:.0f} työllistä'.format(s,e,htv1-htv2,tyoll1-tyoll2))
        print('- kokoaikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(kokotyolliset1-kokotyolliset2),kokotyolliset1,kokotyolliset2))
        print('- osa-aikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(osatyolliset1-osatyolliset2),osatyolliset1,osatyolliset2))
        print('Työllisiä {:.0f} vs {:.0f} htv'.format(htv1,htv2))
        print('Työllisyysastevaikutus {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(tyollaste1-tyollaste2)*100,tyollaste1*100,tyollaste2*100))
        print('- kokoaikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(kokota1-kokota2)*100,kokota1*100,kokota2*100))
        print('- osa-aikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(osata1-osata2)*100,osata1*100,osata2*100))
        
        # epävarmuus
        delta=1.96*1.0/np.sqrt(self.n_pop)
        print('epävarmuus työllisyysasteissa {:.4f}, hajonta {:.4f}'.format(delta,haj1))
        
        if True:
            unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
            tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2=cc2.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
            tyoll_distrib2,tyoll_distrib_bu2=cc2.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        
            self.plot_compare_empdistribs(emp_distrib,emp_distrib2)
            print('Jakauma ansiosidonnainen+tmtuki+putki, no max age')
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1=label1,label2=label2)
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1=label1,label2=label2,logy=False)
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1=label1,label2=label2,logy=False,diff=True)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=False,label1=label1,label2=label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=True,label1=label1,label2=label2)     

            unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
            tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2=cc2.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
            tyoll_distrib2,tyoll_distrib_bu2=cc2.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
        
            self.plot_compare_empdistribs(emp_distrib,emp_distrib2)
            print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1=label1,label2=label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=False,label1=label1,label2=label2)     
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=True,label1=label1,label2=label2)     
        
        print(label2)
        keskikesto=self.comp_unemp_durations(return_q=False)
        self.plot_unemp_durdistribs(keskikesto)
        
        print(label1)
        keskikesto=cc2.comp_unemp_durations(return_q=False)
        self.plot_unemp_durdistribs(keskikesto)
        
        
        tyoll_virta,tyot_virta=self.comp_virrat(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        tyoll_virta2,tyot_virta2=cc2.comp_virrat(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        self.plot_compare_virrat(tyoll_virta,tyoll_virta2,virta_label='Työllisyys')
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=40,max_time=64,virta_label='Työttömyys')
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=55,max_time=64,virta_label='Työttömyys')

        tyoll_virta,tyot_virta=self.comp_virrat(ansiosid=True,tmtuki=False,putki=True,outsider=False)
        tyoll_virta2,tyot_virta2=cc2.comp_virrat(ansiosid=True,tmtuki=False,putki=True,outsider=False)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=40,max_time=64,virta_label='ei-tm-Työttömyys')
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=55,max_time=64,virta_label='ei-tm-Työttömyys')

        tyoll_virta,tyot_virta=self.comp_virrat(ansiosid=False,tmtuki=True,putki=True,outsider=False)
        tyoll_virta2,tyot_virta2=cc2.comp_virrat(ansiosid=False,tmtuki=True,putki=True,outsider=False)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=40,max_time=64,virta_label='tm-Työttömyys')
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=55,max_time=64,virta_label='tm-Työttömyys')
        
    def comp_employed(self,emp):
        nn=np.sum(emp,1)
        if self.minimal:
            tyoll_osuus=emp[:,1]/nn
            tyot_osuus=emp[:,0]/nn
            htv_osuus=emp[:,1]/nn
            kokotyo_osuus=tyoll_osuus
            osatyo_osuus=0*tyoll_osuus
            tyoll_osuus=np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            tyot_osuus=np.reshape(tyot_osuus,(tyot_osuus.shape[0],1))
            htv_osuus=np.reshape(htv_osuus,(htv_osuus.shape[0],1))
            kokotyo_osuus=np.reshape(kokotyo_osuus,(osatyo_osuus.shape[0],1))
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö 
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            tyoll_osuus=(emp[:,1]+emp[:,8]+emp[:,9]+emp[:,10])/nn
            tyot_osuus=(emp[:,0]+emp[:,4]+emp[:,13])/nn
            htv_osuus=(emp[:,1]+0.5*emp[:,8]+emp[:,9]+0.5*emp[:,10])/nn
            kokotyo_osuus=(emp[:,1]+emp[:,9])/nn
            osatyo_osuus=(emp[:,8]+emp[:,10])/nn
            
            tyoll_osuus=np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            tyot_osuus=np.reshape(tyot_osuus,(tyot_osuus.shape[0],1))
            htv_osuus=np.reshape(htv_osuus,(htv_osuus.shape[0],1))
            osatyo_osuus=np.reshape(osatyo_osuus,(osatyo_osuus.shape[0],1))
            kokotyo_osuus=np.reshape(kokotyo_osuus,(osatyo_osuus.shape[0],1))
            
        return tyoll_osuus,htv_osuus,tyot_osuus,kokotyo_osuus,osatyo_osuus

    def comp_employed_detailed(self,emp):
        if self.minimal:
            ansiosid_osuus=emp[:,0]/np.sum(emp,1)
            tm_osuus=ansiosid_osuus*0
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö 
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            ansiosid_osuus=(emp[:,0]+emp[:,4])/np.sum(emp,1)
            tm_osuus=(emp[:,13])/np.sum(emp,1)
            
        return ansiosid_osuus,tm_osuus

            
    def get_demog(self):
        # vuosi 2019
        if self.year==2019:
            demog=np.array([60796,59796,62211,64230,66911,69316,69568,72329,71682,72790,  # 20-29 y
                            71313,71434,68069,68967,70985,72961,74223,73437,70361,69800,  # 30-39 y
                            69056,69134,69773,69996,68516,65276,59488,61407,63138,65064,  # 40-49 y
                            66339,70970,72571,72862,73052,74260,74821,73880,73260,72650,  # 50-59 y
                            72007,69453,72098,73585,72476,72173,70726,72730,69771,71560,  # 60-69 y
                            73079,73689,72408,69492,61063,47990,44814,35182,49918,34273  # 70-79 y
                            ])
        else: # 2018
            demog=np.array([59640,61894,63720,66388,68734,69063,71766,71153,72182,70708,
                            70898,67638,68548,70664,72648,73972,73161,70145,69593,68880,
                            69052,69678,69956,68421,65258,59442,61378,63178,65147,66433,
                            71078,72753,73031,73207,74480,75114,74138,73574,72991,72416,
                            69831,72549,74158,73129,72894,71459,73497,70572,72530,74150,
                            74793,73683,70796,62308,49080,45971,36179,51476,35499,40134  # 70-79 y
                            ])
                        
        demog2=np.zeros((self.n_time,1))
        k2=0
        for k in np.arange(self.min_age,self.max_age,self.timestep):
            ind=int(np.floor(k))-self.min_age
            demog2[k2]=demog[ind]
            k2+=1
            
        demog2[-2:]=demog2[-3]

        return demog,demog2
        
    def comp_tyollisyys_stats(self,emp,scale_time=True,start=20,end=63.5,full=False,tyot_stats=False,shapes=False):
        demog,demog2=self.get_demog()
              
        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1
        
        tyollosuus,htvosuus,tyot_osuus,kokotyo_osuus,osatyo_osuus=self.comp_employed(emp)
        
        d=np.squeeze(demog2[min_cage:max_cage])
        htv=np.round(scale*np.sum(d*np.squeeze(htvosuus[min_cage:max_cage])))
        tyollvaikutus=np.round(scale*np.sum(d*np.squeeze(tyollosuus[min_cage:max_cage])))
        tyottomat=np.round(scale*np.sum(d*np.squeeze(tyot_osuus[min_cage:max_cage])))
        osatyollvaikutus=np.round(scale*np.sum(d*np.squeeze(osatyo_osuus[min_cage:max_cage])))
        kokotyollvaikutus=np.round(scale*np.sum(d*np.squeeze(kokotyo_osuus[min_cage:max_cage])))
        haj=np.mean(np.std(tyollosuus[min_cage:max_cage]))
        
        tyollaste=tyollvaikutus/(np.sum(d)*scale)
        osatyollaste=osatyollvaikutus/(np.sum(d)*scale)
        kokotyollaste=kokotyollvaikutus/(np.sum(d)*scale)
        
        if tyot_stats:
            d2=np.squeeze(demog2)
            tyolliset_ika=np.squeeze(scale*d2*np.squeeze(htvosuus))
            tyottomat_ika=np.squeeze(scale*d2*np.squeeze(tyot_osuus))
            htv_ika=np.squeeze(scale*d2*np.squeeze(htvosuus))
            tyolliset_osuus=np.squeeze(tyollosuus)
            tyottomat_osuus=np.squeeze(tyot_osuus)
            return tyolliset_ika,tyottomat_ika,htv_ika,tyolliset_osuus,tyottomat_osuus
        else:
            if full:
                return htv,tyollvaikutus,haj,tyollaste,tyollosuus,osatyollvaikutus,kokotyollvaikutus,osatyollaste,kokotyollaste
            else:
                return htv,tyollvaikutus,haj,tyollaste,tyollosuus

    def comp_state_stats(self,state,scale_time=True,start=20.25,end=63.5):
        demog,demog2=self.get_demog()
              
        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1
        
        vaikutus=np.round(scale*np.sum(demog2[min_cage:max_cage]*state[min_cage:max_cage]))
            
        return vaikutus

    def get_reward(self):
        return np.sum(self.rewstate)/self.n_pop

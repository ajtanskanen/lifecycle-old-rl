'''

    episodestats.py

    implements statistic that are used in producing employment statistics for the
    lifecycle model

'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class EpisodeStats():
    def __init__(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage):
        self.reset(timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage)

    def reset(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage):
        self.min_age=min_age
        self.max_age=max_age
        self.min_retirementage=min_retirementage
        self.minimal=minimal
        self.n_employment=n_emps
        self.n_time=n_time
        self.timestep=timestep # 0.25 = 3kk askel
        self.inv_timestep=int(np.round(1/self.timestep)) # pitöisi olla kokonaisluku
        self.n_pop=n_pop
        self.env=env
        
        if self.minimal:
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

    def add(self,n,act,r,state,newstate,debug=False,plot=False,aveV=None): 
        if self.minimal:
            emp,_,_,a,_=self.env.state_decode(state) # current employment state
            newemp,_,newsal,a2,tis=self.env.state_decode(newstate)
            g=0
            bu=0
        else:
            emp,_,_,_,a,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
            newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,ura,oof,bu,wr,pr=self.env.state_decode(newstate)
    
        t=int(np.round((a2-self.min_age)*self.inv_timestep))
        
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            if a2>self.min_retirementage and newemp==3:
                newemp=2
            self.empstate[t,newemp]+=1
            self.alive[t]+=1
            self.rewstate[t,newemp]+=r
            self.actions[t,n]=act
            self.popempstate[t,n]=newemp
            self.salaries[t,n]=newsal
            self.salaries_emp[t,newemp]+=newsal
            self.time_in_state[t,newemp]+=tis
            if not self.minimal:
                self.gempstate[t,newemp,g]+=1
                self.stat_wage_reduction[t,newemp]+=wr
                self.galive[t,g]+=1
                self.stat_tyoura[t,newemp]+=ura
                self.stat_toe[t,newemp]+=toe
                self.stat_pension[t,newemp]+=newpen
                self.stat_paidpension[t,newemp]+=paidpens
                self.stat_unemp_len[t,n]=tis
                self.out_of_work[t,newemp]+=oof
                self.popunemprightleft[t,n]=-self.env.unempright_left(newemp,tis,bu,a2,ura,oof)
                self.popunemprightused[t,n]=bu
    
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
        
    def comp_tyollistymisdistribs(self,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,max_age=100):
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
        
        for k in range(self.n_pop):
            prev_state=self.popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if self.popempstate[t,k]!=prev_state:
                        if prev_state in unempset and self.popempstate[t,k] in empset:
                            tyoll_distrib.append((t-prev_trans)*self.timestep)
                            tyoll_distrib_bu.append(self.popunemprightleft[t,k])
                            prev_state=self.popempstate[t,k]
                            prev_trans=t
                        else: # some other state
                            prev_state=self.popempstate[t,k]
                            prev_trans=t
                    
        return tyoll_distrib,tyoll_distrib_bu

    def comp_empdistribs(self,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,max_age=100):
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
            
        empset=set([1,10])
        unempset=set(unempset)
        
        for k in range(self.n_pop):
            prev_state=self.popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if self.popempstate[t,k]!=prev_state:
                        if prev_state in unempset and self.popempstate[t,k] not in unempset:
                            unemp_distrib.append((t-prev_trans)*self.timestep)
                            unemp_distrib_bu.append(self.popunemprightleft[t,k])
                            
                            prev_state=self.popempstate[t,k]
                            prev_trans=t
                        elif prev_state in empset and self.popempstate[t,k] not in unempset:
                            emp_distrib.append((t-prev_trans)*self.timestep)
                            prev_state=self.popempstate[t,k]
                            prev_trans=t
                        else: # some other state
                            prev_state=self.popempstate[t,k]
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
        
        ax.plot(x2[:-1],scaled,label=label)
        ax.plot(x3[:-1],scaled3,label='perus')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
        plt.show()
        
    def plot_vlines_unemp(self,point=0):
        axvcolor='r'
        plt.axvline(x=300/(12*21.5),color=axvcolor)
        plt.text(310/(12*21.5),point,'300',rotation=90)        
        plt.axvline(x=400/(12*21.5),color=axvcolor)
        plt.text(410/(12*21.5),point,'400',rotation=90)        
        plt.axvline(x=500/(12*21.5),color=axvcolor)
        plt.text(510/(12*21.5),point,'500',rotation=90)                    
        
    def plot_tyolldistribs(self,emp_distrib,tyoll_distrib,tyollistyneet=True,max=10):
        max_time=55
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        if not tyollistyneet:
            scaled=scaled0
            x2=x0
        else:
            scaled,x2=np.histogram(tyoll_distrib,x)
        jaljella=np.cumsum(scaled0[::-1])[::-1] # jöljellö olevien summa
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
        ax.bar(x2[1:-1],scaled[1:],align='center',width=self.timestep)
        plt.xlim(0,max)
        plt.show()        

    def plot_tyolldistribs_both(self,emp_distrib,tyoll_distrib,max=10):
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        scaled=scaled0
        scaled_tyoll,x2=np.histogram(tyoll_distrib,x)
            
        jaljella=np.cumsum(scaled0[::-1])[::-1] # jöljellö olevien summa
        scaled=scaled/jaljella
        jaljella_tyoll=np.cumsum(scaled0[::-1])[::-1] # jöljellö olevien summa
        scaled_tyoll=scaled_tyoll/jaljella_tyoll
        fig,ax=plt.subplots()
        ax.set_xlabel('työttömyysjakson pituus [v]')
        point=0.6
        self.plot_vlines_unemp(point)
        ax.plot(x2[1:-1],scaled[1:],label='pois siirtyneiden osuus')
        ax.plot(x2[1:-1],scaled_tyoll[1:],label='työllistyneiden osuus')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_ylabel('pois siirtyneiden osuus')

        plt.xlim(0,max)
        plt.ylim(0,0.8)
        plt.show()        

    def plot_tyolldistribs_both_bu(self,emp_distrib,tyoll_distrib,max=2):
        max_time=4
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(-max_time,0,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        scaled=scaled0
        scaled_tyoll,x2=np.histogram(tyoll_distrib,x)
            
        jaljella=np.cumsum(scaled0[::-1])[::-1] # jöljellö olevien summa
        #jaljella=np.cumsum(scaled0)
        #print(jaljella,scaled)
        scaled=scaled/jaljella
        jaljella_tyoll=np.cumsum(scaled0[::-1])[::-1] # jöljellö olevien summa
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

    def plot_compare_tyolldistribs(self,emp_distrib,tyoll_distrib,emp_distrib2,tyoll_distrib2,tyollistyneet=True,max=4,label='vaihtoehto'):
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        if not tyollistyneet:
            scaled=scaled0
            x2=x0
        else:
            scaled,x2=np.histogram(tyoll_distrib,x)
        #scaled,x2=np.histogram(emp_distrib,x)
        jaljella=np.cumsum(scaled0[::-1])[::-1] # jöljellö olevien summa
        scaled=scaled/jaljella
        scaled0,x0=np.histogram(emp_distrib2,x)
        if not tyollistyneet:
            scaled3=scaled0
            x3=x0
        else:
            scaled3,x3=np.histogram(tyoll_distrib2,x)
        #scaled3,x3=np.histogram(emp_distrib2,x)
        jaljella3=np.cumsum(scaled0[::-1])[::-1] # jöljellö olevien summa
        scaled3=scaled3/jaljella3
        #print(scaled,jaljella,scaled.shape,x2.shape)
        fig,ax=plt.subplots()
        ax.set_xlabel('työttömyysjakson pituus [v]')
        if tyollistyneet:
            ax.set_ylabel('työllistyneiden osuus')
        else:
            ax.set_ylabel('pois siirtyneiden osuus')
        self.plot_vlines_unemp()
        #ax.bar(x2[1:-1],scaled[1:])
        ax.plot(x2[1:-1],scaled[1:],label=label)
        ax.plot(x3[1:-1],scaled3[1:],label='perus')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlim(0,max)
        plt.show()        
                
    def plot_unempdistribs(self,unemp_distrib,max=10):
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

    def plot_compare_unempdistribs(self,unemp_distrib,unemp_distrib2,max=4,label='vaihtoehto'):
        #fig,ax=plt.subplots()
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(self.timestep,max_time,nn_time)
        scaled,x2=np.histogram(unemp_distrib,x)
        print('Perus keskikesto {} v {} Keskikesto {} v'.format(np.mean(unemp_distrib2),label,np.mean(unemp_distrib)))
        print('Skaalaamaton Perus lkm {} v {} lkm {} v'.format(len(unemp_distrib2),label,len(unemp_distrib)))
        print('Skaalaamaton Perus työtpäiviä yht {} v {} työtpäiviä yht {} v'.format(sum(unemp_distrib2),label,sum(unemp_distrib)))
        #scaled=scaled/np.sum(unemp_distrib)
        scaled=scaled/np.sum(scaled)
        scaled3,x3=np.histogram(unemp_distrib2,x)
        #scaled3=scaled3/np.sum(unemp_distrib2)
        scaled3=scaled3/np.sum(scaled3)
        fig,ax=plt.subplots()
        self.plot_vlines_unemp(0.9)
        ax.set_xlabel('Työttömyysjakson pituus [v]')
        ax.set_ylabel('Osuus')
        ax.plot(x[:-1],scaled,label=label)
        ax.plot(x[:-1],scaled3,label='perus')
        ax.set_yscale('log')
        plt.ylim(1e-4,1.0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlim(0,max)
        plt.show()   

    def comp_empratios(self,emp,alive,unempratio=True):
        employed=emp[:,1]
        retired=emp[:,2]
        unemployed=emp[:,0]

        if not self.minimal:
            disabled=emp[:,3]
            piped=emp[:,4]
            mother=emp[:,5]
            dad=emp[:,6]
            kotihoidontuki=emp[:,7]
            vetyo=emp[:,8]
            veosatyo=emp[:,9]
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

    def comp_gempratios(self,emp,alive,unempratio=True):
        if female==True:
            g=np.arange(4,6)
        else:
            g=np.arange(0,2)
        employed=emp[:,1,g]
        retired=emp[:,2,g]
        unemployed=emp[:,0,g]

        if not self.minimal:
            disabled=emp[:,3,g]
            piped=emp[:,4,g]
            mother=emp[:,5,g]
            dad=emp[:,6,g]
            kotihoidontuki=emp[:,7,g]
            vetyo=emp[:,8,g]
            veosatyo=emp[:,9,g]
            osatyo=emp[:,10,g]
            outsider=emp[:,11,g]
            student=emp[:,12,g]
            tyomarkkinatuki=emp[:,13,g]
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

    def plot_outsider(self):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*(self.empstate[:,11]+self.empstate[:,5]+self.empstate[:,6]+self.empstate[:,7])/self.alive[:,0],label='työvoiman ulkopuolella, ei opiskelija, sis. vanh.vapaat')
        emp_statsratio=100*self.outsider_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
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

    def plot_emp(self):

        tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(self.empstate,self.alive,unempratio=False)

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,tyollisyysaste,label='työllisyysaste')
        ax.plot(x,tyottomyysaste,label='työttömien osuus')
        emp_statsratio=100*self.emp_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend()
        plt.show()

        if not self.minimal:
            fig,ax=plt.subplots()
            ax.stackplot(x,osatyoaste,100-osatyoaste,
                        labels=('osatyössä','kokoaikaisessa työssä')) #, colors=pal) pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
            ax.legend()
            plt.show()

        empstate_ratio=100*self.empstate/self.alive
        self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',stack=True)

        if not self.minimal:
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',ylimit=20,stack=False)

        if not self.minimal:
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',parent=True,stack=False)
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',unemp=True,stack=False)

        self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',start_from=60,stack=True)

    def plot_unemp(self,unempratio=True):
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
        #print(labeli)
        ax.plot(x,unempratio_stat,label='havainto')
        ax.plot(x,tyottomyysaste)
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
            else:
                gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
                leg='Naiset'
        
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive,unempratio=unempratio)
        
            ax.plot(x,tyottomyysaste,label='{} {}'.format(labeli2,leg))
            
        if unempratio:
            ax.plot(x,100*self.unempratio_stats(g=1),label='havainto, naiset')
            ax.plot(x,100*self.unempratio_stats(g=2),label='havainto, miehet')
            labeli='keskimääräinen työttömyysaste '+str(ka_tyottomyysaste)      
            ylabeli='Työttömyysaste [%]'
        else:
            ax.plot(x,100*self.unemp_stats(g=1),label='havainto, naiset')
            ax.plot(x,100*self.unemp_stats(g=2),label='havainto, miehet')
            labeli='keskimääräinen työttömien osuus väestöstö '+str(ka_tyottomyysaste)
            ylabeli='Työttömien osuus väestöstö [%]'
            
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel(ylabeli)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()        

    def plot_parttime_ratio(self):
        '''
        Plottaa osatyötä tekevien osuus väestöstö
        '''
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        labeli2='Osatyötä tekevien osuus'
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='Miehet'
                gempstate=np.sum(self.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
            else:
                gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
                leg='Naiset'
        
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive,unempratio=False)
        
            ax.plot(x,osatyoaste,label='{} {}'.format(labeli2,leg))
            
            
        o_x=np.array([20,30,40,50,60,70])
        f_osatyo=np.array([55,21,16,12,18,71])
        m_osatyo=np.array([32,8,5,4,9,65])
        ax.plot(o_x,f_osatyo,label='havainto, naiset')
        ax.plot(o_x,m_osatyo,label='havainto, miehet')
        labeli='osatyöaste '#+str(ka_tyottomyysaste)      
        ylabeli='Osatyön osuus työnteosta [%]'
            
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel(ylabeli)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

        
    def plot_unemp_shares(self):
        empstate_ratio=100*self.empstate/self.alive
        self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',onlyunemp=True,stack=True)

    def plot_group_emp(self):
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='Miehet'
                gempstate=np.sum(self.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
            else:
                gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
                leg='Naiset'
        
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive)
        
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,tyollisyysaste,label='työllisyysaste {}'.format(leg))
            #ax.plot(x,tyottomyysaste,label='työttömyys {}'.format(leg))
            
        emp_statsratio=100*self.emp_stats(g=1)
        ax.plot(x,emp_statsratio,label='havainto, naiset')
        emp_statsratio=100*self.emp_stats(g=2)
        ax.plot(x,emp_statsratio,label='havainto, miehet')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def plot_pensions(self):
        if not self.minimal:
            self.plot_ratiostates(self.stat_pension,ylabel='Tuleva eläke [e/v]',stack=False)

    def plot_career(self):    
        if not self.minimal:
            self.plot_ratiostates(self.stat_tyoura,ylabel='Työuran pituus [v]',stack=False)

    def plot_ratiostates(self,statistic,ylabel='',ylimit=None, show_legend=True, parent=False,\
                         unemp=False,start_from=None,stack=False,no_ve=False):
        self.plot_states(statistic/self.empstate,ylabel=ylabel,ylimit=ylimit,no_ve=no_ve,\
                    show_legend=show_legend,parent=parent,unemp=unemp,start_from=start_from,\
                    stack=stack)

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
                    start_from=None,stack=True,save=False,filename='fig.png',yminlim=None,ymaxlim=None,onlyunemp=False):
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
        if not self.minimal:
            ura_disab=statistic[:,3]
            ura_pipe=statistic[:,4]
            ura_mother=statistic[:,5]
            ura_dad=statistic[:,6]
            ura_kht=statistic[:,7]
            ura_vetyo=statistic[:,8]
            ura_veosatyo=statistic[:,9]
            ura_osatyo=statistic[:,10]
            ura_outsider=statistic[:,11]
            ura_student=statistic[:,12]
            ura_tyomarkkinatuki=statistic[:,13]
            ura_army=statistic[:,14]

        if no_ve:
            ura_ret=0*ura_ret

        fig,ax=plt.subplots()
        if stack:
            pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
            #alpha=0.8
            if parent:
                if not self.minimal:
                    ax.stackplot(x,ura_mother,ura_dad,ura_kht,
                        labels=('äitiysvapaa','isyysvapaa','khtuki'), colors=pal)
            elif unemp:
                if not self.minimal:
                    ax.stackplot(x,ura_unemp,ura_pipe,ura_student,ura_outsider,ura_tyomarkkinatuki,
                        labels=('tyött','putki','opiskelija','ulkona','tm-tuki'), colors=pal)
                else:
                    ax.stackplot(x,ura_unemp,labels=('tyött'), colors=pal)
            elif onlyunemp:
                if not self.minimal:
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
                if not self.minimal:
                    ax.stackplot(x,ura_emp,ura_osatyo,ura_vetyo,ura_veosatyo,ura_unemp,ura_tyomarkkinatuki,ura_pipe,ura_disab,ura_mother,ura_dad,ura_kht,ura_ret,ura_student,ura_outsider,ura_army,
                        labels=('työssä','osatyö','ve+työ','ve+osatyö','työtön','tm-tuki','työttömyysputki','tk-eläke','äitiysvapaa','isyysvapaa','kh-tuki','vanhuuseläke','opiskelija','työvoiman ulkop.','armeijassa'), 
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
                if not self.minimal:
                    ax.plot(x,ura_mother,label='äitiysvapaa')
                    ax.plot(x,ura_dad,label='isyysvapaa')
                    ax.plot(x,ura_kht,label='khtuki')
            elif unemp:
                ax.plot(x,ura_unemp,label='tyött')
                if not self.minimal:
                    ax.plot(x,ura_tyomarkkinatuki,label='tm-tuki')
                    ax.plot(x,ura_student,label='student')
                    ax.plot(x,ura_outsider,label='outsider')
                    ax.plot(x,ura_pipe,label='putki')
            else:
                ax.plot(x,ura_unemp,label='tyött')
                ax.plot(x,ura_ret,label='eläke')
                ax.plot(x,ura_emp,label='työ')
                if not self.minimal:
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
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if ylimit is not None:
            ax.set_ylim([0,ylimit])  
        #fig.tight_layout()
        plt.show()

        if save:
            plt.savefig(filename,bbox_inches='tight')

    def plot_toe(self):    
        if not self.minimal:
            self.plot_ratiostates(self.stat_toe,'työssäolo-ehdon pituus 28 kk aikana [v]',stack=False)
            
    def plot_oof(self):
        # oof-dataa ei lasketa
        #if not self.minimal:
        #    self.plot_ratiostates(self.out_of_work,'Poissa työstö [v]',stack=False)
        return

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

        rr=np.sum(total_reward)/self.n_pop
        print('yhteensä reward {r}'.format(r=rr))

    def plot_wage_reduction(self):
        self.plot_ratiostates(self.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False)
        self.plot_ratiostates(self.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False,unemp=True)
        #self.plot_ratiostates(np.log(1.0+self.stat_wage_reduction),ylabel='log 5wage-reduction tilassa',stack=False)

    def plot_distrib(self,label='',plot_emp=False,plot_bu=False,ansiosid=False,tmtuki=False,putki=False,outsider=False,max_age=500,laaja=False,max=4):
        unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=ansiosid,tmtuki=tmtuki,putki=putki,outsider=outsider,max_age=max_age,laaja=laaja)
        tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=ansiosid,tmtuki=tmtuki,putki=putki,outsider=outsider,max_age=max_age,laaja=laaja)
        if plot_emp:
            self.plot_empdistribs(emp_distrib)
        if plot_bu:
            self.plot_unempdistribs_bu(unemp_distrib_bu)
        else:
            self.plot_unempdistribs(unemp_distrib)
        #self.plot_tyolldistribs(unemp_distrib,tyoll_distrib,tyollistyneet=False)     
        if plot_bu:
            self.plot_tyolldistribs_both_bu(unemp_distrib_bu,tyoll_distrib_bu,max=max)
        else:
            self.plot_tyolldistribs_both(unemp_distrib,tyoll_distrib,max=max)

    def plot_stats(self):
        self.plot_emp()
        print('Lisäpäivillä on {:.0f} henkilöä'.format(self.count_putki()))
        self.plot_unemp(unempratio=True)
        self.plot_unemp(unempratio=False)
        self.plot_unemp_shares()
        if not self.minimal:
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
        self.plot_oof()
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
        Löhde: Tilastokeskus
        '''
        if g==0: # kaikki
            emp=np.array([0.461,0.545,0.567,0.599,0.645,0.678,0.706,0.728,0.740,0.752,0.758,0.769,0.776,0.781,0.787,0.795,0.801,0.807,0.809,0.820,0.820,0.829,0.831,0.833,0.832,0.828,0.827,0.824,0.822,0.817,0.815,0.813,0.807,0.802,0.796,0.788,0.772,0.763,0.752,0.728,0.686,0.630,0.568,0.382,0.217,0.142,0.106,0.086,0.011,0.003,0.002])
        elif g==1: # naiset
            emp=np.array([ 0.554,0.567,0.584,0.621,0.666,0.685,0.702,0.723,0.727,0.734,0.741,0.749,0.753,0.762,0.768,0.777,0.788,0.793,0.798,0.813,0.816,0.827,0.832,0.834,0.835,0.835,0.833,0.836,0.833,0.831,0.828,0.827,0.824,0.821,0.815,0.812,0.798,0.791,0.779,0.758,0.715,0.664,0.596,0.400,0.220,0.136,0.098,0.079,0.011,0.004,0.002 ])
        else: # miehet
            emp=np.array([ 0.374,0.524,0.550,0.579,0.626,0.671,0.710,0.733,0.752,0.769,0.774,0.788,0.798,0.800,0.805,0.812,0.814,0.820,0.819,0.826,0.824,0.831,0.831,0.831,0.828,0.821,0.822,0.813,0.811,0.803,0.803,0.798,0.790,0.783,0.776,0.764,0.746,0.735,0.723,0.696,0.656,0.596,0.539,0.362,0.214,0.148,0.115,0.094,0.012,0.002,0.002 ])

        return self.map_ratios(emp)
        
        
    def disab_stat(self,g):
        '''
        Työkyvyttömyyselökkeellö olevien osuus väestöstö
        Löhde: ETK
        '''
        if g==1:
            ratio=np.array([0.014412417,0.017866162,0.019956194,0.019219484,0.022074491,0.022873602,0.024247334,0.025981477,0.025087389,0.023162522,0.025013013,0.023496079,0.025713399,0.025633996,0.028251301,0.028930719,0.028930188,0.030955287,0.031211716,0.030980726,0.035395247,0.03522291,0.035834422,0.036878386,0.040316277,0.044732619,0.046460599,0.050652725,0.054797849,0.057018324,0.0627497,0.067904263,0.072840649,0.079978222,0.083953327,0.092811744,0.106671337,0.119490669,0.129239815,0.149503982,0.179130081,0.20749958,0.22768029,0.142296259,0.135142865,0.010457403,0,0,0,0,0])
        else:
            ratio=np.array([0.0121151,0.0152247,0.0170189,0.0200570,0.0196213,0.0208018,0.0215082,0.0223155,0.0220908,0.0213913,0.0214263,0.0242843,0.0240043,0.0240721,0.0259648,0.0263371,0.0284309,0.0270143,0.0286249,0.0305952,0.0318945,0.0331264,0.0350743,0.0368707,0.0401613,0.0431067,0.0463718,0.0487914,0.0523801,0.0569297,0.0596571,0.0669273,0.0713361,0.0758116,0.0825295,0.0892805,0.1047429,0.1155854,0.1336167,0.1551418,0.1782882,0.2106220,0.2291799,0.1434176,0.1301574,0.0110726,0,0,0,0,0])
            
        return self.map_ratios(ratio)
        
    def student_stats(self,g=0):
        '''
        Opiskelijoiden osuus väestöstö
        Löhde: Tilastokeskus
        '''
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
        Löhde: Tilastokeskus
        '''
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
        Löhde: Tilastokeskus
        '''
        if g==0: # kaikki
            emp_ratio=np.array([ 0.115,0.070,0.065,0.066,0.066,0.069,0.073,0.075,0.077,0.079,0.079,0.079,0.076,0.075,0.072,0.067,0.065,0.063,0.062,0.057,0.055,0.050,0.048,0.048,0.047,0.046,0.045,0.044,0.042,0.044,0.042,0.043,0.042,0.043,0.043,0.044,0.045,0.045,0.045,0.044,0.044,0.040,0.038,0.022,0.010,0.007,0.004,0.004,0.004,0.004,0.004 ])
        elif g==1: # naiset
            emp_ratio=np.array([ 0.077,0.066,0.068,0.072,0.074,0.082,0.089,0.092,0.098,0.099,0.101,0.098,0.098,0.093,0.088,0.082,0.076,0.074,0.071,0.065,0.061,0.053,0.049,0.049,0.046,0.045,0.044,0.041,0.042,0.041,0.040,0.039,0.040,0.040,0.041,0.041,0.042,0.042,0.042,0.043,0.044,0.041,0.039,0.023,0.012,0.007,0.005,0.004,0.004,0.004,0.004 ])
        else: # miehet
            emp_ratio=np.array([ 0.151,0.074,0.063,0.060,0.057,0.057,0.057,0.059,0.058,0.059,0.059,0.061,0.056,0.057,0.058,0.053,0.054,0.052,0.053,0.051,0.050,0.046,0.047,0.046,0.047,0.046,0.047,0.046,0.042,0.046,0.044,0.046,0.045,0.047,0.044,0.046,0.047,0.048,0.048,0.045,0.044,0.039,0.037,0.021,0.009,0.007,0.004,0.004,0.005,0.003,0.004 ])

        return self.map_ratios(emp_ratio)
        
    def pensioner_stats(self,g=0):
        '''
        Elökkeellö olevien osuus väestöstö
        Löhde: Tilastokeskus
        '''
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
        Löhde: Tilastokeskus
        '''
        emp_ratio=np.zeros(self.n_time)
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
        Löhde: Tilastokeskus
        '''
        emp_ratio=self.emp_stats(g=g)
        unemp_ratio=self.unemp_stats(g=g)
        ratio=unemp_ratio/(emp_ratio+unemp_ratio)
        return ratio # ei mapata, on jo tehty!

    def save_sim(self,filename):
        f = h5py.File(filename, 'w')
        ftype='float64'
        dset = f.create_dataset('n_pop', data=self.n_pop, dtype=ftype)
        dset = f.create_dataset('empstate', data=self.empstate, dtype=ftype)
        dset = f.create_dataset('gempstate', data=self.gempstate, dtype=ftype)
        dset = f.create_dataset('deceiced', data=self.deceiced, dtype=ftype)
        dset = f.create_dataset('rewstate', data=self.rewstate, dtype=ftype)
        dset = f.create_dataset('salaries_emp', data=self.salaries_emp, dtype=ftype)
        dset = f.create_dataset('actions', data=self.actions, dtype=ftype)
        dset = f.create_dataset('alive', data=self.alive, dtype=ftype)
        dset = f.create_dataset('galive', data=self.galive, dtype=ftype)
        dset = f.create_dataset('siirtyneet', data=self.siirtyneet, dtype=ftype)
        dset = f.create_dataset('pysyneet', data=self.pysyneet, dtype=ftype)
        dset = f.create_dataset('salaries', data=self.salaries, dtype=ftype)
        dset = f.create_dataset('aveV', data=self.aveV, dtype=ftype)
        dset = f.create_dataset('time_in_state', data=self.time_in_state, dtype=ftype)
        dset = f.create_dataset('stat_tyoura', data=self.stat_tyoura, dtype=ftype)
        dset = f.create_dataset('stat_toe', data=self.stat_toe, dtype=ftype)
        dset = f.create_dataset('stat_pension', data=self.stat_pension, dtype=ftype)
        dset = f.create_dataset('stat_paidpension', data=self.stat_paidpension, dtype=ftype)
        dset = f.create_dataset('stat_unemp_len', data=self.stat_unemp_len, dtype=ftype)
        dset = f.create_dataset('popempstate', data=self.popempstate, dtype=ftype)
        dset = f.create_dataset('stat_wage_reduction', data=self.stat_wage_reduction, dtype=ftype)
        dset = f.create_dataset('popunemprightleft', data=self.popunemprightleft, dtype=ftype)
        dset = f.create_dataset('popunemprightused', data=self.popunemprightused, dtype=ftype)
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
        if 'n_pop' in f:
            self.n_pop=int(f.get('n_pop').value)
        else:
            self.n_pop=1_000
        if n_pop is not None:
            self.n_pop=n_pop
            
        self.empstate=f.get('empstate').value
        self.gempstate=f.get('gempstate').value
        self.deceiced=f.get('deceiced').value
        self.rewstate=f.get('rewstate').value
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
        
        f.close()

    def render(self,load=None):
        if load is not None:
            self.load_sim(load)

        #self.plot_stats(5)
        self.plot_stats()
        #self.plot_reward()   

    def compare_with(self,cc2,label2='vaihtoehto',label='perus'):
        diff_emp=self.empstate/self.n_pop-cc2.empstate/cc2.n_pop
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        #x=range(self.age_min,self.age_min+self.n_time)

        s=21
        e=63

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1=self.comp_employed(self.empstate)
        tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2=self.comp_employed(cc2.empstate)
        htv1,tyoll1,haj1,tyollaste1,tyolliset1=self.comp_tyollisyys_stats(self.empstate/self.n_pop,scale_time=True,start=s,end=e)
        htv2,tyoll2,haj2,tyollaste2,tyolliset2=self.comp_tyollisyys_stats(cc2.empstate/cc2.n_pop,scale_time=True,start=s,end=e)
        ansiosid_osuus1,tm_osuus1=self.comp_employed_detailed(self.empstate)
        ansiosid_osuus2,tm_osuus2=self.comp_employed_detailed(cc2.empstate)
        
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Työllisyysaste [%]')
        ax.plot(x,100*tyolliset1,label=label2)
        ax.plot(x,100*tyolliset2,label=label1)
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Ero osuuksissa [%]')
        diff_emp=diff_emp*100
        ax.plot(x,100*(tyot_osuus1-tyot_osuus2),label='työttömyys')
        ax.plot(x,100*(kokotyo_osuus1-kokotyo_osuus2),label='kokoaikatyö')
        if not self.minimal:
            ax.plot(x,100*(osatyo_osuus1-osatyo_osuus2),label='osa-aikatyö')
            ax.plot(x,100*(tyolliset1-tyolliset2),label='työ yhteensä')
            ax.plot(x,100*(htv_osuus1-htv_osuus2),label='htv yhteensä')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Ero osuuksissa [%]')
        diff_emp=diff_emp*100
        ax.plot(x,100*(tyot_osuus2),label='työttömyys, perus')
        ax.plot(x,100*tyot_osuus1,label='työttömyys, vaihtoehto')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Ero osuuksissa [%]')
        diff_emp=diff_emp*100
        ax.plot(x,100*ansiosid_osuus2,label='ansiosid. työttömyys, perus')
        ax.plot(x,100*ansiosid_osuus1,label='ansiosid. työttömyys, vaihtoehto')
        ax.plot(x,100*tm_osuus2,label='tm-tuki, perus')
        ax.plot(x,100*tm_osuus1,label='tm-tuki, vaihtoehto')
        ax.legend()
        plt.show()

        print('Työllisyysvaikutus {:.0f}-{:.0f}-vuotiaisiin noin {:.0f} htv ja {:.0f} työllistä'.format(s,e,htv1-htv2,tyoll1-tyoll2))
        print('Työllisyysastevaikutus {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(tyollaste1-tyollaste2)*100,tyollaste1*100,tyollaste2*100))
        
        # epävarmuus
        delta=1.96*1.0/np.sqrt(self.n_pop)
        print('epävarmuus työllisyysasteissa {:.4f}, hajonta {:.4f}'.format(delta,haj1))
        
        unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        unemp_distrib2,emp_distrib2,unemp_distrib_bu2=cc2.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        tyoll_distrib2,tyoll_distrib_bu2=cc2.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        
        self.plot_compare_empdistribs(emp_distrib,emp_distrib2)
        print('Jakauma ansiosidonnainen+tmtuki+putki, no max age')
        self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label=label)
        self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=False,label=label)     
        self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=True,label=label)     

        unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
        tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
        unemp_distrib2,emp_distrib2,unemp_distrib_bu2=cc2.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
        tyoll_distrib2,tyoll_distrib_bu2=cc2.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
        
        self.plot_compare_empdistribs(emp_distrib,emp_distrib2)
        print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
        self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label=label)
        self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=False,label=label)     
        self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=True,label=label)     
        
        
    def comp_employed(self,emp):
        nn=np.sum(emp,1)
        if self.minimal:
            tyoll_osuus=emp[:,1]/nn
            tyot_osuus=emp[:,0]/nn
            htv_osuus=emp[:,1]/nn
            kokotyo_osuus=tyoll_osuus
            osatyo_osuus=0
            tyoll_osuus=np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            tyot_osuus=np.reshape(tyot_osuus,(tyot_osuus.shape[0],1))
            tv_osuus=np.reshape(htv_osuus,(htv_osuus.shape[0],1))
            kokotyo_osuus=np.reshape(kokotyo_osuus,(osatyo_osuus.shape[0],1))
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö 
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            tyoll_osuus=(emp[:,1]+emp[:,8]+emp[:,9]+emp[:,10])/nn
            tyot_osuus=(emp[:,0]+emp[:,4]+emp[:,13])/nn
            htv_osuus=(emp[:,1]+emp[:,8]+0.5*emp[:,9]+0.5*emp[:,10])/nn
            kokotyo_osuus=(emp[:,1]+emp[:,8])/nn
            osatyo_osuus=(emp[:,9]+emp[:,10])/nn
            
            tyoll_osuus=np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            tyot_osuus=np.reshape(tyot_osuus,(tyot_osuus.shape[0],1))
            htv_osuus=np.reshape(htv_osuus,(htv_osuus.shape[0],1))
            osatyo_osuus=np.reshape(osatyo_osuus,(osatyo_osuus.shape[0],1))
            kokotyo_osuus=np.reshape(kokotyo_osuus,(osatyo_osuus.shape[0],1))
            
        return tyoll_osuus,htv_osuus,tyot_osuus,kokotyo_osuus,osatyo_osuus

    def comp_employed_number(self,emp):
        demog,demog2=self.get_demog()
        
        # drop singleton dim
        dem=np.squeeze(demog2)
    
        nn=np.sum(emp,1)
    
        if self.minimal:
            tyolliset_osuus=emp[:,1]/nn
            tyottomat_osuus=emp[:,0]/nn
            tyolliset=tyolliset_osuus*dem
            tyottomat=tyottomat_osuus*dem
            htv=emp[:,1]/nn
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö 
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            tyolliset_osuus=(emp[:,1]+emp[:,8]+emp[:,9]+emp[:,10])/nn
            q=(emp[:,1]+emp[:,8]+emp[:,9]+emp[:,10])
            tyottomat_osuus=(emp[:,0]+emp[:,4]+emp[:,13])/nn
            tyolliset=tyolliset_osuus*dem
            tyottomat=tyottomat_osuus*dem
            htv=(emp[:,1]+emp[:,8]+0.5*emp[:,9]+0.5*emp[:,10])/nn*dem
            tyolliset=np.reshape(tyolliset,(tyolliset.shape[0],1))
            tyottomat=np.reshape(tyottomat,(tyottomat.shape[0],1))
            tyolliset_osuus=np.reshape(tyolliset_osuus,(tyolliset_osuus.shape[0],1))
            tyottomat_osuus=np.reshape(tyottomat_osuus,(tyottomat_osuus.shape[0],1))
            htv=np.reshape(htv,(htv.shape[0],1))
            
        return tyolliset,tyottomat,htv,tyolliset_osuus,tyottomat_osuus

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
        demog=np.array([61663,63354,65939,68253,68543,71222,70675,71691,70202,70535, # 20-29 y
                        67315,68282,70431,72402,73839,73065,70040,69501,68857,69035, # 30-39 y
                        69661,69965,68429,65261,59498,61433,63308,65305,66580,71263, # 40-49 y
                        72886,73253,73454,74757,75406,74448,73940,73343,72808,70259, # 50-59 y
                        73065,74666,73766,73522,72213,74283,71273,73404,75153,75888  # 60-69 y
                        ])
        demog2=np.zeros((self.n_time,1))
        k2=0
        for k in np.arange(self.min_age,self.max_age,self.timestep):
            ind=int(np.floor(k))-self.min_age
            demog2[k2]=demog[ind]
            k2+=1

        return demog,demog2
        
    def comp_tyollisyys_stats(self,emp,scale_time=True,start=20.25,end=63.5):
        demog,demog2=self.get_demog()
              
        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1
        
        tyollosuus,htvosuus,_,_,_=self.comp_employed(emp)
        
        htv=np.round(scale*np.sum(demog2[min_cage:max_cage]*htvosuus[min_cage:max_cage]))
        tyollvaikutus=np.round(scale*np.sum(demog2[min_cage:max_cage]*tyollosuus[min_cage:max_cage]))
        haj=np.mean(np.std(tyollosuus[min_cage:max_cage]))
        
        tyollaste=tyollvaikutus/(np.sum(demog2[min_cage:max_cage])*scale)
            
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
        total_reward=np.sum(self.rewstate,axis=1)
        rr=np.sum(total_reward)/self.n_pop
        return rr

    
class SimStats(EpisodeStats):
    def run_simstats(self,results,save,n,plot=True,startn=0):
        '''
        Laskee statistiikat ajoista
        '''
        #n=self.load_hdf(results+'_simut','n')
        e_rate=np.zeros((n,self.n_time))
        agg_htv=np.zeros(n)
        agg_tyoll=np.zeros(n)
        agg_rew=np.zeros(n)
        t_aste=np.zeros(self.n_time)
        emps=np.zeros((n,self.n_time,self.n_employment))
        emp_tyolliset=np.zeros((n,self.n_time))
        emp_tyottomat=np.zeros((n,self.n_time))
        emp_tyolliset_osuus=np.zeros((n,self.n_time))
        emp_tyottomat_osuus=np.zeros((n,self.n_time))
        emp_htv=np.zeros((n,self.n_time))

        self.load_sim(results+'_'+str(100+startn))
        base_empstate=self.empstate/self.n_pop
        emps[0,:,:]=base_empstate
        htv_base,tyoll_base,haj_base,tyollaste_base,tyolliset_base=self.comp_tyollisyys_stats(base_empstate,scale_time=True)
        reward=self.get_reward()
        agg_htv[0]=htv_base
        agg_tyoll[0]=tyoll_base
        agg_rew[0]=reward
        best_rew=reward
        best_emp=0
        t_aste[0]=tyollaste_base
        
        tyolliset_ika,tyottomat,htv_ika,tyolliset_osuus,tyottomat_osuus=self.comp_employed_number(base_empstate)
        
        emp_tyolliset[0,:]=tyolliset_ika[:,0]
        emp_tyottomat[0,:]=tyottomat[:,0]
        emp_tyolliset_osuus[0,:]=tyolliset_osuus[:,0]
        emp_tyottomat_osuus[0,:]=tyottomat_osuus[:,0]
        emp_htv[0,:]=htv_ika[:,0]

        if plot:
            fig,ax=plt.subplots()
            ax.set_xlabel('Ikä [v]')
            ax.set_ylabel('Työllisyysaste [%]')
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,100*tyolliset_base,alpha=0.9,lw=2.0)

        for i in range(startn+1,n): 
            self.load_sim(results+'_'+str(100+i))
            empstate=self.empstate/self.n_pop
            emps[i,:,:]=empstate
            reward=self.get_reward()
            if reward>best_rew:
                best_rew=reward
                best_emp=i

            htv,tyollvaikutus,haj,tyollisyysaste,tyolliset=self.comp_tyollisyys_stats(empstate,scale_time=True)
            if plot:
                ax.plot(x,100*tyolliset,alpha=0.5,lw=0.5)
    
            agg_htv[i]=htv
            agg_tyoll[i]=tyollvaikutus
            agg_rew[i]=reward
            t_aste[i]=tyollisyysaste

            tyolliset_ika,tyottomat,htv_ika,tyolliset_osuus,tyottomat_osuus=self.comp_employed_number(empstate)
            
            emp_tyolliset[i,:]=tyolliset_ika[:,0]
            emp_tyottomat[i,:]=tyottomat[:,0]
            emp_tyolliset_osuus[i,:]=tyolliset_osuus[:,0]
            emp_tyottomat_osuus[i,:]=tyottomat_osuus[:,0]
            emp_htv[i,:]=htv_ika[:,0]

        self.save_simstats(save,agg_htv,agg_tyoll,agg_rew,\
                            emp_tyolliset,emp_tyolliset_osuus,\
                            emp_tyottomat,emp_tyottomat_osuus,\
                            emp_htv,emps,\
                            best_rew,best_emp)
                    
        # save the best
        self.load_sim(results+'_'+str(100+best_emp))
        self.save_sim(results+'_best')
                    
        print('best_emp',best_emp)
        
    def fit_norm(self,diff):
        diff_stdval=np.std(diff)
        diff_meanval=np.mean(diff)
        diff_minval=np.min(diff)
        diff_maxval=np.max(diff)
        sz=(diff_maxval-diff_minval)/10
        x=np.linspace(diff_minval,diff_maxval,1000)
        y=norm.pdf(x,diff_meanval,diff_stdval)*diff.shape[0]*sz
    
        return x,y
        
    def count_putki_dist(self,emps):
        putki=[]
    
        for k in range(emps.shape[0]):
            putki.append(self.count_putki(emps[k,:,:]))
            
        putkessa=np.median(np.asarray(putki))
        return putkessa
        
    def plot_simstats(self,filename):
        agg_htv,agg_tyoll,agg_rew,emp_tyolliset,emp_tyolliset_osuus,\
            emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,best_emp,emps=self.load_simstats(filename)

        print('lisäpäivillä on {:.0f} henkilöä'.format(self.count_putki_dist(emps)))

        mean_htv=np.mean(agg_htv)
        median_htv=np.median(agg_htv)
        mean_tyoll=np.mean(agg_tyoll)
        median_tyoll=np.median(agg_tyoll)
        std_htv=np.std(agg_htv)
        std_tyoll=np.std(agg_tyoll)
        diff_htv=agg_htv-mean_htv
        diff_tyoll=agg_tyoll-median_tyoll
        
        x,y=self.fit_norm(diff_htv)
        
        m_mean=np.mean(emp_tyolliset_osuus,axis=0)
        m_median=np.median(emp_tyolliset_osuus,axis=0)
        s_emp=np.std(emp_tyolliset_osuus,axis=0)
        m_best=emp_tyolliset_osuus[best_emp,:]

        if self.minimal:
            print('Vaikutus työllisyyteen keskiarvo {:.0f} htv mediaan {:.0f} htv std {:.0f} htv'.format(mean_htv,median_htv,std_htv))
        else:
            print('Vaikutus työllisyyteen keskiarvo {:.0f} htv, mediaani {:.0f} htv std {:.0f} htv\n   keskiarvo {:.0f} työllistä, mediaani {:.0f} työllistä, std {:.0f} työllistä'.format(mean_htv,median_htv,std_htv,mean_tyoll,median_tyoll,std_tyoll))

        fig,ax=plt.subplots()
        ax.set_xlabel('Poikkeama työllisyydessä [htv]')
        ax.set_ylabel('Lukumäärä')
        ax.hist(diff_htv)
        ax.plot(x,y)
        plt.show()

        if not self.minimal:
            x,y=self.fit_norm(diff_tyoll)
            fig,ax=plt.subplots()
            ax.set_xlabel('Poikkeama työllisyydessä [henkilöä]')
            ax.set_ylabel('Lukumäärä')
            ax.hist(diff_tyoll)
            ax.plot(x,y)
            plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Palkkio')
        ax.set_ylabel('Lukumäärä')
        ax.hist(agg_rew)
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Työllisyysaste [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*m_mean,label='keskiarvo')
        ax.plot(x,100*m_median,label='mediaani')
        #ax.plot(x,100*(m_emp+s_emp),label='ka+std')
        #ax.plot(x,100*(m_emp-s_emp),label='ka-std')
        ax.plot(x,100*m_best,label='paras')
        emp_statsratio=100*self.emp_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Hajonta työllisyysasteessa [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*s_emp)
        plt.show()

    def get_simstats(self,filename1,plot=False):
        agg_htv,agg_tyoll,agg_rew,emp_tyolliset,emp_tyolliset_osuus,\
            emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,best_emp,emps=self.load_simstats(filename1)

        mean_htv=np.mean(agg_htv)
        median_htv=np.median(agg_htv)
        mean_tyoll=np.mean(agg_tyoll)
        median_tyoll=np.median(agg_tyoll)
        std_htv=np.std(agg_htv)
        diff_htv=agg_htv-mean_htv
        diff_tyoll=agg_tyoll-mean_tyoll
        
        m_mean=np.mean(emp_tyolliset_osuus,axis=0)
        m_median=np.median(emp_tyolliset_osuus,axis=0)
        mn_median=np.median(emp_tyolliset,axis=0)
        s_emp=np.std(emp_tyolliset_osuus,axis=0)
        m_best=emp_tyolliset_osuus[best_emp,:]

        h_mean=np.mean(emp_htv,axis=0)
        h_median=np.median(emp_htv,axis=0)
        hs_emp=np.std(emp_htv,axis=0)
        h_best=emp_htv[best_emp,:]


        if self.minimal:
            u_tmtuki=0*np.median(emps[:,:,0],axis=0)
            u_ansiosid=np.median(emps[:,:,0],axis=0)
        else:
            u_tmtuki=np.median(emps[:,:,13],axis=0)
            u_ansiosid=np.median(emps[:,:,0]+emps[:,:,4],axis=0)
    
        if plot:
            fig,ax=plt.subplots()
            ax.set_xlabel('Poikkeama työllisyydessä [htv]')
            ax.set_ylabel('Lukumäärä')
            ax.hist(diff_htv)
            plt.show()

            if not self.minimal:
                fig,ax=plt.subplots()
                ax.set_xlabel('Poikkeama työllisyydessä [henkilöä]')
                ax.set_ylabel('Lukumäärä')
                ax.hist(diff_tyoll)
                plt.show()

            fig,ax=plt.subplots()
            ax.set_xlabel('Palkkio')
            ax.set_ylabel('Lukumäärä')
            ax.hist(agg_rew)
            plt.show()    
    
        return m_best,m_mean,m_median,s_emp,median_htv,u_tmtuki,u_ansiosid,h_median,mn_median #,emp_tyolliset

    def compare_simstats(self,filename1,filename2,label1='perus',label2='vaihtoehto'):
        m_best1,m_emp1,m_median1,s_emp1,median_htv1,u_tmtuki1,u_ansiosid1,h_median1,mn_median1=self.get_simstats(filename1)
        m_best2,m_emp2,m_median2,s_emp2,median_htv2,u_tmtuki2,u_ansiosid2,h_median2,mn_median2=self.get_simstats(filename2)
        
        print('Vaikutus työllisyysasteeseen {} htv ({} vs {})'.format(median_htv2-median_htv1,median_htv2,median_htv1))

        #if self.minimal:
        #    print('Vaikutus työllisyysasteen keskiarvo {} htv mediaan {} htv'.format(d['mean_htv'],d['median_htv']))
        #else:
        #    print('Vaikutus työllisyysasteen keskiarvo {} htv, mediaani {} htv\n                        keskiarvo {} työllistä, mediaani {} työllistä'.format(d['mean_htv'],d['median_htv'],d['mean_tyoll'],d['median_tyoll']))

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Työllisyys [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*m_median1[1:],label=label1)
        ax.plot(x[1:],100*m_median2[1:],label=label2)
        #emp_statsratio=100*self.emp_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Työllisyysero [hlö/htv]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],mn_median2[1:]-mn_median1[1:],label=label2+' miinus '+label1)
        ax.plot(x[1:],h_median2[1:]-h_median1[1:],label=label2+' miinus '+label1+' htv')
        #emp_statsratio=100*self.emp_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus väestöstä [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*u_tmtuki1[1:],label='tm-tuki, '+label1)
        ax.plot(x[1:],100*u_tmtuki2[1:],label='tm-tuki, '+label2)
        ax.plot(x[1:],100*u_ansiosid1[1:],label='ansiosidonnainen, '+label1)
        ax.plot(x[1:],100*u_ansiosid2[1:],label='ansiosidonnainen, '+label2)
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('työllisyysero [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*(m_median2[1:]-m_median1[1:]),label=label1)
        #emp_statsratio=100*self.emp_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        #ax.legend()
        plt.show()

        demog1,demog2=self.get_demog()
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('cumsum työllisyys [lkm]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        cs=self.timestep*np.cumsum(h_median2[1:]-h_median1[1:])
        c2=self.timestep*np.cumsum(h_median1[1:])
        c1=self.timestep*np.cumsum(h_median2[1:])
        ax.plot(x[1:],cs,label=label1)
        #emp_statsratio=100*self.emp_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        #ax.legend()
        plt.show()

        for age in set([63,63.25,63.5,50]):
            mx=self.map_age(age)-1
            print('Kumulatiivinen työllisyysvaikutus {} vuotiaana {:.1f} htv ({} vs {})'.format(age,cs[mx],c1[mx],c2[mx]))

    def save_simstats(self,filename,agg_htv,agg_tyoll,agg_rew,emp_tyolliset,emp_tyolliset_osuus,\
                        emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,best_emp):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset('agg_htv', data=agg_htv, dtype='float64')
        dset = f.create_dataset('agg_tyoll', data=agg_tyoll, dtype='float64')
        dset = f.create_dataset('agg_rew', data=agg_rew, dtype='float64')
        dset = f.create_dataset('emp_tyolliset', data=emp_tyolliset, dtype='float64')
        dset = f.create_dataset('emp_tyolliset_osuus', data=emp_tyolliset_osuus, dtype='float64')
        dset = f.create_dataset('emp_tyottomat', data=emp_tyottomat, dtype='float64')
        dset = f.create_dataset('emp_tyottomat_osuus', data=emp_tyottomat_osuus, dtype='float64')
        dset = f.create_dataset('emp_htv', data=emp_htv, dtype='float64')
        dset = f.create_dataset('emps', data=emps, dtype='float64')
        dset = f.create_dataset('best_rew', data=best_rew, dtype='float64')
        dset = f.create_dataset('best_emp', data=best_emp, dtype='float64')
        
        f.close()

    def load_simstats(self,filename):
        f = h5py.File(filename, 'r')
        #n_pop = f.get('n_pop').value
        agg_htv = f.get('agg_htv').value
        agg_tyoll = f.get('agg_tyoll').value
        agg_rew = f.get('agg_rew').value
        emps = f.get('emps').value
        best_rew = f.get('best_rew').value
        best_emp = int(f.get('best_emp').value)
        emp_tyolliset = f.get('emp_tyolliset').value
        emp_tyolliset_osuus = f.get('emp_tyolliset_osuus').value
        emp_tyottomat = f.get('emp_tyottomat').value
        emp_tyottomat_osuus = f.get('emp_tyottomat_osuus').value
        emp_htv = f.get('emp_htv').value
        
        f.close()

        return agg_htv,agg_tyoll,agg_rew,emp_tyolliset,emp_tyolliset_osuus,\
               emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,best_emp,emps
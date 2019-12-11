'''

    episodestats.py

    implements statistic that are used in producing employment statistics for the
    lifecycle model

'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.inv_timestep=int(np.round(1/self.timestep)) # pitäisi olla kokonaisluku
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
        self.siirtyneet=np.zeros((self.n_time,n_emps))
        self.pysyneet=np.zeros((self.n_time,n_emps))
        self.salaries=np.zeros((self.n_time,self.n_pop))
        self.aveV=np.zeros((self.n_time,self.n_pop))
        self.time_in_state=np.zeros((self.n_time,n_emps))
        self.stat_tyoura=np.zeros((self.n_time,n_emps))
        self.stat_toe=np.zeros((self.n_time,n_emps))
        self.stat_pension=np.zeros((self.n_time,n_emps))
        self.stat_paidpension=np.zeros((self.n_time,n_emps))
        self.stat_unemp_len=np.zeros((self.n_time,self.n_pop))

    def add(self,n,act,r,state,newstate,debug=False,plot=False,aveV=None): #,dyn=False):
        #if debug:
        #    print((int(state[0]),int(state[1]),state[2],state[3],state[4]),':',act,(int(newstate[0]),int(newstate[1]),newstate[2],newstate[3],newstate[4]))
    
        if self.minimal:
            emp,_,_,a,_=self.env.state_decode(state) # current employment state
            newemp,_,newsal,a2,tis=self.env.state_decode(newstate)
            g=0
        else:
            emp,_,_,_,a,_,_,_,_,_=self.env.state_decode(state) # current employment state
            newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,ura=self.env.state_decode(newstate)
    
        #if emp==2 and a<self.min_retirementage:
        #    emp=12
        #if newemp==2 and a2<self.min_retirementage:
        #    emp=12
    
        t=int(np.round((a-self.min_age)*self.inv_timestep))
        #print(t,r,a,emp,newemp)
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            if a2>self.min_retirementage and newemp==3:
                newemp=2
            self.empstate[t,newemp]+=1
            self.alive[t]+=1
            self.rewstate[t,newemp]+=r
            self.actions[t,n]=act
            self.salaries[t,n]=newsal
            self.salaries_emp[t,newemp]+=newsal
            self.salaries_emp[t,newemp]+=newsal
            self.time_in_state[t,newemp]+=tis
            if not self.minimal:
                self.gempstate[t,newemp,g]+=1
                self.galive[t,g]+=1
                self.stat_tyoura[t,newemp]+=ura
                self.stat_toe[t,newemp]+=toe
                self.stat_pension[t,newemp]+=newpen
                self.stat_paidpension[t,newemp]+=paidpens
                self.stat_unemp_len[t,n]=tis
    
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

    def comp_empratios(self,emp,alive):
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

        if not self.minimal:
            tyollisyysaste=100*(employed+osatyo+veosatyo+vetyo)/alive[:,0]
            osatyoaste=100*(osatyo+veosatyo)/(employed+osatyo+veosatyo+vetyo)
            tyottomyysaste=100*(unemployed+piped)/(unemployed+employed+piped+osatyo+veosatyo+vetyo)
            ka_tyottomyysaste=100*np.sum(unemployed+piped)/np.sum(unemployed+employed+piped+osatyo+veosatyo+vetyo)
        else:
            tyollisyysaste=100*(employed)
            osatyoaste=np.zeros(employed.shape)
            tyottomyysaste=100*(unemployed)/(unemployed+employed)
            ka_tyottomyysaste=100*np.sum(unemployed)/np.sum(unemployed+employed)

        return tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste

    def plot_emp(self):

        tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(self.empstate,self.alive)

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,tyollisyysaste,label='työllisyysaste')
        ax.plot(x,tyottomyysaste,label='työttömyys')
        emp_statsratio=100*self.emp_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Työttömyysaste (ka '+str(ka_tyottomyysaste)+')')
        ax.plot(x,tyottomyysaste)
        plt.show()

        if not self.minimal:
            fig,ax=plt.subplots()
            ax.plot(x,osatyoaste,label='osatyössäolevie kaikista töissäolevista')
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
            
        emp_statsratio=100*self.emp_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Osuus tilassa [%]')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

            #fig,ax=plt.subplots()
            #ax.set_xlabel('Ikä [v]')
            #ax.set_ylabel('Työttömyysaste (ka '+str(ka_tyottomyysaste)+')')
            #ax.plot(x,tyottomyysaste)
            #plt.show()

            #if not self.minimal:
            #    fig,ax=plt.subplots()
            #    ax.plot(x,osatyoaste,label='osatyössäolevie kaikista töissäolevista')
            #    ax.legend()
            #    plt.show()
    
            #empstate_ratio=100*gempstate/self.alive
            #self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',stack=True)

            #if not self.minimal:
            #    self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',ylimit=20,stack=False)

            #if not self.minimal:
            #    self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',parent=True,stack=False)
            #    self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',unemp=True,stack=False)

            #self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',start_from=60,stack=True)

    def plot_pensions(self):
        if not self.minimal:
            self.plot_ratiostates(self.stat_pension,ylabel='Tuleva eläke [e/v]',stack=False)

    def plot_career(self):    
        if not self.minimal:
            self.plot_ratiostates(self.stat_tyoura,ylabel='Työuran pituus [v]',stack=False)

    def plot_ratiostates(self,statistic,ylabel='',ylimit=None, show_legend=True, parent=False,\
                         unemp=False,start_from=None,stack=False):
        self.plot_states(statistic/self.empstate,ylabel=ylabel,ylimit=ylimit,\
                    show_legend=show_legend,parent=parent,unemp=unemp,start_from=start_from,stack=stack)

    def plot_states(self,statistic,ylabel='',ylimit=None,show_legend=True,parent=False,unemp=False,
                    start_from=None,stack=True,save=False,filename='fig.png',yminlim=None,ymaxlim=None):
        if start_from is None:
            x=np.linspace(self.min_age,self.max_age,self.n_time)
        else:
            x_n = self.max_age-60+1
            x_t = int(np.round(x_n*self.inv_timestep+1))
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
                    ax.stackplot(x,ura_unemp,ura_pipe,ura_student,ura_outsider,
                        labels=('tyött','putki','opiskelija','ulkona'), colors=pal)
                else:
                    ax.stackplot(x,ura_unemp,labels=('tyött'), colors=pal)
            else:
                if not self.minimal:
                    ax.stackplot(x,ura_emp,ura_osatyo,ura_vetyo,ura_veosatyo,ura_unemp,ura_pipe,ura_disab,ura_mother,ura_dad,ura_kht,ura_ret,ura_student,ura_outsider,
                        labels=('työssä','osatyö','ve+työ','ve+osatyö','työtön','työttömyysputki','tk','äitiysvapaa','isyysvapaa','khtuki','vanhuuseläke','opiskelija','ulkona'), 
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
                    ax.plot(x,ura_mother,label='äitiysvapaa')
                    ax.plot(x,ura_dad,label='isyysvapaa')
                    ax.plot(x,ura_kht,label='khtuki')
                    ax.plot(x,ura_vetyo,label='ve+työ')
                    ax.plot(x,ura_veosatyo,label='ve+osatyö')
                    ax.plot(x,ura_osatyo,label='osatyö')
                    ax.plot(x,ura_student,label='student')
                    ax.plot(x,ura_outsider,label='outsider')
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
            self.plot_ratiostates(self.stat_toe,'Työssäolo-ehdon pituus 28 kk aikana [v]',stack=False)

    def plot_sal(self):
        self.plot_ratiostates(self.salaries_emp,'Keskipalkka [e/v]',stack=False)

    def plot_moved(self):
        siirtyneet_ratio=self.siirtyneet/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        pysyneet_ratio=self.pysyneet/self.alive*100
        self.plot_states(pysyneet_ratio,ylabel='Pysyneet tilassa',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(pysyneet_ratio,1))))

    def plot_ave_stay(self):
        self.plot_ratiostates(self.time_in_state,ylabel='Ka kesto tilassa',stack=False)

    def plot_reward(self):
        self.plot_ratiostates(self.rewstate,ylabel='Keskireward tilassa',stack=False)

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        total_reward=np.sum(self.rewstate,axis=1)
        fig,ax=plt.subplots()
        ax.plot(x,total_reward)
        ax.set_xlabel('Aika')
        ax.set_ylabel('Koko reward tilassa')
        ax.legend()
        plt.show()  

        rr=np.sum(total_reward)/self.n_pop
        print('Yhteensä reward {r}'.format(r=rr))

    def plot_stats(self):
        self.plot_emp()
        if not self.minimal:
            self.plot_group_emp()
        self.plot_sal()
        self.plot_moved()
        self.plot_ave_stay()
        self.plot_reward()
        self.plot_pensions()
        self.plot_career()
        self.plot_toe()

    def plot_img(self,img,xlabel="Eläke",ylabel="Palkka",title="Employed"):
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        heatmap = plt.pcolor(img) 
        plt.colorbar(heatmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.show()
        
    def emp_stats(self):
        emp_ratio=np.zeros(self.n_time)
        #emp_ratio[15:20]=0.245
        #emp_ratio[20:25]=0.595
        emp_ratio[self.map_age(25):self.map_age(30)]=0.747
        emp_ratio[self.map_age(30):self.map_age(35)]=0.789
        emp_ratio[self.map_age(35):self.map_age(40)]=0.836
        emp_ratio[self.map_age(40):self.map_age(45)]=0.867
        emp_ratio[self.map_age(45):self.map_age(50)]=0.857
        emp_ratio[self.map_age(50):self.map_age(55)]=0.853
        emp_ratio[self.map_age(55):self.map_age(60)]=0.791
        emp_ratio[self.map_age(60):self.map_age(65)]=0.517
        emp_ratio[self.map_age(65):self.map_age(70)]=0.141
        #emp_ratio[70:74]=0.073

        return emp_ratio

    def save_sim(self,filename):
        f = h5py.File(filename, 'w')
        ftype='float64'
        dset = f.create_dataset('n_pop', data=self.n_pop, dtype=ftype)
        dset = f.create_dataset('empstate', data=self.empstate, dtype=ftype)
        dset = f.create_dataset('deceiced', data=self.deceiced, dtype=ftype)
        dset = f.create_dataset('rewstate', data=self.rewstate, dtype=ftype)
        dset = f.create_dataset('salaries_emp', data=self.salaries_emp, dtype=ftype)
        dset = f.create_dataset('actions', data=self.actions, dtype=ftype)
        dset = f.create_dataset('alive', data=self.alive, dtype=ftype)
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
            self.n_pop=f.get('n_pop').value
        else:
            self.n_pop=1_000
        if n_pop is not None:
            self.n_pop=n_pop
            
        self.empstate=f.get('empstate').value
        self.deceiced=f.get('deceiced').value
        self.rewstate=f.get('rewstate').value
        self.salaries_emp=f.get('salaries_emp').value
        self.actions=f.get('actions').value
        self.alive=f.get('alive').value
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
        f.close()

    def render(self,load=None):
        if load is not None:
            self.load_sim(load)

        #self.plot_stats(5)
        self.plot_stats()
        #self.plot_reward()   

    def compare_with(self,cc2):
        diff_emp=self.empstate/self.n_pop-cc2.empstate/cc2.n_pop
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        #x=range(self.age_min,self.age_min+self.n_time)

        s=30
        e=63

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1=self.comp_employed(self.empstate)
        tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2=self.comp_employed(cc2.empstate)
        htv1,tyoll1,haj1,tyollaste1,tyolliset1=self.comp_tyollisyys_stats(self.empstate/self.n_pop,scale_time=True,start=s,end=e)
        htv2,tyoll2,haj2,tyollaste2,tyolliset2=self.comp_tyollisyys_stats(cc2.empstate/cc2.n_pop,scale_time=True,start=s,end=e)
        #htv,tyollvaikutus,haj,tyollaste,tyollosuus=self.comp_tyollisyys_stats(diff_emp,scale_time=True)
        
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Työllisyysaste [%]')
        ax.plot(x,100*tyolliset1,label='vaihtoehto')
        ax.plot(x,100*tyolliset2,label='perus')
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

        print('Työllisyysvaikutus {}-{}-vuotiaisiin noin {t} htv ja {h} työllistä'.format(s,e,t=htv1-htv2,
            h=tyoll1-tyoll2))
        print('Työllisyysastevaikutus {}-{}-vuotiailla noin {} prosenttia'.format(s,e,(tyollaste1-tyollaste2)*100))
        
        # epävarmuus
        delta=1.96*1.0/np.sqrt(self.n_pop)
        print('Epävarmuus työllisyysasteissa {}, hajonta {}'.format(delta,haj1))
        
    def comp_employed(self,emp):
        if self.minimal:
            tyoll_osuus=emp[:,1]/np.sum(emp,1)
            tyot_osuus=emp[:,0]/np.sum(emp,1)
            htv=emp[:,1]/np.sum(emp,1)
            kokotyo_osuus=tyoll_osuus
            osatyo_osuus=0
        else:
            tyoll_osuus=(emp[:,1]+emp[:,8]+emp[:,9]+emp[:,10])/np.sum(emp,1)
            tyot_osuus=(emp[:,0]+emp[:,4])/np.sum(emp,1)
            htv=(emp[:,1]+emp[:,8]+0.5*emp[:,9]+0.5*emp[:,10])/np.sum(emp,1)
            kokotyo_osuus=(emp[:,1]+emp[:,8])/np.sum(emp,1)
            osatyo_osuus=(emp[:,9]+emp[:,10])/np.sum(emp,1)
            
        return tyoll_osuus,htv,tyot_osuus,kokotyo_osuus,osatyo_osuus
    
        
    def comp_tyollisyys_stats(self,emp,scale_time=True,start=30,end=63):
        demog=np.array([61663,63354,65939,68253,68543,71222,70675,71691,70202,70535, # 20-29 y
                        67315,68282,70431,72402,73839,73065,70040,69501,68857,69035, # 30-39 y
                        69661,69965,68429,65261,59498,61433,63308,65305,66580,71263, # 40-49 y
                        72886,73253,73454,74757,75406,74448,73940,73343,72808,70259, # 50-59 y
                        73065,74666,73766,73522,72213,74283,71273,73404,75153,75888  # 60-69 y
                        ])
              
        demog2=np.zeros(self.n_time)
        k2=0
        for k in np.arange(self.min_age,self.max_age,self.timestep):
            ind=int(np.floor(k))-self.min_age
            demog2[k2]=demog[ind]
            k2+=1
              
        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)
        
        tyollosuus,htvosuus,_,_,_=self.comp_employed(emp)
        
        htv=np.round(scale*np.sum(demog2[min_cage:max_cage]*htvosuus[min_cage:max_cage]))
        tyollvaikutus=np.round(scale*np.sum(demog2[min_cage:max_cage]*tyollosuus[min_cage:max_cage]))
        haj=np.mean(np.std(tyollosuus[min_cage:max_cage]))
            
        tyollaste=tyollvaikutus/sum(demog[min_cage:max_cage])
            
        return htv,tyollvaikutus,haj,tyollaste,tyollosuus

    def get_reward(self):
        total_reward=np.sum(self.rewstate,axis=1)
        rr=np.sum(total_reward)/self.n_pop
        return rr

    
class SimStats(EpisodeStats):
    def run_simstats(self,results,save,n,plot=True):
        '''
        Multiple stats, not used
        '''
        #n=self.load_hdf(results+'_simut','n')
        e_rate=np.zeros((n,self.n_time))
        diff_rate=np.zeros((n,self.n_time))
        agg_htv=np.zeros(n)
        agg_tyoll=np.zeros(n)
        agg_rew=np.zeros(n)
        diff_htv=np.zeros(n)
        diff_tyoll=np.zeros(n)
        t_aste=np.zeros(self.n_time)
        mean_hvt=np.zeros(self.n_time)
        std_htv=np.zeros(self.n_time)
        mean_emp=np.zeros((self.n_time,self.n_employment))
        std_emp=np.zeros((self.n_time,self.n_employment))
        emps=np.zeros((n,self.n_time,self.n_employment))

        self.load_sim(results+'_100')
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

        print(tyollaste_base)

        if plot:
            fig,ax=plt.subplots()
            ax.set_xlabel('työllisyysaste')
            ax.set_ylabel('lkm')
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            #print(x.shape,tyoll_base.shape,self.empstate.shape)
            print(self.min_age,self.max_age,x.shape)
            ax.plot(x,100*tyolliset_base)

        for i in range(1,n): 
            self.load_sim(results+'_'+str(100+i))
            empstate=self.empstate/self.n_pop
            emps[i,:,:]=empstate
            reward=self.get_reward()
            if reward>best_rew:
                best_rew=reward
                best_emp=i

            htv,tyollvaikutus,haj,tyollisyysaste,tyolliset=self.comp_tyollisyys_stats(empstate,scale_time=True)
            if plot:
                ax.plot(x,100*tyolliset)
    
            agg_htv[i]=htv
            agg_tyoll[i]=tyollvaikutus
            agg_rew[i]=reward
            diff_htv[i]=htv-htv_base
            diff_tyoll[i]=tyollvaikutus-tyoll_base
            t_aste[i]=tyollisyysaste
    
        if plot:
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            emp_statsratio=100*self.emp_stats()
            ax.plot(x,emp_statsratio,label='havainto')
            plt.show()

        mean_emp=np.mean(emps,axis=0)
        std_emp=np.std(emps,axis=0)
        median_emp=np.median(emps,axis=0)

        #print(agg_htv,agg_tyoll)
        self.save_simstats(save,diff_htv,diff_tyoll,agg_htv,agg_tyoll,agg_rew,\
                            mean_emp,std_emp,median_emp,emps,best_rew,best_emp)
                    
        # save the best
        self.load_sim(results+'_'+str(100+best_emp))
        self.save_sim(results+'_best')
                    
        print('best_emp',best_emp)

    def plot_simstats(self,filename):
        #print('load',filename)
        diff_htv,diff_tyoll,agg_htv,agg_tyoll,agg_rew,mean_emp,std_emp,median_emp,emps,\
            best_rew,best_emp,n_pop=self.load_simstats(filename)

        mean_htv=np.mean(agg_htv)
        median_htv=np.median(agg_htv)
        mean_tyoll=np.mean(agg_tyoll)
        median_tyoll=np.median(agg_tyoll)
        std_htv=np.std(agg_htv)
        diff_htv=agg_htv-mean_htv
        diff_tyoll=agg_tyoll-mean_tyoll

        if self.minimal:
            m_emp=mean_emp[:,1]
            m_median=median_emp[:,1]
            s_emp=std_emp[:,1]
            m_best=emps[best_emp,:,1]
        else:
            m_emp=mean_emp[:,1]+mean_emp[:,10]+mean_emp[:,8]+mean_emp[:,9]
            m_median=median_emp[:,1]+median_emp[:,10]+median_emp[:,8]+median_emp[:,9]
            s_emp=std_emp[:,1]
            m_best=emps[best_emp,:,1]+emps[best_emp,:,10]+emps[best_emp,:,8]+emps[best_emp,:,9]

        if self.minimal:
            print('Vaikutus työllisyyteen keskiarvo {} htv mediaan {} htv'.format(mean_htv,median_htv))
        else:
            print('Vaikutus työllisyyteen keskiarvo {} htv, mediaani {} htv\n                        keskiarvo {} työllistä, mediaani {} työllistä'.format(mean_htv,median_htv,mean_tyoll,median_tyoll))

        fig,ax=plt.subplots()
        ax.set_xlabel('poikkeama työllisyydessä [htv]')
        ax.set_ylabel('lkm')
        ax.hist(diff_htv)
        plt.show()

        if not self.minimal:
            fig,ax=plt.subplots()
            ax.set_xlabel('poikkeama työllisyydessä [henkilöä]')
            ax.set_ylabel('lkm')
            ax.hist(diff_tyoll)
            plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Palkkio')
        ax.set_ylabel('lkm')
        ax.hist(agg_rew)
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä [v]')
        ax.set_ylabel('Työllisyysaste [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*m_emp,label='keskiarvo')
        ax.plot(x,100*m_median,label='mediaani')
        #ax.plot(x,100*(m_emp+s_emp),label='ka+std')
        #ax.plot(x,100*(m_emp-s_emp),label='ka-std')
        ax.plot(x,100*m_best,label='paras')
        emp_statsratio=100*self.emp_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.legend()
        plt.show()

    def get_simstats(filename):
        diff1_htv,diff1_tyoll,agg1_htv,agg1_tyoll,agg1_rew,mean1_emp,std1_emp,median1_emp,emps1,\
            best1_rew,best1_emp,n_pop=self.load_simstats(filename1)

        mean_htv=np.mean(agg_htv)
        median_htv=np.median(agg_htv)
        mean_tyoll=np.mean(agg_tyoll)
        median_tyoll=np.median(agg_tyoll)
        std_htv=np.std(agg_htv)
        diff_htv=agg_htv-mean_htv
        diff_tyoll=agg_tyoll-mean_tyoll

        if self.minimal:
            m_emp=mean_emp[:,1]
            m_median=median_emp[:,1]
            s_emp=std_emp[:,1]
            m_best=emps[best_emp,:,1]
        else:
            m_emp=mean_emp[:,1]+mean_emp[:,10]+mean_emp[:,8]+mean_emp[:,9]
            m_median=median_emp[:,1]+median_emp[:,10]+median_emp[:,8]+median_emp[:,9]
            s_emp=std_emp[:,1]
            m_best=emps[best_emp,:,1]+emps[best_emp,:,10]+emps[best_emp,:,8]+emps[best_emp,:,9]
    
        if plot:
            fig,ax=plt.subplots()
            ax.set_xlabel('poikkeama työllisyydessä [htv]')
            ax.set_ylabel('lkm')
            ax.hist(diff_htv)
            plt.show()

            if not self.minimal:
                fig,ax=plt.subplots()
                ax.set_xlabel('poikkeama työllisyydessä [henkilöä]')
                ax.set_ylabel('lkm')
                ax.hist(diff_tyoll)
                plt.show()

            fig,ax=plt.subplots()
            ax.set_xlabel('Palkkio')
            ax.set_ylabel('lkm')
            ax.hist(agg_rew)
            plt.show()    
    
        return m_best,m_emp,m_meadian,s_emp

    def compare_simstats(self,filename1,filename2):
        #print('load',filename)

        if self.minimal:
            print('Vaikutus työllisyysasteen keskiarvo {} htv mediaan {} htv'.format(mean_htv,median_htv))
        else:
            print('Vaikutus työllisyysasteen keskiarvo {} htv, mediaani {} htv\n                        keskiarvo {} työllistä, mediaani {} työllistä'.format(mean_htv,median_htv,mean_tyoll,median_tyoll))

        fig,ax=plt.subplots()
        ax.set_xlabel('keskimääräinen työllisyys')
        ax.set_ylabel('työllisyysaste')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*m_emp1,label='keskiarvo1')
        ax.plot(x,100*m_median1,label='mediaani1')
        ax.plot(x,100*m_best1,label='paras1')
        ax.plot(x,100*m_emp2,label='keskiarvo2')
        ax.plot(x,100*m_median2,label='mediaani2')
        ax.plot(x,100*m_best2,label='paras2')
        emp_statsratio=100*self.emp_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.legend()
        plt.show()


    def save_simstats(self,filename,diff_htv,diff_tyoll,agg_htv,agg_tyoll,agg_rew,mean_emp,\
                      std_emp,median_emp,emps,best_rew,best_emp):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset('n_pop', data=n_pop, dtype='float64')
        dset = f.create_dataset('agg_htv', data=agg_htv, dtype='float64')
        dset = f.create_dataset('agg_tyoll', data=agg_tyoll, dtype='float64')
        dset = f.create_dataset('diff_htv', data=diff_htv, dtype='float64')
        dset = f.create_dataset('diff_tyoll', data=diff_tyoll, dtype='float64')
        dset = f.create_dataset('agg_rew', data=agg_rew, dtype='float64')
        dset = f.create_dataset('mean_emp', data=mean_emp, dtype='float64')
        dset = f.create_dataset('std_emp', data=std_emp, dtype='float64')
        dset = f.create_dataset('median_emp', data=median_emp, dtype='float64')
        dset = f.create_dataset('emps', data=emps, dtype='float64')
        dset = f.create_dataset('best_rew', data=best_rew, dtype='float64')
        dset = f.create_dataset('best_emp', data=best_emp, dtype='float64')
        f.close()

    def load_simstats(self,filename):
        f = h5py.File(filename, 'r')
        n_pop = f.get('n_pop').value
        agg_htv = f.get('agg_htv').value
        agg_tyoll = f.get('agg_tyoll').value
        diff_htv = f.get('diff_htv').value
        diff_tyoll = f.get('diff_tyoll').value
        agg_rew = f.get('agg_rew').value
        mean_emp = f.get('mean_emp').value
        std_emp = f.get('std_emp').value
        emps = f.get('emps').value
        median_emp = f.get('median_emp').value
        best_rew = f.get('best_rew').value
        best_emp = int(f.get('best_emp').value)
        f.close()
        return diff_htv,diff_tyoll,agg_htv,agg_tyoll,agg_rew,mean_emp,std_emp,median_emp,\
               emps,best_rew,best_emp,n_pop

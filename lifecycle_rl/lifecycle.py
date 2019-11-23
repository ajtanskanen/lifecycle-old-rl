import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
from fin_benefits import Benefits
import gym_unemployment
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from stable_baselines.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines import A2C, ACER, DQN, ACKTR #, TRPO
from stable_baselines.common import set_global_seeds
#from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
from .openai_monitor import Monitor
from .vec_monitor import VecMonitor
import os


class Lifecycle():

    def __init__(self,env=None,minimal=False,timestep=0.25,ansiopvraha_kesto300=None,\
                    ansiopvraha_kesto400=None,karenssi_kesto=None,\
                    ansiopvraha_toe=None,perustulo=None,mortality=None,\
                    randomness=None,deterministic=None):

        '''
        Alusta muuttujat
        '''
        self.timestep=timestep # 0.25 = 3kk askel
        self.inv_timestep=int(np.round(1/self.timestep)) # pitäisi olla kokonaisluku
        self.min_age = 20
        self.max_age = 70
        self.min_retirementage=65
        self.max_retirementage=70
        self.n_pop = 1000
                
        # apumuuttujia
        self.n_age = self.max_age-self.min_age+1
        self.n_time = int(np.round(self.n_age*self.inv_timestep+1))
        self.gamma = 0.92**timestep # skaalataan vuositasolle!
        
        self.karenssi_kesto=0.25
        
        if karenssi_kesto is not None:
            self.karenssi_kesto=karenssi_kesto

        if mortality is not None:
            self.mortality=mortality
        else:
            self.mortality=False
            
        if mortality is not None:
            self.deterministic=deterministic
        else:
            self.deterministic=False

        self.randomness=randomness

        if ansiopvraha_kesto300 is not None:
            self.ansiopvraha_kesto300=ansiopvraha_kesto300
        else:
            self.ansiopvraha_kesto300=None

        if ansiopvraha_kesto400 is not None:
            self.ansiopvraha_kesto400=ansiopvraha_kesto400
        else:
            self.ansiopvraha_kesto400=None
        
        if ansiopvraha_toe is not None:
            self.ansiopvraha_toe=ansiopvraha_toe
        else:
            self.ansiopvraha_toe=None
                
        if perustulo is not None:
            self.perustulo=perustulo
        else:
            self.perustulo=False

        if env is None:
            self.environment='unemployment-v0'
        else:
            self.environment=env

        # alustetaan gym-environment
        if minimal:
            #if EK:
            #    self.environment='unemploymentEK-v0'
            #else:
            #    self.environment='unemployment-v0'

            self.minimal=True
            self.gym_kwargs={'step': self.timestep,'gamma':self.gamma,
                'min_age': self.min_age, 'max_age': self.max_age,\
                'min_retirementage': self.min_retirementage, 'max_retirementage':self.max_retirementage,\
                'deterministic': self.deterministic}
            self.n_employment = 3
            self.n_acts = 3
        else:
            #if EK:
            #    self.environment='unemploymentEK-v1'
            #else:
            #    self.environment='unemployment-v1'

            self.minimal=False
            self.gym_kwargs={'step': self.timestep,'gamma':self.gamma,\
                'min_age': self.min_age, 'max_age': self.max_age,\
                'min_retirementage': self.min_retirementage, 'max_retirementage':self.max_retirementage,\
                'ansiopvraha_kesto300': self.ansiopvraha_kesto300,'ansiopvraha_kesto400': self.ansiopvraha_kesto400,\
                'ansiopvraha_toe': self.ansiopvraha_toe,\
                'perustulos': self.perustulo, 'karenssi_kesto': self.karenssi_kesto,\
                'mortality': self.mortality, 'randomness': self.randomness,\
                'deterministic': self.deterministic}
            self.n_acts = 4
            if self.mortality:
                self.n_employment = 14
            else:
                self.n_employment = 13
            
        # Create log dir
        self.log_dir = "tmp/" # +str(env_id)
        os.makedirs(self.log_dir, exist_ok=True)
            
            
        self.env = gym.make(self.environment,kwargs=self.gym_kwargs)
                
    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        print('Parameters of lifecycle:\ntimestep {}\ngamma {} ({} per anno)\nmin_age {}\nmax_age {}\nmin_retirementage {}'.format(self.timestep,self.gamma,self.gamma**(1.0/self.timestep),self.min_age,self.max_age,self.min_retirementage))
        print('max_retirementage {}\nansiopvraha_kesto300 {}\nansiopvraha_kesto400 {}\nansiopvraha_toe {}'.format(self.max_retirementage,self.ansiopvraha_kesto300,self.ansiopvraha_kesto400,self.ansiopvraha_toe))
        print('perustulo {}\nkarenssi_kesto {}\nmortality {}\nrandomness {}\ndeterministic {}\n'.format(self.perustulo,self.karenssi_kesto,self.mortality,self.randomness,self.deterministic))

    def map_age(self,age,start_zero=False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)

    def episodestats_init(self):
        n_emps=self.n_employment
        self.empstate=np.zeros((self.n_time,n_emps))
        self.deceiced=np.zeros((self.n_time,1))
        self.alive=np.zeros((self.n_time,1))
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

        self.pop_num=0
        
    def episodestats(self,k,act,r,state,newstate,debug=False,plot=False,dyn=False):
        #if debug:
        #    print((int(state[0]),int(state[1]),state[2],state[3],state[4]),':',act,(int(newstate[0]),int(newstate[1]),newstate[2],newstate[3],newstate[4]))
            
        if dyn:
            n=k
        else:
            n=self.pop_num[k]
            
        if self.minimal:
            emp,_,_,a,_=self.env.state_decode(state) # current employment state
            newemp,_,newsal,a2,tis=self.env.state_decode(newstate)
        else:
            emp,_,_,_,a,_,_,_,_,_=self.env.state_decode(state) # current employment state
            newemp,_,newpen,newsal,a2,tis,paidpens,pink,toe,ura=self.env.state_decode(newstate)
            
        #if emp==2 and a<self.min_retirementage:
        #    emp=12
        #if newemp==2 and a2<self.min_retirementage:
        #    emp=12
            
        t=int(np.round((a-self.min_age)*self.inv_timestep))
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
                self.stat_tyoura[t,newemp]+=ura
                self.stat_toe[t,newemp]+=toe
                self.stat_pension[t,newemp]+=newpen
                self.stat_paidpension[t,newemp]+=paidpens
                self.stat_unemp_len[t,n]=tis

            if not emp==newemp:
                self.siirtyneet[t,emp]+=1
            else:
                self.pysyneet[t,emp]+=1
        elif newemp<0:
            self.deceiced[t]+=1
            
    def episodestats_exit(self):
        plt.close(self.episode_fig)
        
    def plot_statsV(self,aveV):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä')
        ax.set_ylabel('Keskiarvo V')
        mV=np.mean(aveV,axis=1)
        ax.plot(x,mV)
        plt.show()
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä')
        ax.set_ylabel('palkka')
        ax.plot(x,mV[:,0:10])
        plt.show()                
    
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
    
    def plot_emp(self):
        employed=self.empstate[:,1]
        retired=self.empstate[:,2]
        unemployed=self.empstate[:,0]
        if not self.minimal:
            disabled=self.empstate[:,3]
            piped=self.empstate[:,4]
            mother=self.empstate[:,5]
            dad=self.empstate[:,6]
            kotihoidontuki=self.empstate[:,7]
            vetyo=self.empstate[:,8]
            veosatyo=self.empstate[:,9]
            osatyo=self.empstate[:,10]
            outsider=self.empstate[:,11]
            student=self.empstate[:,12]
        
        if not self.minimal:
            tyollisyysaste=100*(employed+osatyo+veosatyo+vetyo)/self.alive[:,0]
            osatyoaste=100*(osatyo+veosatyo)/(employed+osatyo+veosatyo+vetyo)
            tyottomyysaste=100*(unemployed+piped)/(unemployed+employed+piped+osatyo+veosatyo+vetyo)
            ka_tyottomyysaste=100*np.sum(unemployed+piped)/np.sum(unemployed+employed+piped+osatyo+veosatyo+vetyo)
        else:
            tyollisyysaste=100*(employed)/self.alive[:,0]
            osatyoaste=np.zeros(self.alive.shape)
            tyottomyysaste=100*(unemployed)/(unemployed+employed)
            ka_tyottomyysaste=100*np.sum(unemployed)/np.sum(unemployed+employed)
        
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
        self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]')

        if not self.minimal:
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',ylimit=20)
        
        if not self.minimal:
            #x=np.linspace(self.min_age,self.max_age,self.n_time)
            #fig,ax=plt.subplots()
            #ax.plot(x,outsider_ratio,label='ulkopuolella')
            #ax.set_xlabel('Ikä [v]')
            #ax.set_ylabel('Osuus tilassa [%]')
            #ax.set_title('Virhe')
            ##ax.legend()
            #plt.show()
            
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',parent=True)
            self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',unemp=True)
        
        self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',start_from=60)
        
    def plot_pensions(self):
        if not self.minimal:
            self.plot_ratiostates(self.stat_pension,ylabel='Tuleva eläke [e/v]')
        
    def plot_career(self):            
        if not self.minimal:
            self.plot_ratiostates(self.stat_tyoura,ylabel='Työuran pituus [v]')

    def plot_ratiostates(self,statistic,ylabel='',ylimit=None, show_legend=True, parent=False,\
                         unemp=False,start_from=None):
        self.plot_states(statistic/self.empstate,ylabel=ylabel,ylimit=ylimit,\
                    show_legend=show_legend,parent=parent,unemp=unemp,start_from=start_from)

    def plot_states(self,statistic,ylabel='',ylimit=None,show_legend=True,parent=False,unemp=False,\
                    start_from=None):
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
        plt.show()    
        
    def plot_toe(self):            
        if not self.minimal:
            self.plot_ratiostates(self.stat_toe,'Työssäolo-ehdon pituus 28 kk aikana [v]')
        
    def plot_sal(self):
        self.plot_ratiostates(self.salaries_emp,'Keskipalkka [e/v]')

    def plot_moved(self):
        siirtyneet_ratio=self.siirtyneet/self.alive
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet tilasta')
        pysyneet_ratio=self.pysyneet/self.alive
        self.plot_states(pysyneet_ratio,ylabel='Pysyneet tilassa')
        
    def plot_ave_stay(self):
        self.plot_ratiostates(self.time_in_state,ylabel='Ka kesto tilassa')
        
    def plot_reward(self):
        self.plot_ratiostates(self.rewstate,ylabel='Keskireward tilassa')
        
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
        
    def get_multiprocess_env(self,rlmodel,save,start_from=None,debug=False,modify_load=True,dir='saved'):
    
        if start_from is None:
            start_from=save
            
        # multiprocess environment
        if rlmodel=='a2c':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 16])
            n_cpu = 4
        elif rlmodel=='acer':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 16])
            n_cpu = 4
        elif rlmodel=='acktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[512, 512]) # 256, 256?
            n_cpu = 12
        elif rlmodel=='lstm':
            policy_kwargs = dict()
            n_cpu = 4
        elif rlmodel=='trpo':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 16])
            n_cpu = 4
        else:
            policy_kwargs = dict()
            n_cpu = 1
            rlmodel='dqn'
            
        if modify_load:
            savename=dir+'/'+rlmodel+'_'+save
            loadname=dir+'/'+rlmodel+'_'+start_from
        else:
            savename=save
            loadname=start_from
            
        if debug:
            n_cpu=1
        
        return policy_kwargs,n_cpu,savename,loadname
        
    def setup_rlmodel(self,rlmodel,loadname,env,batch,policy_kwargs,learning_rate,max_grad_norm,cont,tensorboard=False):
        #print('loadname=',loadname)
        if cont:
            if rlmodel=='a2c':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                if tensorboard:
                    model = A2C.load(loadname, env=env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                     tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
                else:
                    model = A2C.load(loadname, env=env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                     policy_kwargs=policy_kwargs)                
            elif rlmodel=='acer':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = ACER.load(loadname, env=env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                  tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            elif rlmodel=='acktr' or rlmodel=='lnacktr':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                       learning_rate=learning_rate, \
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                       learning_rate=learning_rate, \
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
            elif rlmodel=='lstm':
                from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy # for A2C, ACER
                model = ACKTR.load(loadname, env=env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                   tensorboard_log="./a2c_unemp_tensorboard/", learning_rate=learning_rate, \
                                   policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
            elif rlmodel=='trpo':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = TRPO.load(loadname, env=env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                   tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            else:        
                from stable_baselines.deepq.policies import MlpPolicy # for DQN
                model = DQN.load(loadname, env=env, verbose=1,gamma=self.gamma,batch_size=32,\
                                 learning_starts=self.n_time,\
                                 tensorboard_log="./a2c_unemp_tensorboard/",prioritized_replay=True, \
                                 policy_kwargs=policy_kwargs)
        else:
            if rlmodel=='a2c':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = A2C(MlpPolicy, env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time, \
                            tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            elif rlmodel=='acer':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = ACER(MlpPolicy, env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time, \
                             tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            elif rlmodel=='acktr':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                if tensorboard:
                    model = ACKTR(MlpPolicy, env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                tensorboard_log="./a2c_unemp_tensorboard/", learning_rate=learning_rate, \
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
                else:
                    model = ACKTR(MlpPolicy, env, verbose=0,gamma=self.gamma,n_steps=batch*self.n_time,\
                                learning_rate=learning_rate, \
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
            elif rlmodel=='lnacktr':
                from stable_baselines.common.policies import LnMlpPolicy # for A2C, ACER
                if tensorboard:
                    model = ACKTR(LnMlpPolicy, env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                tensorboard_log="./a2c_unemp_tensorboard/", learning_rate=learning_rate, \
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
                else:
                    model = ACKTR(LnMlpPolicy, env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                                learning_rate=learning_rate, \
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
            elif rlmodel=='lstm':
                from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy # for A2C, ACER
                model = ACKTR(MlpLstmPolicy, env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time,\
                            tensorboard_log="./a2c_unemp_tensorboard/", learning_rate=learning_rate, \
                            policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
            elif rlmodel=='trpo':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = TRPO(MlpPolicy, env, verbose=1,gamma=self.gamma,n_steps=batch*self.n_time, \
                             tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            else:
                from stable_baselines.deepq.policies import MlpPolicy # for DQN
                model = DQN(MlpPolicy, env, verbose=1,gamma=self.gamma,batch_size=32, \
                            learning_starts=self.n_time,\
                            tensorboard_log="./a2c_unemp_tensorboard/",prioritized_replay=True, \
                            policy_kwargs=policy_kwargs) 
                            
        return model            
                
    def make_env(self,env_id, rank, kwargs, seed=None, use_monitor=True):
        """
        Utility function for multiprocessed env.#

        :param env_id: (str) the environment ID
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_id,kwargs=kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.env_seed(seed + rank + 100)

            if use_monitor:
                env = Monitor(env, self.log_dir, allow_early_resets=True)

            return env

        if seed is not None:
            set_global_seeds(seed)
    
        return _init()
        
    def callback(self,_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        #global n_steps, best_mean_reward
        # Print stats every 1000 calls
        min_steps=1000
        mod_steps=50
        if (self.n_steps + 1) % mod_steps == 0 and self.n_steps > min_steps:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            #print(x,y)
            if len(x) > 0:
                mean_reward = np.mean(y[-min_steps:])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(self.log_dir + 'best_model.pkl')
        self.n_steps += 1
        return True        
        
    def train(self,train=False,debug=False,steps=20000,cont=False,rlmodel='dqn',\
                save='unemp',pop=None,batch=1,max_grad_norm=0.5,learning_rate=0.25,\
                start_from=None,modify_load=True,dir='saved',max_n_cpu=100,plot=True,\
                use_vecmonitor=True):

        self.best_mean_reward, self.n_steps = -np.inf, 0
        
        if pop is not None:
            self.n_pop=pop

        if start_from is None:
            start_from=save
            
        self.rlmodel=rlmodel
        
        self.episodestats_init()
        
        # multiprocess environment
        #print(save,type(dir))
        policy_kwargs,n_cpu,savename,loadname=self.get_multiprocess_env(self.rlmodel,save,\
                                                    start_from,debug=debug,modify_load=modify_load,dir=dir)  

        #print(savename,loadname)
        self.savename=save
        n_cpu=min(max_n_cpu,n_cpu)
        
        nonvec=False
        if nonvec:
            env=self.env
        else:
            if use_vecmonitor:
                env = SubprocVecEnv([lambda: self.make_env(self.environment, i, self.gym_kwargs, use_monitor=False) for i in range(n_cpu)])
                env = VecMonitor(env,filename='vecmonitor.json')
            else:
                env = SubprocVecEnv([lambda: self.make_env(self.environment, i, self.gym_kwargs) for i in range(n_cpu)])
            
            if False:
                env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])

        #normalize=False
        #if normalize:
        #    normalize_kwargs={}
        #    env = VecNormalize(env, **normalize_kwargs)
            
        model=self.setup_rlmodel(self.rlmodel,loadname,env,batch,policy_kwargs,learning_rate,max_grad_norm,cont)
        print('training...')
        model.learn(total_timesteps=steps, callback=self.callback,log_interval=100)
        model.save(savename)
        print('done')
        
        if plot:
            results_plotter.plot_results([self.log_dir], steps, results_plotter.X_TIMESTEPS, "Plotter")
            plt.show()
        
        del model 
        
    def save_simstats(self,filename,diff_htv,diff_tyoll,agg_htv,agg_tyoll,agg_rew,mean_emp,\
                      std_emp,median_emp,emps,best_rew,best_emp):
        f = h5py.File(filename, 'w')
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
               emps,best_rew,best_emp
        
    def save_to_hdf(self,filename,nimi,arr,dtype):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset(nimi, data=arr, dtype=dtype)
        f.close()
        
    def load_hdf(self,filename,nimi):
        f = h5py.File(filename, 'r')
        val=f.get(nimi).value
        f.close()        
        return val
        
    def save_sim(self,filename):
        f = h5py.File(filename, 'w')
        ftype='float64'
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
        dset = f.create_dataset('pop_num', data=self.pop_num, dtype='int64')
        f.close()
        
    def load_sim(self,filename):
        f = h5py.File(filename, 'r')
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
        self.pop_num=f.get('pop_num').value
        f.close()

    def simulate(self,debug=False,rlmodel=None,plot=True,load=None,pop=None,\
                 max_grad_norm=0.5,learning_rate=0.25,start_from=None,\
                 deterministic=False,save='simulate',modify_load=True,dir='saved'):

        if pop is not None:
            self.n_pop=pop
            
        if load is not None:
            self.savename=load
            
        if start_from is None:
            start_from=self.savename

        if rlmodel is not None:
            self.rlmodel=rlmodel

        self.episodestats_init()
        
        # multiprocess environment
        policy_kwargs,n_cpu,savename,loadname=self.get_multiprocess_env(self.rlmodel,self.savename,\
                                                    start_from,debug=debug,modify_load=modify_load,dir=dir)
        
        nonvec=False
        if nonvec:
            env=self.env
            #env.seed(4567)
            #env.env_seed(4567)            
        else:
            env = SubprocVecEnv([lambda: self.make_env(self.environment, i, self.gym_kwargs) for i in range(n_cpu)])
            #env = SubprocVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])
            #env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])

        normalize=False
        if normalize:
            normalize_kwargs={}
            env = VecNormalize(env, **normalize_kwargs)
            
        print('predicting...')

        if self.rlmodel=='a2c':
            model = A2C.load(savename, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        elif self.rlmodel=='acer':
            model = ACER.load(savename, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        elif self.rlmodel=='acktr':
            model = ACKTR.load(savename, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        elif self.rlmodel=='trpo':
            model = TRPO.load(savename, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        else:        
            model = DQN.load(savename, env=env, verbose=1,gamma=self.gamma,prioritized_replay=True,\
                             policy_kwargs=policy_kwargs)

        states = env.reset()
        n=n_cpu-1
        p=np.zeros(self.n_pop)
        self.pop_num=np.array([k for k in range(n_cpu)])
        tqdm_e = tqdm(range(int(self.n_pop)), desc='Population', leave=True, unit=" p")
        
        while n<self.n_pop:
            act, predstate = model.predict(states,deterministic=deterministic)
            newstate, rewards, dones, infos = env.step(act)

            done=False
            for k in range(n_cpu):
                #emp,pension,wage,age,time_in_state=self.state_decode(states[k])
                #print('Tila {} palkka {} ikä {} t-i-s {} eläke {}'.format(\
                #    emp,wage,age,time_in_state,pension))
                if dones[k]:
                    #print(infos[k]['terminal_observation'])
                    terminal_state=infos[k]['terminal_observation']
                    self.episodestats(k,0,rewards[k],states[k],newstate[k],debug=debug)                
                    self.episodestats(k,act[k],rewards[k],states[k],terminal_state,debug=debug)
                    tqdm_e.update(1)
                    n+=1
                    tqdm_e.set_description("Pop " + str(n))
                    done=True
                    self.pop_num[k]=n # =np.max(self.pop_num)+1
                else:
                    self.episodestats(k,act[k],rewards[k],states[k],newstate[k],debug=debug)                
                    
            #if done:
            #    states = env.reset()
            #else:
            states = newstate
        
        self.save_sim(save)
        
        print('done')        
            
        if plot:
            self.render()
        
        if False:
            return self.emp
            
    def render(self,load=None):
        if load is not None:
            self.load_sim(load)
    
        #self.plot_stats(5)
        self.plot_stats()
        #self.plot_reward()        
            
        
    def run_dummy(self,strategy='emp',debug=False,pop=None):
        '''
        Lasketaan työllisyysasteet ikäluokittain
        '''
        
        self.episodestats_init()
        if pop is not None:
            self.n_pop=pop
        
        print('simulating...')

        initial=(0,0,0,0,self.min_age,0,0)
        self.env.seed(1234) 
        self.pop_num=np.array([0])
        states = env.reset()
        n=0
        p=np.zeros(self.n_pop)
        self.pop_num=np.array([k for k in range(n_cpu)])
        while n<self.n_pop:
            emp,_,_,_,age=self.env.state_decode(state) # current employment state    

            if strategy=='random':
                act=np.random.randint(3)
            elif strategy=='emp':
                if age>=self.min_retirementage:
                    if emp!=2:
                        act=2
                    else:
                        act=0
                else:
                    if emp==0:
                        act=1
                    else:
                        act=0
            elif strategy=='unemp':
                if age>=self.retirement_age:
                    if emp!=2:
                        act=2
                    else:
                        act=0
                else:
                    if emp==1:
                        act=1
                    else:
                        act=0
            elif strategy=='alt':
                if age>=self.min_retirementage:
                    if emp!=2:
                        act=2
                    else:
                        act=0
                else:
                    act=1
                
            newstate,r,done,_=self.env.step(act)
            self.episodestats(0,act,r,state,newstate,debug=debug)
            state=newstate
            self.pop_num[0]=n # =np.max(self.pop_num)+1
            
            if done:
                if debug:
                    print('done')
                break        
                
        #self.plot_stats(5)        
        self.plot_stats()
        self.plot_reward()        

    def comp_tyollisyys_stats(self,emp,scale_time=True):
        demog=np.array([61663,63354,65939,68253,68543,71222,70675,71691,70202,70535,67315,68282,70431,72402,73839,\
                      73065,70040,69501,68857,69035,69661,69965,68429,65261,59498,61433,63308,65305,66580,71263,\
                      72886,73253,73454,74757,75406,74448,73940,73343,72808,70259,73065,74666,73766,73522,72213,\
                      74283,71273,73404,75153,75888])
                      
        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        if self.minimal:
            htv=np.round(scale*np.sum(demog[5:42]*(emp[5:42,1])))
            tyollvaikutus=np.round(scale*np.sum(demog[5:42]*(emp[5:42,1])))
            haj=np.mean(np.std(emp[5:42,1]))
        else:
            htv=np.round(scale*np.sum(demog[5:42]*(emp[5:42,1]+0.5*emp[5:42,10])))
            tyollvaikutus=np.round(scale*np.sum(demog[5:42]*(emp[5:42,1]+emp[5:42,10])))
            haj=np.mean(np.std((emp[5:42,1]+0.5*emp[5:42,10])))
            
        return htv,tyollvaikutus,haj

    def compare_with(self,cc2):
        diff_emp=self.empstate/self.n_pop-cc2.empstate/cc2.n_pop
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        #x=range(self.age_min,self.age_min+self.n_time)
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä')
        ax.set_ylabel('Ero työttömyysasteessa')
        ax.plot(x,diff_emp[:,0],label='työttömyys')
        ax.plot(x,diff_emp[:,1],label='kokoaikatyö')
        if not self.minimal:
            ax.plot(x,diff_emp[:,10],label='osa-aikatyö')
        ax.legend()
        plt.show()
        pop=np.array([61663,63354,65939,68253,68543,71222,70675,71691,70202,70535,67315,68282,70431,72402,73839,73065,70040,69501,68857,69035,69661,69965,68429,65261,59498,61433,63308,65305,66580,71263,72886,73253,73454,74757,75406,74448,73940,73343,72808,70259,73065,74666,73766,73522,72213,74283,71273,73404,75153,75888])
        
        if self.minimal:
            htv=np.round(cc2.timestep*np.sum(pop[5:42]*(diff_emp[5:42,1])))
            tyollvaikutus=np.round(cc2.timestep*np.sum(pop[5:42]*(diff_emp[5:42,1])))
            haj=np.mean(np.std(diff_emp[5:42,1]))
        else:
            htv=np.round(cc2.timestep*np.sum(pop[5:42]*(diff_emp[5:42,1]+0.5*diff_emp[5:42,10])))
            tyollvaikutus=np.round(cc2.timestep*np.sum(pop[5:42]*(diff_emp[5:42,1]+diff_emp[5:42,10])))
            haj=np.mean(np.std((diff_emp[5:42,1]+0.5*diff_emp[5:42,10])))
        print('Työllisyysvaikutus 25-62-vuotiaisiin noin {t} htv ja {h} työllistä'.format(t=htv,h=tyollvaikutus))
        
        # epävarmuus
        delta=1.96*1.0/np.sqrt(self.n_pop)
        print('Epävarmuus työllisyysasteissa {}, hajonta {}'.format(delta,haj))
        
    def run_results(self,n=2,steps1=100,steps2=100,pop=1_000,rlmodel='acktr',\
               save='perusmalli',debug=False,simut='simut',results='simut_res',\
               results_dir='results',save_dir='saved',deterministic=True,\
               train=True,predict=True,batch=1,cont=False,load=''):
               
        self.n_pop=pop

        if train:   
            if cont:
                self.ntrain(n=n,steps1=steps1,steps2=steps2,rlmodel=rlmodel,save=save,debug=debug,\
                            dir=save_dir,batch=batch,cont=cont,start_from=load)
            else:            
                self.ntrain(n=n,steps1=steps1,steps2=steps2,rlmodel=rlmodel,save=save,debug=debug,\
                            dir=save_dir,batch=batch)
        if predict:
            self.npredict(n=n,pop=pop,rlmodel=rlmodel,results=results_dir+'/'+results,\
                          load=save,debug=debug,save_dir=save_dir,deterministic=deterministic)
        self.run_simstats(results_dir+'/'+results,save=results_dir+'/'+results+'_stats')
        self.plot_simstats(results_dir+'/'+results+'_stats')
        
    def ntrain(self,n=10,steps1=2_000_000,steps2=1_000_000,rlmodel='acktr',\
               save='simut',debug=False,dir='saved',batch=1,cont=False,load=''):
               
        if cont:
            self.train(steps=steps1,cont=cont,rlmodel='acktr',save=save+'_100',batch=batch,debug=debug,\
                       add_dir=True,dir=save_dir,load=load)
        else:
            self.train(steps=steps1,cont=False,rlmodel='acktr',save=save+'_100',batch=batch,debug=debug,\
                       add_dir=True,dir=save_dir)
        
        for i in range(1,n):
            self.train(steps=steps2,cont=True,rlmodel=rlmodel,save=save+'_'+str(100+i),\
                       debug=debug,start_from=save+'_'+str(100+i-1),add_dir=True,dir=dir,batch=batch)
            
    def npredict(self,n=10,pop=1_00,rlmodel='acktr',results='simut_res',
                 load='malli',debug=False,save_dir='saved',deterministic=False):
        simut=np.zeros((n,self.n_time,self.n_employment))
        
        self.save_to_hdf(results+'_simut','n',n,dtype='int64')
    
        for i in range(0,n):
            self.simulate(pop=pop,rlmodel=rlmodel,plot=False,debug=debug,\
                          load=load+'_'+str(100+i),save=results+'_'+str(100+i),\
                          add_dir=True,dir=save_dir,deterministic=deterministic)
                          
    def get_reward(self):
        total_reward=np.sum(self.rewstate,axis=1)
        rr=np.sum(total_reward)/self.n_pop
        return rr
                          
    def run_simstats(self,results,save,plot=True):
        n=self.load_hdf(results+'_simut','n')
        e_rate=np.zeros((n,self.n_time))
        diff_rate=np.zeros((n,self.n_time))
        agg_htv=np.zeros(n)
        agg_tyoll=np.zeros(n)
        agg_rew=np.zeros(n)
        diff_htv=np.zeros(n)
        diff_tyoll=np.zeros(n)
        mean_hvt=np.zeros(self.n_time)
        std_htv=np.zeros(self.n_time)
        mean_emp=np.zeros((self.n_time,self.n_employment))
        std_emp=np.zeros((self.n_time,self.n_employment))
        emps=np.zeros((n,self.n_time,self.n_employment))

        self.load_sim(results+'_100')
        base_empstate=self.empstate/self.n_pop
        emps[0,:,:]=base_empstate
        htv_base,tyoll_base,haj_base=self.comp_tyollisyys_stats(base_empstate,scale_time=False)
        reward=self.get_reward()
        agg_htv[0]=htv_base
        agg_tyoll[0]=tyoll_base
        agg_rew[0]=reward
        best_rew=reward
        best_emp=0

        if plot:
            fig,ax=plt.subplots()
            ax.set_xlabel('työllisyysaste')
            ax.set_ylabel('lkm')
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            #ax.plot(x,100*tyoll_base)
        
        for i in range(1,n):        
            self.load_sim(results+'_'+str(100+i))
            empstate=self.empstate/self.n_pop
            emps[i,:,:]=empstate
            reward=self.get_reward()
            if reward>best_rew:
                best_rew=reward
                best_emp=i

            diff_emp=empstate-base_empstate
            if self.minimal:
                tyol_aste=(self.empstate[:,1])/self.n_pop
                diff_rate[i,:]=diff_emp[:,1]
            else:
                tyol_aste=(self.empstate[:,1]+self.empstate[:,10]+self.empstate[:,8]+self.empstate[:,9])/self.n_pop
                diff_rate[i,:]=diff_emp[:,1]+diff_emp[:,10]
            
            if plot:
                ax.plot(x,100*tyol_aste)

            e_rate[i,:]=tyol_aste
            htv,tyollvaikutus,haj=self.comp_tyollisyys_stats(empstate,scale_time=False)
            
            agg_htv[i]=htv
            agg_tyoll[i]=tyollvaikutus
            agg_rew[i]=reward
            diff_htv[i]=htv-htv_base
            diff_tyoll[i]=tyollvaikutus-tyoll_base
            
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
        print('best_emp',best_emp)
        
    def plot_simstats(self,filename):
        #print('load',filename)
        diff_htv,diff_tyoll,agg_htv,agg_tyoll,agg_rew,mean_emp,std_emp,median_emp,emps,\
            best_rew,best_emp=self.load_simstats(filename)
        
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
            print('Vaikutus työllisyysasteen keskiarvo {} htv mediaan {} htv'.format(mean_htv,median_htv))
        else:
            print('Vaikutus työllisyysasteen keskiarvo {} htv, mediaani {} htv\n                        keskiarvo {} työllistä, mediaani {} työllistä'.format(mean_htv,median_htv,mean_tyoll,median_tyoll))
        
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
        ax.set_xlabel('keskimääräinen työllisyys')
        ax.set_ylabel('työllisyysaste')
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
            best1_rew,best1_emp=self.load_simstats(filename1)
        
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

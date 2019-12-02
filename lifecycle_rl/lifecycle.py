'''

    lifecycle.py
    
    implements the lifecycle model that predicts how people will act in the presence of
    social security

'''

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
from fin_benefits import Benefits
import gym_unemployment
import h5py
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
import os
from . episodestats import EpisodeStats, SimStats

# use stable baselines
from stable_baselines.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines import A2C, ACER, DQN, ACKTR #, TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
from .vec_monitor import VecMonitor

class Lifecycle():

    def __init__(self,env=None,minimal=False,timestep=0.25,ansiopvraha_kesto300=None,
                    ansiopvraha_kesto400=None,karenssi_kesto=None,
                    ansiopvraha_toe=None,perustulo=None,mortality=None,
                    randomness=None,deterministic=None,include_putki=None):

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

        if include_putki is not None:
            self.include_putki=include_putki
        else:
            self.include_putki=None

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
                'min_age': self.min_age, 'max_age': self.max_age,
                'min_retirementage': self.min_retirementage, 'max_retirementage':self.max_retirementage,
                'deterministic': self.deterministic}
            self.n_employment = 3
            self.n_acts = 3
        else:
            #if EK:
            #    self.environment='unemploymentEK-v1'
            #else:
            #    self.environment='unemployment-v1'

            self.minimal=False
            self.gym_kwargs={'step': self.timestep,'gamma':self.gamma,
                'min_age': self.min_age, 'max_age': self.max_age,
                'min_retirementage': self.min_retirementage, 'max_retirementage':self.max_retirementage,
                'ansiopvraha_kesto300': self.ansiopvraha_kesto300,'ansiopvraha_kesto400': self.ansiopvraha_kesto400,
                'ansiopvraha_toe': self.ansiopvraha_toe,
                'perustulos': self.perustulo, 'karenssi_kesto': self.karenssi_kesto,
                'mortality': self.mortality, 'randomness': self.randomness,
                'deterministic': self.deterministic, 'include_putki': self.include_putki}
            self.n_acts = 4
            if self.mortality:
                self.n_employment = 14
            else:
                self.n_employment = 13
            
        # Create log dir & results dirs
        self.log_dir = "tmp/" # +str(env_id)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs("saved/", exist_ok=True)
        os.makedirs("results/", exist_ok=True)
            
        self.env = gym.make(self.environment,kwargs=self.gym_kwargs)
        self.episodestats=SimStats(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                   self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage)
                
    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        print('Parameters of lifecycle:\ntimestep {}\ngamma {} ({} per anno)\nmin_age {}\nmax_age {}\nmin_retirementage {}'.format(self.timestep,self.gamma,self.gamma**(1.0/self.timestep),self.min_age,self.max_age,self.min_retirementage))
        print('max_retirementage {}\nansiopvraha_kesto300 {}\nansiopvraha_kesto400 {}\nansiopvraha_toe {}'.format(self.max_retirementage,self.ansiopvraha_kesto300,self.ansiopvraha_kesto400,self.ansiopvraha_toe))
        print('perustulo {}\nkarenssi_kesto {}\nmortality {}\nrandomness {}\ndeterministic {}\n'.format(self.perustulo,self.karenssi_kesto,self.mortality,self.randomness,self.deterministic))
        print('include_putki {}\nstep {}\n'.format(self.include_putki,self.timestep))


    def map_age(self,age,start_zero=False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)
        
    def get_multiprocess_env(self,rlmodel,debug=False):
    
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
            
        if debug:
            n_cpu=1
        
        return policy_kwargs,n_cpu
        
    def setup_rlmodel(self,rlmodel,loadname,env,batch,policy_kwargs,learning_rate,
                      max_grad_norm,cont,tensorboard=False,verbose=1,n_cpu=1):
        #print('loadname=',loadname)
        
        batch=max(1,int(np.ceil(batch/n_cpu)))
        #print('batch',batch)
        
        if cont:
            if rlmodel=='a2c':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                if tensorboard:
                    model = A2C.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                     tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
                else:
                    model = A2C.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                     policy_kwargs=policy_kwargs)                
            elif rlmodel=='acer':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = ACER.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                  tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            elif rlmodel=='acktr' or rlmodel=='lnacktr':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       learning_rate=learning_rate, 
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       learning_rate=learning_rate, 
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
            elif rlmodel=='lstm':
                from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy # for A2C, ACER
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       tensorboard_log="./a2c_unemp_tensorboard/", learning_rate=learning_rate, 
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       learning_rate=learning_rate, policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)                
            elif rlmodel=='trpo':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = TRPO.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                   tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            else:        
                from stable_baselines.deepq.policies import MlpPolicy # for DQN
                model = DQN.load(loadname, env=env, verbose=verbose,gamma=self.gamma,batch_size=batch,
                                 learning_starts=self.n_time,
                                 tensorboard_log="./a2c_unemp_tensorboard/",prioritized_replay=True, 
                                 policy_kwargs=policy_kwargs)
        else:
            if rlmodel=='a2c':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = A2C(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time, 
                            tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            elif rlmodel=='acer':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = ACER(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time, 
                             tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            elif rlmodel=='acktr':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                if tensorboard:
                    model = ACKTR(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                tensorboard_log="./a2c_unemp_tensorboard/", learning_rate=learning_rate, 
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
                else:
                    model = ACKTR(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                learning_rate=learning_rate, 
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
            elif rlmodel=='lnacktr':
                from stable_baselines.common.policies import LnMlpPolicy # for A2C, ACER
                if tensorboard:
                    model = ACKTR(LnMlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                tensorboard_log="./a2c_unemp_tensorboard/", learning_rate=learning_rate, 
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
                else:
                    model = ACKTR(LnMlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                learning_rate=learning_rate, 
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
            elif rlmodel=='lstm':
                from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy # for A2C, ACER
                model = ACKTR(MlpLstmPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                            tensorboard_log="./a2c_unemp_tensorboard/", learning_rate=learning_rate, 
                            policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm)
            elif rlmodel=='trpo':
                from stable_baselines.common.policies import MlpPolicy # for A2C, ACER
                model = TRPO(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time, 
                             tensorboard_log="./a2c_unemp_tensorboard/", policy_kwargs=policy_kwargs)
            else:
                from stable_baselines.deepq.policies import MlpPolicy # for DQN
                model = DQN(MlpPolicy, env, verbose=verbose,gamma=self.gamma,batch_size=batch, 
                            learning_starts=self.n_time,
                            tensorboard_log="./a2c_unemp_tensorboard/",prioritized_replay=True, 
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

            #print('monitor=',use_monitor)
    
            #if use_monitor:
            #    env = Monitor(env, self.log_dir, allow_early_resets=True)

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
        
        min_steps=0
        mod_steps=1
        hist_eps=10000
        #print(_locals, _globals)
        if (self.n_steps + 1) % mod_steps == 0 and self.n_steps > min_steps:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            #print(x,y)
            if len(x) > hist_eps:
                mean_reward = np.mean(y[-hist_eps:])
                print(x[-1], 'timesteps', len(y), 'episodes') #, 'mean', mean_reward, 'out of', y[-hist_eps:])

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    print("New best mean reward: {:.2f} - Last best reward per episode: {:.2f}".format(mean_reward,self.best_mean_reward))
                    self.best_mean_reward = mean_reward
                    # Example for saving best model                    print("Saving new best model")
                    print('saved as ',self.bestname)
                    _locals['self'].save(self.bestname)
                else:
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                    
        self.n_steps += 1
            
        return True        
        
    def train(self,train=False,debug=False,steps=20000,cont=False,rlmodel='dqn',
                save='saved/malli',pop=None,batch=1,max_grad_norm=0.5,learning_rate=0.25,
                start_from=None,max_n_cpu=100,plot=True,use_vecmonitor=False,
                bestname='best2',use_callback=False,log_interval=100,verbose=1):

        self.best_mean_reward, self.n_steps = -np.inf, 0
        
        if pop is not None:
            self.n_pop=pop

        self.rlmodel=rlmodel
        self.bestname=bestname
        
        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage)
        
        # multiprocess environment
        #print(save,type(dir))
        policy_kwargs,n_cpu=self.get_multiprocess_env(self.rlmodel,debug=debug)  

        #print(savename,loadname)
        self.savename=save
        n_cpu=min(max_n_cpu,n_cpu)
        
        if debug:
            print('use_vecmonitor',use_vecmonitor)
            print('use_callback',use_callback)
        
        nonvec=False
        if nonvec:
            env=self.env
        else:
            if use_vecmonitor:
                env = SubprocVecEnv([lambda: self.make_env(self.environment, i, self.gym_kwargs, use_monitor=False) for i in range(n_cpu)])
                env = VecMonitor(env,filename=self.log_dir+'monitor.csv')
            else:
                env = SubprocVecEnv([lambda: self.make_env(self.environment, i, self.gym_kwargs, use_monitor=use_callback) for i in range(n_cpu)])
            
            if False:
                env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])

        #normalize=False
        #if normalize:
        #    normalize_kwargs={}
        #    env = VecNormalize(env, **normalize_kwargs)
        
        model=self.setup_rlmodel(self.rlmodel,start_from,env,batch,policy_kwargs,learning_rate,
                                    max_grad_norm,cont,verbose=verbose,n_cpu=n_cpu)
        print('training...')
        
        if use_callback: # tässä ongelma, vecmonitor toimii => kuitenkin monta callbackia
            model.learn(total_timesteps=steps, callback=self.callback,log_interval=log_interval)
        else:
            model.learn(total_timesteps=steps, log_interval=log_interval)
            
        model.save(save)
        print('done')
        
        del model,env
        
    def save_to_hdf(self,filename,nimi,arr,dtype):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset(nimi, data=arr, dtype=dtype)
        f.close()
        
    def load_hdf(self,filename,nimi):
        f = h5py.File(filename, 'r')
        val=f.get(nimi).value
        f.close()        
        return val
        
    def simulate(self,debug=False,rlmodel=None,plot=True,load=None,pop=None,
                 max_grad_norm=0.5,learning_rate=0.25,
                 deterministic=False,save='simulate'):

        if pop is not None:
            self.n_pop=pop
            
        if load is not None:
            self.loadname=load
            
        if rlmodel is not None:
            self.rlmodel=rlmodel

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage)
        
        print('simulating ',self.loadname)
        
        # multiprocess environment
        policy_kwargs,n_cpu=self.get_multiprocess_env(self.rlmodel,debug=debug)
        
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
            model = A2C.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        elif self.rlmodel=='acer':
            model = ACER.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        elif self.rlmodel=='acktr':
            model = ACKTR.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        elif self.rlmodel=='trpo':
            model = TRPO.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        else:        
            model = DQN.load(load, env=env, verbose=1,gamma=self.gamma,prioritized_replay=True,
                             policy_kwargs=policy_kwargs)

        states = env.reset()
        n=n_cpu-1
        p=np.zeros(self.n_pop)
        pop_num=np.array([k for k in range(n_cpu)])
        tqdm_e = tqdm(range(int(self.n_pop)), desc='Population', leave=True, unit=" p")
        
        while n<self.n_pop:
            act, predstate = model.predict(states,deterministic=deterministic)
            newstate, rewards, dones, infos = env.step(act)

            done=False
            for k in range(n_cpu):
                #emp,pension,wage,age,time_in_state=self.state_decode(states[k])
                #print('Tila {} palkka {} ikä {} t-i-s {} eläke {}'.format(
                #    emp,wage,age,time_in_state,pension))
                if dones[k]:
                    #print(infos[k]['terminal_observation'])
                    terminal_state=infos[k]['terminal_observation']  
                    self.episodestats.add(pop_num[k],0,rewards[k],states[k],newstate[k],debug=debug)
                    self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],terminal_state,debug=debug)
                    tqdm_e.update(1)
                    n+=1
                    tqdm_e.set_description("Pop " + str(n))
                    done=True
                    pop_num[k]=n
                else:
                    self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],newstate[k],debug=debug)                
                    
            #if done:
            #    states = env.reset()
            #else:
            states = newstate
        
        self.episodestats.save_sim(save)
        
        #print('done')        
            
        if plot:
            self.render()
        
        if False:
            return self.emp
           
    def render(self,load=None):
        if load is not None:
            self.episodestats.render(load=load)
        else:
            self.episodestats.render()
           
    def run_dummy(self,strategy='emp',debug=False,pop=None):
        '''
        Lasketaan työllisyysasteet ikäluokittain
        '''
        
        self.episodestats.reset()
        if pop is not None:
            self.n_pop=pop
        
        print('simulating...')

        initial=(0,0,0,0,self.min_age,0,0)
        self.env.seed(1234) 
        states = env.reset()
        n=0
        p=np.zeros(self.n_pop)
        pop_num=np.array([k for k in range(n_cpu)])
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
            self.episodestats.add(pop_num[0],act,r,state,newstate,debug=debug)
            state=newstate
            pop_num[0]=n
            
            if done:
                if debug:
                    print('done')
                break        
                
        #self.plot_stats(5)        
        self.plot_stats()
        self.plot_reward()        

    def compare_with(self,cc2):
        '''
        compare_with
        
        compare results obtained another model
        '''
        self.episodestats.compare_with(cc2.episodestats)
        
    def run_results(self,steps1=100,steps2=100,pop=1_000,rlmodel='acktr',
               save='saved/perusmalli',debug=False,simut='simut',results='results/simut_res',
               deterministic=True,train=True,predict=True,batch1=1,batch2=100,cont=False,
               start_from=None,bestname='tmp/best1',plot=False):
               
        '''
        run_results
        
        train a model based on a protocol, and then simulate it
        plot results if needed
        '''
               
        self.n_pop=pop

        if train: 
            print('train...')
            if cont:
                self.run_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,
                                save=save,debug=debug,bestname=bestname,
                                batch1=batch1,batch2=batch2,cont=cont,start_from=start_from)
            else:            
                self.run_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,save=save,
                                 debug=debug,batch1=batch1,batch2=batch2,cont=cont,
                                 bestname=bestname)
        if predict:
            print('predict...')
            self.predict_protocol(pop=pop,rlmodel=rlmodel,results=results,
                          load=save,debug=debug,deterministic=deterministic,
                          bestname=bestname)

        self.episodestats.run_simstats(results,save=results+'_stats')
        self.episodestats.plot_simstats(results+'_stats')
        self.episodestats.load_sim(results+'_best')
                          
    def run_protocol(self,steps1=2_000_000,steps2=1_000_000,rlmodel='acktr',
               save='results/simut',debug=False,batch1=1,batch2=1000,cont=False,
               start_from=None,bestname='best3'):
        '''
        run_protocol
        
        train RL model in two steps:
        1. train with a short batch, not saving the best model
        2. train with a long batch, save the best model during the training
        '''
              
        print('phase 1')
        if cont:
            self.train(steps=steps1,cont=cont,rlmodel='acktr',save=save+'_100',batch=batch1,debug=debug,
                       start_from=start_from,use_callback=False,use_vecmonitor=False,
                       log_interval=1000,verbose=1)
        else:
            self.train(steps=steps1,cont=False,rlmodel='acktr',save=save+'_100',batch=batch1,debug=debug,
                       use_callback=False,use_vecmonitor=False,
                       log_interval=1000,verbose=1)
        
        print('phase 2')
        self.train(steps=steps2,cont=True,rlmodel=rlmodel,save=save+'_101',
                   debug=debug,start_from=save+'_100',batch=batch2,
                   use_callback=True,use_vecmonitor=True,log_interval=1,bestname=bestname)
            
    def predict_protocol(self,pop=1_00,rlmodel='acktr',results='results/simut_res',
                 load='saved/malli',debug=False,deterministic=False,bestname='best5',
                 onlybest=True):
        '''
        predict_protocol
        
        simulate the three models obtained from run_protocol
        '''
                 
        if not onlybest:
            self.save_to_hdf(results+'_simut','n',3,dtype='int64')
    
            for i in range(0,2):
                self.simulate(pop=pop,rlmodel=rlmodel,plot=False,debug=debug,
                              load=load+'_'+str(100+i),save=results+'_'+str(100+i),
                              deterministic=deterministic)
            # simulate the saved best
            self.simulate(pop=pop,rlmodel=rlmodel,plot=False,debug=debug,
                          load=bestname,save=results+'_102',
                          deterministic=deterministic)
        else:
            self.save_to_hdf(results+'_simut','n',1,dtype='int64')
        
            # simulate the saved best
            self.simulate(pop=pop,rlmodel=rlmodel,plot=False,debug=debug,
                          load=bestname,save=results+'_100',
                          deterministic=deterministic)

    def run_verify(self,n=5,steps1=100,steps2=100,pop=1_000,rlmodel='acktr',
               save='saved/perusmalli_verify',debug=False,simut='simut',results='results/simut_res',
               deterministic=True,train=True,predict=True,batch1=1,batch2=100,cont=False,
               start_from=None,bestname='tmp/best1',plot=False):
               
        '''
        run_results
        
        train a model based on a protocol, and then simulate it
        plot results if needed
        '''
               
        self.n_pop=pop
        
        for num in range(n):
        
            bestname2=bestname+'_v'+str(num)
            results2=results+'_v'+str(num)+'_'

            if train: 
                print('{}: train...'.format(num))
                if cont:
                    self.run_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,
                                    save=save,debug=debug,bestname=bestname2,
                                    batch1=batch1,batch2=batch2,cont=cont,start_from=start_from)
                else:            
                    self.run_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,save=save,
                                     debug=debug,batch1=batch1,batch2=batch2,cont=False,
                                     bestname=bestname2)
            if predict:
                print('{}: predict...'.format(num))
                self.predict_protocol(pop=pop,rlmodel=rlmodel,results=results2,
                              load=save,debug=debug,deterministic=deterministic,
                              bestname=bestname2)

            self.episodestats.run_simstats(results2,save=results2+'_stats_'+str(n))
            self.episodestats.plot_simstats(results2+'_stats_'+str(n))

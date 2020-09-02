'''

Bayesian optimization for lifecycle models

The aim is to reproduce employment rate at each age

'''

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events

from . lifecycle import Lifecycle

Class OptimizeLifecycle:
    def __init__(self,env=None,minimal=False,timestep=0.25,ansiopvraha_kesto300=None,
                    ansiopvraha_kesto400=None,karenssi_kesto=None,
                    ansiopvraha_toe=None,perustulo=None,mortality=None,
                    randomness=None,include_putki=None,preferencenoise=None,
                    callback_minsteps=None,pinkslip=True,plotdebug=False,
                    use_sigma_reduction=None,porrasta_putki=None,perustulomalli=None,
                    porrasta_1askel=None,porrasta_2askel=None,porrasta_3askel=None,
                    osittainen_perustulo=None,gamma=None,exploration=None,exploration_ratio=None,
                    year=2018):
        '''
        Alusta muuttujat
        '''
        self.lf=Lifecycle()
        
    def black_box_function(x, y):
        """Function with unknown internals we wish to maximize.

        
        """
        self.lf.run_results()
        return -self.lf.L2error()

    def optimize(self):
        # Bounded region of parameter space
        pbounds = {'kappa_kokoaika_mies': (0.4, 0.7), 'delta_kappa_kokoaika_nais': (0.01, 0.1), 
                   'mu_scale': (0.1,0.3), 'mu_age': (57,63)}

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        # talletus
        logger = JSONLogger(path="./logs.json")
        load_logs(optimizer, logs=["./logs.json"]);
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=2,
            n_iter=3,
        )
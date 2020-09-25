'''

Bayesian optimization for lifecycle models

The aim is to reproduce employment rate at each age

'''

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from pathlib import Path
from os import path

from .lifecycle import Lifecycle

class OptimizeLifecycle():
    def __init__(self,initargs=None,runargs=None):
        '''
        Alusta muuttujat
        '''
        self.runargs=runargs
        self.lf=Lifecycle(**initargs)
        
    def black_box_function(self,**x):
        """Function with unknown internals we wish to maximize.

        
        """
        print(x)
        self.lf.env.set_utility_params(**x)
        self.lf.run_results(**self.runargs)
        return -self.lf.L2error()

    def optimize(self,reset=False):
        # Bounded region of parameter space
        pbounds = {'men_kappa_fulltime': (0.3, 0.6), 'men_kappa_parttime': (0.3, 0.7), #'men_mu_scale': (0.01,0.3), 'men_mu_age': (57,62),
                   'women_kappa_fulltime': (0.3, 0.6), 'women_kappa_parttime': (0.3, 0.7)} #, 'women_mu_scale': (0.01,0.3), 'women_mu_age': (57,62)}

        optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        
        LOG_DIR = Path().absolute() / 'bayes_opt_logs'
        LOG_DIR.mkdir(exist_ok=True)
        filename = 'log_0.json'

        # talletus
        logfile=str(LOG_DIR / filename)
        logger = JSONLogger(path=logfile)
        if Path(logfile).exists() and not reset:
            load_logs(optimizer, logs=[logfile]);
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=2,
            n_iter=20,
        )
        
        print('The best parameters found {}'.format(optimizer.max))
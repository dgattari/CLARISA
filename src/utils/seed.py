
# src/utils/seed.py

from src.train import legacy_train_base as base # esto vamos a tener que refactorizar cuando toque el base

def set_global_seed(seed: int): # vamos a tener que refactorizar el base de aqui
    base.set_global_seed(seed)
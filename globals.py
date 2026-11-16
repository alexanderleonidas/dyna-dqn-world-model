import torch
import os

# if GPU is to be used
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Set the random seed
SEED = 42

# Hyperparameters
NUM_EPISODES = 500 # number of episodes to train the agent for
BATCH_SIZE = 128 # number of transitions sampled from the replay buffer
MEMORY_CAPACITY = 10000 # size of the replay buffer
GAMMA = 0.99 # discount factor as mentioned in the previous section
EPS_START = 1.0 # starting value of epsilon
EPS_END = 0.05 # final value of epsilon
EPS_DECAY = 2500 # controls the rate of exponential decay of epsilon, higher value means slower decay
TAU = 0.005 # update rate of the target network
LR = 3e-4 # learning rate of the optimiser (also used for the World Model if applicable)

# World Model hyperparameters
PLANNING_STEPS = 2

# Optuna hyperparameter optimization settings
NUM_TRIALS = 10
TIMEOUT = 600
OBJECTIVE = 'avg_100' # 'avg_100', 'avg_all', 'real_steps'

# Experiment directories
RESULTS_DIR = "results" # directory to save the results
MODEL_DIR = "models" # directory to save the models

# Make directories if they don't exist'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
import numpy as np

# this code generates data that simulates two tasks which elicit a 
# shared activation pattern for certain voxels(default is features 10-20)
# this is presented here as each task having two "rules" with one rule in common for both tasks

#### arguments:
# signal = SNR, defined as ratio of mean/std

# trials = number of trials per task set

# inf_features = number of informative features per rule

# non_inf_features = number of non-informative features
####

def sim_data(signal, trials=100, inf_features=10, non_inf_features=270):

  # task set 1: rule A and rule B have mean equal to signal
  ts1_rule_A = np.random.normal(signal,1,(trials,inf_features))
  ts1_rule_B = np.random.normal(signal,1,(trials,inf_features))
  ts1_rule_C = np.random.normal(0,1,(trials,inf_features))
  ts1_rule_0 = np.random.normal(0,1,(trials,non_inf_features))

  ts1 = np.hstack((ts1_rule_A,ts1_rule_B,ts1_rule_C,ts1_rule_0))

  # set 2: rule B and rule C have mean equal to signal
  ts2_rule_A = np.random.normal(0,1,(trials,inf_features))
  ts2_rule_B = np.random.normal(signal,1,(trials,inf_features))
  ts2_rule_C = np.random.normal(signal,1,(trials,inf_features))
  ts2_rule_0 = np.random.normal(0,1,(trials,non_inf_features))

  ts2 = np.hstack((ts2_rule_A,ts2_rule_B,ts2_rule_C,ts2_rule_0))

  samples = np.vstack((ts1,ts2))
  labels = np.repeat(np.array([0,1]),trials)
  
  return samples, labels

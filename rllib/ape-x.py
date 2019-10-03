import ray
from ray import tune
from ray.rllib.agents.dqn.apex import ApexTrainer

ray.init()
tune.run(ApexTrainer, config={"env": "BreakoutNoFrameskip-v4",
                              "double_q": False,
                              "dueling": False,
                              "num_atoms": 1,
                              "noisy": False,
                              "n_step": 3,
                              "lr": .0001,
                              "adam_epsilon": .00015,
                              "hiddens": [512],
                              "buffer_size": 1000000,
                              "schedule_max_timesteps": 2000000,
                              "exploration_final_eps": 0.01,
                              "exploration_fraction": .1,
                              "prioritized_replay_alpha": 0.5,
                              "beta_annealing_fraction": 1.0,
                              "final_prioritized_replay_beta": 1.0,
                              "num_gpus": 1,
                              "num_workers": 10,
                              "num_envs_per_worker": 5,
                              "sample_batch_size": 20,
                              "train_batch_size": 512,
                              "per_worker_exploration": True,
                              "worker_side_prioritization": True,
                              "target_network_update_freq": 50000,
                              "timesteps_per_iteration": 25000
                              })  # "eager": True for eager execution

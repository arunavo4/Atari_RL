import ray
from ray import tune
from ray.rllib.agents.dqn.dqn import DQNTrainer

ray.init()
tune.run(DQNTrainer,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         stop={
             "episode_reward_mean": 20,
         },
         config={"env": "PongDeterministic-v4",
                 "num_atoms": 51,
                 "noisy": True,
                 "gamma": 0.99,
                 "lr": .0001,
                 "hiddens": [512],
                 "learning_starts": 10000,
                 "buffer_size": 50000,
                 "sample_batch_size": 4,
                 "train_batch_size": 32,
                 "schedule_max_timesteps": 2000000,
                 "exploration_final_eps": 0.0,
                 "exploration_fraction": .000001,
                 "target_network_update_freq": 500,
                 "prioritized_replay": True,
                 "prioritized_replay_alpha": 0.5,
                 "beta_annealing_fraction": 0.2,
                 "final_prioritized_replay_beta": 1.0,
                 "n_step": 3,
                 "num_gpus": 1,
                 "model": {
                     "grayscale": True,
                     "zero_mean": False,
                     "dim": 42
                 }

                 })  # "eager": True for eager execution

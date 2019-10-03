import ray
from ray import tune
from ray.rllib.agents.impala import ImpalaTrainer


ray.init()
tune.run(ImpalaTrainer, config={"env": "BreakoutNoFrameskip-v4",
                                "sample_batch_size": 50,
                                "train_batch_size": 500,
                                "num_workers": 10,
                                "num_envs_per_worker": 5,
                                "lr_schedule": [
                                    [0, 0.0005],
                                    [20000000, 0.000000000001],
                                ],
                                },
         )  # "eager": True for eager execution

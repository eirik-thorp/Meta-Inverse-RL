import click
import metaworld
import torch

from pemirl2 import PEMIRL, PEMIRLWorker
from context_conditioned_policy import ContextConditionedPolicy

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.trainer import Trainer
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.embeddings import MLPEncoder
from garage.torch import global_device
from garage.torch import set_gpu_mode


@wrap_experiment(archive_launch_repo=False)
def MetaIRL_pick_place(ctxt=None, use_gpu=True):

    set_seed(1)

    num_train_tasks = 100
    num_test_tasks = 15

    ml1 = metaworld.ML1('pick-place-v1')
    train_env = MetaWorldSetTaskEnv(ml1, 'train')
    env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                 env=train_env,
                                 wrapper=lambda env, _: normalize(env))

    env = env_sampler.sample(num_train_tasks)

    test_env = MetaWorldSetTaskEnv(ml1, 'test')
    test_env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                      env=test_env,
                                      wrapper=lambda env, _: normalize(env))
    
    latent_dim = 9
    augmented_env = PEMIRL.augment_env_spec(env[0](), latent_dim)
    net_size = 64
    inner_policy = GaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size])
    
    sampler = LocalSampler(agents=None,
                           envs=env[0](),
                           max_episode_length=env[0]().spec.max_episode_length,
                           n_workers=1,
                           worker_class=PEMIRLWorker)

    pemirl = PEMIRL(env,
                    experts_dir='data/experts/ml1_pick_place_train.pkl',
                    policy_class=ContextConditionedPolicy,
                    inner_policy=inner_policy,
                    reward_encoder_class=MLPEncoder,
                    reward_encoder_hidden_sizes=(64, 64),
                    context_encoder_class=GaussianMLPPolicy,
                    context_encoder_hidden_sizes=(128, 128),
                    latent_dim=latent_dim,
                    sampler=sampler,
                    test_env_sampler=test_env_sampler,
                    optimizer_class=torch.optim.Adam,
                    num_train_tasks=num_train_tasks,
                    num_test_tasks=num_test_tasks,
                    num_steps_per_epoch=10,
                    num_initial_steps=1000,
                    meta_batch_size=50,
                    batch_size=16,
                    airl_itr=10,
                    use_information_bottleneck=True,
                    use_next_obs_in_context=False,
                    replay_buffer_size=5000000,
                    discount=0.99,
                    state_only=True)
    
    set_gpu_mode(use_gpu, gpu_id=0)
    
    if use_gpu:
        pemirl.to()

    trainer = Trainer(ctxt)
    trainer.setup(algo=pemirl, env=env[0]())
    trainer.train(n_epochs=3000)

if __name__ == "__main__":
    MetaIRL_pick_place()

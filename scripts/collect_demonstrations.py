import metaworld
import pickle
from collections import defaultdict

from garage.experiment import Snapshotter
from garage.torch.algos.pearl import PEARLWorker
from garage import EpisodeBatch
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.sampler import LocalSampler, WorkerFactory
from garage.experiment.deterministic import get_seed
from garage.envs.mujoco import HalfCheetahVelEnv
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage import rollout
import time

def collect_demonstrations(train_env_sampler, test_env_sampler, n_train_tasks, n_test_tasks, eps_per_task, snap_dir):

    train_env_updates = train_env_sampler.sample(n_train_tasks)
    test_env_updates = test_env_sampler.sample(n_test_tasks)

    snapshotter = Snapshotter()
    data = snapshotter.load(snap_dir)
    algo = data["algo"]

    worker_args = dict(deterministic=True, accum_context=True)

    single_env = train_env_updates[0]()
    #print(single_env.spec.max_episode_length)
    max_episode_length = single_env.spec.max_episode_length

    sampler = LocalSampler.from_worker_factory(
            WorkerFactory(seed=get_seed(),
                            max_episode_length=max_episode_length,
                            n_workers=1,
                            worker_class=PEARLWorker,
                            worker_args=worker_args),
            agents=algo.get_exploration_policy(),
            envs=single_env)

    n_exploration_eps = 20

    train_paths = defaultdict(list)
    test_paths = defaultdict(list)

    for eval_itr, env_up in enumerate(train_env_updates):
        print(eval_itr)

        env = env_up()

        good = False
        tries = 0
        while not good:
            policy = algo.get_exploration_policy()
            eps = EpisodeBatch.concatenate(*[
                sampler.obtain_samples(eval_itr, 1, policy,
                                                    env)
                for _ in range(n_exploration_eps)
            ])

            adapted_policy = algo.adapt_policy(policy, eps)

            all_adapted_eps = sampler.obtain_samples(
                eval_itr,
                20 * eps_per_task * max_episode_length,
                adapted_policy, env)
                
            for adapted_eps in all_adapted_eps.to_list():
                tries += 1
                if (adapted_eps['env_infos']['success'].any()):
                    train_paths[eval_itr].append(adapted_eps)
                    print("good")
                    good = True
                if tries > 30 and not good:
                    test_paths[eval_itr].append(adapted_eps)
                    print("bad")
                    good = True
        
    for eval_itr, env_up in enumerate(test_env_updates):
        print(eval_itr)

        env = env_up()

        good = False
        tries = 0
        while not good:
            policy = algo.get_exploration_policy()
            eps = EpisodeBatch.concatenate(*[
                sampler.obtain_samples(eval_itr, 1, policy,
                                                    env)
                for _ in range(n_exploration_eps)
            ])

            adapted_policy = algo.adapt_policy(policy, eps)

            all_adapted_eps = sampler.obtain_samples(
                eval_itr,
                20 * eps_per_task * max_episode_length,
                adapted_policy, env)
                
            for adapted_eps in all_adapted_eps.to_list():
                tries += 1
                if (adapted_eps['env_infos']['success'].any()):
                    test_paths[eval_itr].append(adapted_eps)
                    print("good")
                    good = True
                    tries = 0
                if tries > 30 and not good:
                    test_paths[eval_itr].append(adapted_eps)
                    print("bad")
                    good = True

    return train_paths, test_paths

if __name__ == "__main__":

    set_seed(1)

    max_episode_length = 200
    n_train_tasks = 100
    n_test_tasks = 15
    eps_per_task = 10
    snap_dir = 'data/local/experiment/pearl_metaworld_ml1_pick_place'

    ml1 = metaworld.ML1('pick-place-v1')
    train_env = MetaWorldSetTaskEnv(ml1, 'train')
    train_env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                 env=train_env,
                                 wrapper=lambda env, _: normalize(env))
    
    test_env = MetaWorldSetTaskEnv(ml1, 'test')
    test_env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                      env=test_env,
                                      wrapper=lambda env, _: normalize(env))
    
    train_paths, test_paths = collect_demonstrations(
        train_env_sampler, test_env_sampler, n_train_tasks, 
        n_test_tasks, eps_per_task, snap_dir)

    with open('data/experts/ml1_pick_place_train.pkl', 'wb') as f:
        pickle.dump(train_paths, f)
    with open('data/experts/ml1_pick_place_test.pkl', 'wb') as f:
        pickle.dump(test_paths, f)

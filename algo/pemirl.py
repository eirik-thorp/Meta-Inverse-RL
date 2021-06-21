import numpy as np
import copy
from collections import defaultdict
import pickle

import torch
import torch.functional as F

import akro
from dowel import logger, tabular

from garage import EnvSpec, InOutSpec, StepType
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker

from garage.np import discount_cumsum
from garage.np.algos import RLAlgorithm
from garage.np.optimizers import BatchDataset

from garage.torch import (global_device, compute_advantages, filter_valids)
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.optimizers import (ConjugateGradientOptimizer,
                                     OptimizerWrapper)


class PEMIRL(RLAlgorithm):

    r"""A PEMIRL model based on https://arxiv.org/abs/1909.09314.
    Args:
        env (list[Environment]): Batch of sampled environment updates(
            EnvUpdate), which, when invoked on environments, will configure
            them with new tasks.
        experts_dir (String): Directory where expert trajectories are stored
        policy_class (type): Class implementing
                :pyclass:`~ContextConditionedPolicy`
        inner_policy (garage.torch.policies.Policy): Policy.
        reward_encoder_class (torch.nn.Module): Reward encoder
        reward_hidden_sizes (list[int]): Output dimension of dense layer(s) of
            the reward encoder.
        encoder_class (garage.torch.embeddings.ContextEncoder): Encoder class
            for the encoder in context-conditioned policy.
        encoder_hidden_sizes (list[int]): Output dimension of dense layer(s) of
            the context encoder.
        latent_dim (int): Size of latent context vector.
        sampler (garage.sampler.Sampler): Sampler.
        test_env_sampler (garage.experiment.SetTaskSampler): Sampler for test
            tasks.
        optimizer_class (type): Type of optimizer for training networks.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int or None): Number of tasks for testing.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        meta_batch_size (int): Meta batch size - #TODO: number of tasks per iteration (?)
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        use_information_bottleneck (bool): False means latent context is
            deterministic.
        use_next_obs_in_context (bool): Whether or not to use next observation
            in distinguishing between tasks.
        replay_buffer_size (int): Maximum samples in replay buffer.
        discount (float): RL discount factor.
        info_coeff (float): info coefficient.
        state_only (bool): Whether or not to use next state instead of action for
            expert trajectories.
        TODO - add rest"""

    def __init__(self,
                 env,
                 experts_dir,
                 policy_class,
                 inner_policy,
                 reward_encoder_class,
                 reward_encoder_hidden_sizes,
                 context_encoder_class,
                 context_encoder_hidden_sizes,
                 latent_dim,
                 sampler,
                 test_env_sampler,
                 optimizer_class,#=torch.optim.Adam,
                 num_train_tasks,
                 num_test_tasks,
                 num_steps_per_epoch,
                 num_initial_steps,
                 meta_batch_size=20,
                 batch_size=12,
                 use_information_bottleneck=True,
                 use_next_obs_in_context=False,
                 replay_buffer_size=1000000,
                 discount=0.99,
                 info_coeff=0.1,
                 imitation_coeff=0.01,
                 airl_itr=15,
                 state_only=False,
                 policy_ent_coeff=0.1,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=True,
                 entropy_method='max',
                 gae_lambda=0.98,
                 center_adv=False,
                 positive_adv=False):

        self._env = env
        self._single_env = env[0]()
        self._experts_dir = experts_dir
        self._sampler = sampler
        self._latent_dim = latent_dim
        self._replay_buffer_size = replay_buffer_size
        self._meta_batch_size = meta_batch_size
        self._batch_size = batch_size
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_train_tasks = num_train_tasks
        self._num_test_tasks = num_test_tasks
        self._num_initial_steps = num_initial_steps
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs_in_context = use_next_obs_in_context
        self._state_only = state_only
        self.airl_itr = airl_itr
        self.gamma = discount
        self.info_coeff = info_coeff
        self.imitation_coeff = imitation_coeff

        ### Environment Specs ###
        # Max Episode Length
        self.T = self._single_env.spec.max_episode_length
        # Observation Space
        self.O = int(np.prod(self._single_env.spec.observation_space.shape))
        # Action Space
        self.A = int(np.prod(self._single_env.spec.action_space.shape))

        self._task_idx = None
        self._is_resuming = False

        ### Entropy Configuration ###
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._policy_ent_coeff = policy_ent_coeff

        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)

        ### For Advantage computation in TRPO ###
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv

        if num_test_tasks is None:
            num_test_tasks = test_env_sampler.n_tasks
        if num_test_tasks is None:
            raise ValueError('num_test_tasks must be provided if '
                             'test_env_sampler.n_tasks is None')

        self.test_env_sampler = test_env_sampler

        ### Initialize Context Encoder ###

        #encoder_spec = self.get_env_spec(self._single_env, latent_dim,
        #                                 'encoder', self.max_episode_length)
        #encoder_in_dim = int(np.prod(encoder_spec.input_space.shape))
        #encoder_out_dim = int(np.prod(encoder_spec.output_space.shape))
        #context_encoder = context_encoder_class(input_dim=encoder_in_dim,
        #                                output_dim=encoder_out_dim,
        #                                hidden_sizes=context_encoder_hidden_sizes)

        obs_dim = int(np.prod(self._single_env.observation_space.shape))
        action_dim = int(np.prod(self._single_env.action_space.shape))
        aug_obs = akro.Box(low=np.tile(np.concatenate((self._single_env.spec.observation_space.low, self._single_env.spec.action_space.low)), self.T),
                           high=np.tile(np.concatenate((self._single_env.spec.observation_space.high, self._single_env.spec.action_space.high)), self.T),
                           dtype=np.float32)
        aug_act = akro.Box(low=np.zeros(latent_dim),
                           high=np.ones(latent_dim),
                           dtype=np.float32)

        encoder_spec = EnvSpec(aug_obs, aug_act)

        self.context_encoder = context_encoder_class(
            env_spec=encoder_spec, hidden_sizes=context_encoder_hidden_sizes, max_std=1e8)
        
        ### Initialize Reward Encoder ###
        self.reward_input_dim = int(np.prod(self._single_env.spec.observation_space.shape)) + self._latent_dim
        self._reward_encoder = reward_encoder_class(
            self.reward_input_dim, 1, reward_encoder_hidden_sizes)

        ### Value Function for defining reward ###
        self._vf = reward_encoder_class(self.reward_input_dim, 1, reward_encoder_hidden_sizes)

        ### Baseline for TRPO ###
        base_spec = self.get_env_spec(self._single_env, latent_dim, 'vf', self.T)
        self._baseline = GaussianMLPValueFunction(env_spec=base_spec,
                                                  hidden_sizes=(32, 32),
                                                  hidden_nonlinearity=torch.tanh,
                                                  output_nonlinearity=None)

        ### Initialize Context-Conditioned Policy ###
        self._policy = policy_class(
            latent_dim=latent_dim,
            context_encoder=self.context_encoder,
            reward_encoder=self._reward_encoder,
            policy=inner_policy,
            use_information_bottleneck=use_information_bottleneck,
            use_next_obs=use_next_obs_in_context)

        ### Initialize Buffers to store transitions ###

        # PathBuffer to store samples obtained from the environment
        self._replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(meta_batch_size)
        }
        # PathBuffer to store expert trajectories
        self._expert_traj_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        #Policy Optimizer for TRPO
        self._policy_optimizer = OptimizerWrapper(
            (ConjugateGradientOptimizer, dict(max_constraint_value=0.01)),
            self._policy.networks[1])

        #Baseline optimizer for TRPO
        self._baseline_optimizer = OptimizerWrapper(
            (torch.optim.Adam, dict(lr=2.5e-4)),
            self._baseline,
            max_optimization_epochs=10,
            minibatch_size=64)

        ### Opimizers for AIRL ###
        central_opt_params = list(self._vf.parameters()) + list(self._reward_encoder.parameters()) +\
        list(self._policy.networks[0].parameters()) + list(self._policy.networks[1].parameters())

        self._central_optimizer = optimizer_class(
            central_opt_params,
            lr=1e-3,
        )

        # self._context_optimizer = optimizer_class(
        #     list(self._policy.networks[0].parameters()),
        #     lr=1e-3,
        # )

        # self._reward_optimizer = optimizer_class(
        #     list(self._reward_encoder.parameters()),
        #     lr=1e-3,
        # )

        # self._airl_policy_optimizer = optimizer_class(
        #     list(self._policy.networks[0].parameters()) + list(self._policy.networks[1].parameters()),
        #     lr=1e-3,
        # )


    def train(self, trainer):

        #Get expert trajectories from directory and store in buffer
        self._obtain_experts(self._experts_dir)

        #Initialize Context-Encoder Parameters with Meta-IL
        #if not self._is_resuming:
        #    logger.log('Pre-Training...')
        #    self._pre_train()

        #Main training loop
        for _ in trainer.step_epochs():

            epoch = trainer.step_itr / self._num_steps_per_epoch

            logger.log('Training...')

            optimizer = torch.optim.Adam(
            list(self._policy.networks[0].parameters()) + list(self._policy.networks[1].parameters()),
            lr=1e-3, #TODO - Experiment with different learning rates
            )

            kl_weight = 0.1

            for _ in range(self._num_steps_per_epoch):

                indices = np.random.choice(range(self._num_train_tasks),
                                        400)

                #Get expert demonstrations for this batch
                #exp_obs, _, exp_acts, _#, _, _ = self._sample_experts(indices)
                exp_obs, _, exp_acts, _ = self._sample_experts(indices)

                exp_obs = exp_obs.view(-1, self.O)
                #exp_acts = exp_acts.contiguous().view(-1, self.A)
                exp_acts = exp_acts.view(-1, self.A)

                exp_traj = torch.cat((exp_obs, exp_acts), dim=-1).view(-1, self.T*(self.O+self.A))
                dist, _ = self._policy.context_forward(exp_traj)
                z = dist.rsample()
                #z = dist.mean
    
                #Probability of observing latent variable in prior distribution
                #log_pz = self.log_normal_pdf(z, torch.zeros(size=dist.mean.size()), torch.zeros(size=dist.variance.size()))
                log_pz = self.log_normal_pdf(z.view(-1, self._latent_dim), torch.zeros(dist.mean.size(), device=global_device()), torch.zeros(dist.variance.size(), device=global_device()))

                #Probability of observing latent variable in posterior distribution
                log_qz = self.log_normal_pdf(z.view(-1, self._latent_dim), dist.mean, dist.variance.log())
                
                #Repeat latent variable across trajectory
                z_tile = z.view(-1, self._latent_dim).unsqueeze(1).repeat(1, self.T, 1).view(-1, self._latent_dim)

                # Calculate Meta-IL loss
                # This is the log-likelihood of choosing expert actions on current policy
                obs_in = torch.cat((exp_obs, z_tile), dim=-1)
                dist, _ = self._policy.inner_forward(obs_in)
                policy_likelihood_loss = -(dist.log_prob(exp_acts.view(-1, self.A))).mean()

                # Calculate losses
                latent_loss = kl_weight*(log_qz - log_pz).mean()
                loss = policy_likelihood_loss + latent_loss

                #Back propagate loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            indices = np.random.choice(range(self._num_train_tasks),
                                    self._meta_batch_size)

            #Sample expert demonstrations
            # data shape is (task, batch, T, feat)
            #exp_obs, exp_next_obs, exp_actions, exp_rewards, exp_success, exp_lengths = self._sample_experts(indices)
            exp_obs, exp_next_obs, exp_actions, exp_rewards = self._sample_experts(indices)
            exp_traj_batch = torch.cat([exp_obs, exp_actions], dim=-1).view(self._meta_batch_size, self.T*(self.O + self.A))

            #For logging only
            returns = torch.sum(exp_rewards, dim=2).mean(dim=1)
            #exp_succ = torch.any(exp_success, dim=2).float().mean(dim=1)
            with tabular.prefix('Expert' + '/'):
                tabular.record('AverageReturn', np.mean(returns.cpu().detach().numpy()))
                tabular.record('MaxReturn', np.max(returns.cpu().detach().numpy()))
                tabular.record('MinReturn', np.min(returns.cpu().detach().numpy()))
                #tabular.record('SuccessRate', np.mean(exp_succ.detach().numpy()))

            #Infer latent variable based on expert trajectories
            self._policy.context_get_actions(exp_traj_batch)

            #Obtain samples from policy, with latent context serving as part of the observation
            for i, idx in enumerate(indices):
                self._replay_buffers[i].clear()
                self._task_idx = i
                self._policy._task_idx = i
                self._obtain_samples(trainer, epoch, self._batch_size)
            
            #Obtain samples from current batch of tasks
            # data shape should be (task, batch, T, feat) where T is max length of an episode
            obs, next_obs, actions, next_actions, actions_log_prob, rewards, successes, lengths = self._sample_data(indices)

            #For logging purposes only
            returns = torch.sum(rewards, dim=2).mean(dim=1)
            successes = successes > 0.5
            succ = torch.any(successes, dim=2).float().mean(dim=1)
            with tabular.prefix('Sample' + '/'):
                tabular.record('AverageReturn', np.mean(returns.cpu().detach().numpy()))
                tabular.record('MaxReturn', np.max(returns.cpu().detach().numpy()))
                tabular.record('MinReturn', np.min(returns.cpu().detach().numpy()))
                tabular.record('SuccessRate', np.mean(succ.cpu().detach().numpy()))


                """
                #Get random tasks for this step - meta_batch_size is number of tasks
                indices = np.random.choice(range(self._num_train_tasks),
                                        self._meta_batch_size)
                #Sample expert demonstrations
                # data shape is (task, batch, T, feat)
                #exp_obs, exp_next_obs, exp_actions, exp_rewards, exp_success, exp_lengths = self._sample_experts(indices)
                exp_obs, exp_next_obs, exp_actions, exp_rewards = self._sample_experts(indices)
                exp_traj_batch = torch.cat([exp_obs, exp_actions], dim=-1).view(self._meta_batch_size, self.T*(self.O + self.A))
                #For logging only
                returns = torch.sum(exp_rewards, dim=2).mean(dim=1)
                #exp_succ = torch.any(exp_success, dim=2).float().mean(dim=1)
                with tabular.prefix('Expert' + '/'):
                    tabular.record('AverageReturn', np.mean(returns.cpu().detach().numpy()))
                    tabular.record('MaxReturn', np.max(returns.cpu().detach().numpy()))
                    tabular.record('MinReturn', np.min(returns.cpu().detach().numpy()))
                    #tabular.record('SuccessRate', np.mean(exp_succ.detach().numpy()))
                #Infer latent variable based on expert trajectories
                self._policy.context_get_actions(exp_traj_batch)
                #Obtain samples from policy, with latent context serving as part of the observation
                for i, idx in enumerate(indices):
                    self._replay_buffers[i].clear()
                    self._task_idx = i
                    self._policy._task_idx = i
                    self._obtain_samples(trainer, epoch, self._batch_size)
                
                #Obtain samples from current batch of tasks
                # data shape should be (task, batch, T, feat) where T is max length of an episode
                obs, next_obs, actions, next_actions, actions_log_prob, rewards, successes, lengths = self._sample_data(indices)
                #For logging purposes only
                returns = torch.sum(rewards, dim=2).mean(dim=1)
                successes = successes > 0.5
                succ = torch.any(successes, dim=2).float().mean(dim=1)
                with tabular.prefix('Sample' + '/'):
                    tabular.record('AverageReturn', np.mean(returns.cpu().detach().numpy()))
                    tabular.record('MaxReturn', np.max(returns.cpu().detach().numpy()))
                    tabular.record('MinReturn', np.min(returns.cpu().detach().numpy()))
                    tabular.record('SuccessRate', np.mean(succ.cpu().detach().numpy()))
                # Probability of observing expert actions based on expert observation
                # and context, under current distribution
                indices = list(range(self._num_train_tasks))
                #exp_obs, exp_next_obs, exp_actions, exp_rewards, exp_success, exp_lengths = self._sample_experts(indices, n=2)
                exp_obs, exp_next_obs, exp_actions, exp_rewards = self._sample_experts(indices, n=2)
                exp_action_log_prob = self._policy.get_probability(exp_obs.view(-1, self.T, self.O), exp_actions.view(-1, self.T, self.A))
                exp_obs = exp_obs.view(-1, self.T, self.O)
                exp_next_obs = exp_next_obs.view(-1, self.T, self.O)
                exp_actions = exp_actions.view(-1, self.T, self.A)
                exp_action_log_prob = exp_action_log_prob.view(-1, self.T, 1)
                #Train AIRL
                self._train_airl(obs, next_obs, actions_log_prob, exp_obs, exp_next_obs, exp_actions, exp_action_log_prob, exp_traj_batch)
                #Eval AIRL, score is the discriminator score
                rewards = self._eval_airl(obs, next_obs, actions, actions_log_prob, exp_traj_batch.view(-1, self.T*(self.O+self.A)))
                ### Optimize Policy with TRPO ###
                ## Data shape is X x N X T
                # X - meta_batch_size, N - batch_size, T - max_episode_length
                #rewards = score
                ## X x N x T x O, O is the observation space
                obs_clean = obs.view(self._meta_batch_size, self._batch_size, self.T, self.O)
                ## X x N x T x A, A is the action space
                actions = actions.view(self._meta_batch_size, self._batch_size, self.T, self.A)
                #Latent variable - X x N x T x Z, where Z is the dimension of the latent variable
                z = self.z.view(self._meta_batch_size, self._batch_size, self.T, self._latent_dim)
                # Policy input - X x N x T x (O+Z)
                obs = torch.cat((obs_clean, z), dim=-1)
                #compute returns for each sample for each task
                # all_rets = []
                # for rews in rewards: #loop over tasks
                #     returns = torch.Tensor(
                #         np.stack([
                #             discount_cumsum(rew, self.gamma) #gamma == discount
                #             for rew in rews.numpy() #loop over batch
                #         ]))
                #     all_rets.append(returns)
                # returns = torch.stack(all_rets)
                returns = torch.Tensor(
                np.stack([
                    discount_cumsum(reward, self.gamma)
                    for reward in rewards.view(-1, self.T).cpu().numpy()
                ]))
                with tabular.prefix('IRL' + '/'):
                    tabular.record('AverageReturn', np.mean(returns.cpu().detach().numpy()))
                    tabular.record('MaxReturn', np.max(returns.cpu().detach().numpy()))
                    tabular.record('MinReturn', np.min(returns.cpu().detach().numpy()))
                
                #Copy old policy for computing kl-divergence
                self._old_policy = copy.deepcopy(self._policy)
                lengths = self.T*torch.ones((self._meta_batch_size, self._batch_size, 1, 1)).int().to(global_device())
                returns = returns.to(global_device())
                #Fit and compute baseline
                baselines = []
                advantages = []
                vf_loss_before = []
                vf_loss_after = []
                for obses, rets, rew, leng in zip(obs.view(self._meta_batch_size, -1, self.O + self._latent_dim),
                               returns.view(self._meta_batch_size, -1, ),
                               rewards.view(self._meta_batch_size, -1, ),
                               lengths.view(self._meta_batch_size, -1, )):
                    with torch.no_grad():
                        vf_loss_before.append(self._baseline.compute_loss(obses, rets))
                    for dataset in self._baseline_optimizer.get_minibatch(
                            obses, rets):
                        self._train_value_function(*dataset)
                    with torch.no_grad():
                        vf_loss_after.append(self._baseline.compute_loss(obses, rets))
                        baseline = self._baseline(obses)
                        baselines.append(baseline)
                    if self._maximum_entropy:
                        policy_entropies = self._compute_policy_entropy(obses.view(-1, self.T, self.O + self._latent_dim))
                        rew += self._policy_ent_coeff * policy_entropies.view(-1, )
                    advantage = self._compute_advantage(rew.view(-1, self.T, ).cpu(), leng.view(-1).cpu(), baseline.view(-1, self.T, ).cpu())
                    advantages.append(advantage)
                baselines = torch.stack(baselines).to(global_device())
                advantages = torch.stack(advantages).to(global_device())
                vf_loss_after = torch.stack(vf_loss_after).mean()
                vf_loss_before = torch.stack(vf_loss_before).mean()
                #print(baselines.size())
                #print(advantages.size())
                advs_flat = advantages.view(self._meta_batch_size, -1, )
                    
                #Compute baseline
                #with torch.no_grad():
                    #baselines = self._baseline(obs_clean.view(-1, self.O))
                #    baselines = self._baseline(obs.view(-1, self.O + self._latent_dim))
                #Add entropy to reward
                # if self._maximum_entropy:
                #     policy_entropies = self._compute_policy_entropy(obs.view(-1, self.T, self.O + self._latent_dim))
                #     rewards += self._policy_ent_coeff * policy_entropies.view(self._meta_batch_size, self._batch_size, self.T)
                #valids = successes.view(-1, self.T, )
                #print(successes.view(-1, self.T, 1))
                #print ((successes.view(-1, self.T, 1)).nonzero(as_tuple=True)[0])
                
                #obs_flat = obs.view(self._meta_batch_size, -1, self.O + self._latent_dim)
                #actions_flat = actions.view(self._meta_batch_size, -1, self.A)
                #returns_flat = returns.view(self._meta_batch_size, -1, )
                #lengths = 100*torch.ones((self._meta_batch_size, self._batch_size, 1, 1)).int()
                #obs_flat = torch.cat(filter_valids(obs.view(-1, self.T, self.O + self._latent_dim), lengths.view(-1, )))
                #actions_flat = torch.cat(filter_valids(actions.view(-1, self.T, self.A), lengths.view(-1, )))
                #returns_flat = torch.cat(filter_valids(returns.view(-1, self.T, ), lengths.view(-1, )))
                # advs_flat = self._compute_advantage(
                #     rewards.view(-1, self.T),
                #     lengths.view(-1, ),
                #     baselines.view(-1, self.T))
                #advs_flat = advs_flat.view(self._meta_batch_size, -1, )
                obs_flat = obs.view(self._meta_batch_size, -1, self.O + self._latent_dim).to(global_device())
                actions_flat = actions.view(self._meta_batch_size, -1, self.A).to(global_device())
                returns_flat = returns.view(self._meta_batch_size, -1, ).to(global_device())
                #Compute losses before
                with torch.no_grad():
                    policy_loss_before = self._compute_loss_with_adv(
                        obs_flat, actions_flat, advs_flat)
                    #vf_loss_before = self._baseline.compute_loss(
                    #    obs_flat, returns_flat)
                    kl_before = self._compute_kl_constraint(obs)
                self._train(obs_clean, obs_flat, actions_flat, returns_flat, advs_flat)
                #Compute losses after
                with torch.no_grad():
                    policy_loss_after = self._compute_loss_with_adv(
                        obs_flat, actions_flat, advs_flat)
                    #vf_loss_after = self._baseline.compute_loss(
                    #    obs_flat, returns_flat)
                    kl_after = self._compute_kl_constraint(obs)
                #Logging
                with tabular.prefix('TRPO Policy' + '/'):
                    tabular.record('LossBefore', policy_loss_before.item())
                    tabular.record('LossAfter', policy_loss_after.item())
                    tabular.record('dLoss',
                                (policy_loss_before - policy_loss_after).item())
                    tabular.record('KLBefore', kl_before.item())
                    tabular.record('KL', kl_after.item())
                with tabular.prefix('TRPO Value Function' + '/'):
                    tabular.record('LossBefore', vf_loss_before.item())
                    tabular.record('LossAfter', vf_loss_after.item())
                    tabular.record('dLoss',
                                vf_loss_before.item() - vf_loss_after.item())
            """    
            trainer.step_itr += 1


    def _pre_train(self, batch_size=400, kl_weight=0.1):

        ### Pre-train context encoder with Meta-IL###

        # Copy initial policy parameters
        policy_state_dict = copy.deepcopy(self._policy.networks[1].state_dict())

        # Initialize Optimizer
        optimizer = torch.optim.Adam(
            list(self._policy.networks[0].parameters()) + list(self._policy.networks[1].parameters()),
            lr=1e-3, #TODO - Experiment with different learning rates
        )

        for i in range(self._num_initial_steps):

            indices = np.random.choice(range(self._num_train_tasks),
                                        batch_size)

            #Get expert demonstrations for this batch
            #exp_obs, _, exp_acts, _#, _, _ = self._sample_experts(indices)
            exp_obs, _, exp_acts, _ = self._sample_experts(indices)

            exp_obs = exp_obs.view(-1, self.O)
            #exp_acts = exp_acts.contiguous().view(-1, self.A)
            exp_acts = exp_acts.view(-1, self.A)

            exp_traj = torch.cat((exp_obs, exp_acts), dim=-1).view(-1, self.T*(self.O+self.A))
            dist, _ = self._policy.context_forward(exp_traj)
            z = dist.rsample()
 
            #Probability of observing latent variable in prior distribution
            #log_pz = self.log_normal_pdf(z, torch.zeros(size=dist.mean.size()), torch.zeros(size=dist.variance.size()))
            log_pz = self.log_normal_pdf(z.view(-1, self._latent_dim), torch.zeros(dist.mean.size(), device=global_device()), torch.zeros(dist.variance.size(), device=global_device()))

            #Probability of observing latent variable in posterior distribution
            log_qz = self.log_normal_pdf(z.view(-1, self._latent_dim), dist.mean, dist.variance.log())
            
            #Repeat latent variable across trajectory
            z_tile = z.view(-1, self._latent_dim).unsqueeze(1).repeat(1, self.T, 1).view(-1, self._latent_dim)

            # Calculate Meta-IL loss
            # This is the log-likelihood of choosing expert actions on current policy
            obs_in = torch.cat((exp_obs, z_tile), dim=-1)
            dist, _ = self._policy.inner_forward(obs_in)
            policy_likelihood_loss = -(dist.log_prob(exp_acts.view(-1, self.A))).mean()

            # Calculate losses
            latent_loss = kl_weight*(log_qz - log_pz).mean()
            loss = policy_likelihood_loss + latent_loss

            #Back propagate loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % 100 == 0):
                print(
                    "Pretrain Epoch", i, "PolicyLikelihood", np.mean(policy_likelihood_loss.cpu().detach().numpy()), "KL",  
                    np.mean(latent_loss.cpu().detach().numpy()), "Total", np.mean(loss.cpu().detach().numpy())
                    )

        #Reset policy parameters to initial parameters
        self._policy.networks[1].load_state_dict(policy_state_dict)


    def _train_airl(self, obs, next_obs, actions_log_prob, 
                    exp_obs, exp_next_obs, exp_actions, exp_action_log_prob, exp_traj_batch):

        for _ in range(self.airl_itr):

            obs_batch, next_obs_batch, actions_log_prob_batch = self.sample_batch(
                obs, next_obs, actions_log_prob, batch_size=self._batch_size)
            exp_obs_batch, exp_next_obs_batch, exp_actions_batch, exp_action_log_prob_batch = self.sample_batch(
                exp_obs, exp_next_obs, exp_actions, exp_action_log_prob,
                batch_size=self._meta_batch_size*self._batch_size)

            self._train_airl_once(
                exp_traj_batch, obs_batch, next_obs_batch, actions_log_prob_batch, exp_obs_batch, \
                    exp_next_obs_batch, exp_actions_batch, exp_action_log_prob_batch)

        return


    def _train_airl_once(self, exp_traj_batch, obs, next_obs, actions_log_prob, 
                         exp_obs, exp_next_obs, exp_actions, exp_action_log_prob):

        #Repeat across batch dimension
        exp_traj_batch_tile = (exp_traj_batch.view(self._meta_batch_size, 1, self.T, self.O + self.A)).repeat(1, self._batch_size, 1, 1)

        #-1 = batch_size
        obs = obs.view(self._meta_batch_size, -1, self.T, self.O)
        next_obs = next_obs.view(self._meta_batch_size, -1, self.T, self.O)
        exp_obs = exp_obs.view(self._meta_batch_size, -1, self.T, self.O)
        exp_next_obs = exp_next_obs.view(self._meta_batch_size, -1, self.T, self.O)
        exp_actions = exp_actions.view(self._meta_batch_size, -1, self.T, self.A)
        actions_log_prob = actions_log_prob.view(self._meta_batch_size, -1, self.T, 1)
        exp_action_log_prob = exp_action_log_prob.view(self._meta_batch_size, -1, self.T, 1)

        #Concat data and experts on the batch dimension
        #Shape of this is then (tasks, 2*batch_size, T, -1)
        obs_t = torch.cat([obs, exp_obs], dim=1)
        next_obs_t = torch.cat([next_obs, exp_next_obs], dim=1)
        lprobs = torch.cat([actions_log_prob, exp_action_log_prob], dim=1)
        expert_traj_batch_input = torch.cat([exp_traj_batch_tile, torch.cat([exp_obs, exp_actions], axis=-1)\
            .view(self._meta_batch_size, self._batch_size, self.T, -1)], axis=1)

        #Labels tell if trajectory comes from experts (1) or sampled data from policy (0)
        labels = torch.zeros((self._meta_batch_size, self._batch_size*2, 1, 1), device=global_device())
        labels[:, self._batch_size:, ...] = 1.0

        #Infer latent variable
        dist, _ = self._policy.context_forward(expert_traj_batch_input.view(-1, self.T*(self.O + self.A)))
        z = dist.rsample()
        #Log-Probability of observing latent variable under current distribution
        z_probs = dist.log_prob(z)
        #Repeat latent variable across trajectory dimension
        z = z.view(self._meta_batch_size, self._batch_size*2, self._latent_dim).unsqueeze(2).repeat(1, 1, self.T, 1)

        #Infer Reward
        z = z.view(-1, self._latent_dim)
        obs_in = torch.cat((obs_t.view(-1, self.O), z), dim=1)
        reward = self._reward_encoder(obs_in)
        #Sum over trajectories
        sampled_traj_return = torch.sum(reward.view(self._meta_batch_size, -1, self.T), axis=-1, keepdims=True)

        #Value function
        npotential_input = torch.cat((next_obs_t.view(-1, self.O), z), dim=-1)
        potential_input = torch.cat((obs_t.view(-1, self.O), z), dim=-1)

        fitted_value_fn_n = self._vf(npotential_input)
        fitted_value_fn = self._vf(potential_input)

        ### Define losses ###

        #Policy likelihood loss
        z = z.view(self._meta_batch_size, -1, self.T, self._latent_dim)
        imitation_z = z[:, 0, :, :].contiguous().view(-1, self._latent_dim)
        imitation_obses = exp_traj_batch.view(self._meta_batch_size, 1, self.T, -1)[..., :self.O]
        imitation_acts = exp_traj_batch.view(self._meta_batch_size, 1, self.T, -1)[..., self.O:]
        imitation_input = torch.cat((imitation_obses.view(-1, self.O), imitation_z), dim=-1)
        dist, _ = self._policy.inner_forward(imitation_input)
        policy_likelihood_loss = -(dist.log_prob(imitation_acts.view(-1, self.A))).mean()

        log_p_tau = (reward  + fitted_value_fn_n - fitted_value_fn).view(self._meta_batch_size, -1, self.T, 1)
        log_q_tau = lprobs

        #Discriminator loss
        log_pq = torch.stack((log_p_tau, log_q_tau)).logsumexp(axis=0)
        #discrim_output = torch.exp(log_p_tau - log_pq)
        #print(discrim_output)
        cent_loss = -(labels*(log_p_tau-log_pq) + (1-labels)*(log_q_tau-log_pq)).mean()

        #Compute mutual information-loss
        log_q_m_tau = z_probs.view(self._meta_batch_size, -1, 1)
        squeezed_labels = torch.squeeze(labels, axis=-1)
        # Used for computing gradient w.r.t. psi
        info_loss = -(log_q_m_tau * (1 - squeezed_labels)).mean() / (1 - labels).mean()
        # Used for computing the gradient w.r.t. theta
        a = (1 - squeezed_labels) * log_q_m_tau.detach()
        b = (1-labels).mean()
        info_surr_loss = -torch.mean(
            a * sampled_traj_return -
            a * torch.mean(sampled_traj_return*(1-squeezed_labels), axis=1, keepdims=True) /
            b) / b

        ### Calculate final losses ###

        policy_loss = self.imitation_coeff*policy_likelihood_loss
        context_loss = self.info_coeff*info_loss
        reward_loss = self.info_coeff*info_surr_loss
        loss = cent_loss + policy_loss + context_loss + reward_loss

        ### Backpropagation ###

        self._central_optimizer.zero_grad()
        #self._context_optimizer.zero_grad()
        #self._airl_policy_optimizer.zero_grad()
        #self._reward_optimizer.zero_grad()

        #context_loss.backward(retain_graph=True)
        #reward_loss.backward(retain_graph=True)
        #policy_loss.backward(retain_graph=True)
        loss.backward()

        #self._context_optimizer.step()
        #self._reward_optimizer.step()
        #self._airl_policy_optimizer.step()
        self._central_optimizer.step()


    def _eval_airl(self, obs, next_obs, actions, actions_log_prob, exp_traj):

        with torch.no_grad():

            #Repeat expert trajectory across batch dimension
            exp_traj = exp_traj.unsqueeze(1).repeat(1, actions.size(1), 1)
            #Sample latent variable from current distribution
            dist, _ = self._policy.context_forward(exp_traj.view(-1, self.T*(self.O + self.A)))
            z = dist.rsample()
            #Log-Likelihood of observing current sampled latent variable
            z_probs = dist.log_prob(z)
            #Repeat latent variable across trajectory dimension
            self.z = z.view(self._meta_batch_size, self._batch_size, self._latent_dim).unsqueeze(2).repeat(1, 1, self.T, 1)

            #All trajectories are from sampled data
            labels = torch.zeros((self._meta_batch_size, self._batch_size, 1, 1), device=global_device())

            obs_in = torch.cat([obs.view(self._meta_batch_size, self._batch_size, self.T, self.O), self.z], dim=-1)
            nobs_in = torch.cat([next_obs.view(self._meta_batch_size, self._batch_size, self.T, self.O), self.z], dim=-1)

            #Infer reward
            reward = self._reward_encoder(obs_in.view(-1, self.reward_input_dim))
            sampled_traj_return = torch.sum(reward.view(self._meta_batch_size, -1, self.T), axis=-1, keepdims=True)

            #Value function potential function
            fitted_value_fn_n = self._vf(nobs_in.view(-1, self.reward_input_dim))
            fitted_value_fn = self._vf(obs_in.view(-1, self.reward_input_dim))

            #Define losses
            log_p_tau = (reward  + fitted_value_fn_n - fitted_value_fn).view(self._meta_batch_size, -1, self.T, 1)
            log_q_tau = actions_log_prob

            #Discriminator loss
            log_pq = torch.logsumexp(torch.stack((log_p_tau, log_q_tau)), axis=0)
            discrim_output = torch.exp(log_p_tau - log_pq)
            cent_loss = -(labels*(log_p_tau-log_pq) + (1-labels)*(log_q_tau-log_pq)).mean()

            log_q_m_tau = z_probs.view(self._meta_batch_size, -1, 1)
            squeezed_labels = torch.squeeze(labels, axis=-1)
            # Used for computing gradient w.r.t. psi
            info_loss = -(log_q_m_tau * (1 - squeezed_labels)).mean() / (1 - labels).mean()

            info_surr_loss = - torch.mean(
                (1 - squeezed_labels) * log_q_m_tau.detach() * sampled_traj_return -
                (1 - squeezed_labels) * log_q_m_tau.detach() * torch.mean(sampled_traj_return*(1-squeezed_labels), axis=1, keepdims=True) /
                (1-labels).mean()) / (1-labels).mean()

            loss = cent_loss + self.info_coeff*info_loss

            imitation_z = self.z.view(self._meta_batch_size, self._batch_size, self.T, -1)[:,0,...].contiguous().view(-1, self._latent_dim)
            imitation_expert = exp_traj.view(self._meta_batch_size, -1, self.T, self.O + self.A)
            imitation_obses = imitation_expert[:,0,:,:self.O].contiguous()
            imitation_acts = imitation_expert[:,0,:,self.O:].contiguous()
            imitation_input = torch.cat((imitation_obses.view(-1, self.O), imitation_z), dim=-1)
            dist, _ = self._policy.inner_forward(imitation_input)
            policy_likelihood_loss = -(dist.log_prob(imitation_acts.view(-1, self.A))).mean()

            prefix = "AIRL"
            with tabular.prefix(prefix + '/'):
                tabular.record('AverageDiscrim-Output', np.mean(discrim_output.cpu().detach().numpy()))
                tabular.record('AverageCentral-Loss', np.mean(cent_loss.cpu().detach().numpy()))
                tabular.record('AverageInfo-Loss', np.mean(info_loss.cpu().detach().numpy()))
                tabular.record('AverageInfo-Surrogate-Loss', np.mean(info_surr_loss.cpu().detach().numpy()))
                tabular.record('AveragePolicy-Loss', np.mean(policy_likelihood_loss.cpu().detach().numpy()))

        return reward.view(self._meta_batch_size, self._batch_size, self.T, 1)


    def _train(self, obs_clean, obs, actions, returns, advs):
        r"""Train the policy and value function with minibatch.
        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.
        """
        self._train_policy(obs.view(self._meta_batch_size, -1, self.O + self._latent_dim).cpu(),
                           actions.view(self._meta_batch_size, -1, self.A).cpu(),
                           advs.view(self._meta_batch_size, -1, 1).cpu())
        # for _ in range(5):
        #     obs_batch, actions_batch, rewards_batch, advs_batch = self.sample_batch(
        #         obs.view(self._meta_batch_size, -1, self.T, self.O+self._latent_dim),
        #         actions.view(self._meta_batch_size, -1, self.T, self.A),
        #         rewards.view(self._meta_batch_size, -1, self.T, 1),
        #         advs.view(self._meta_batch_size, -1, self.T, 1), batch_size=64)
        #     self._train_policy(
        #         obs_batch.view(self._meta_batch_size, -1, self.O + self._latent_dim),
        #         actions_batch.view(self._meta_batch_size, -1, self.A),
        #         rewards_batch.view(self._meta_batch_size, -1, ),
        #         advs_batch.view(self._meta_batch_size, -1, ))
        #self._train_value_function(obs, returns)
        #for dataset in self._policy_optimizer.get_minibatch(
        #        obs, actions, rewards, advs):
        #    self._train_policy(*dataset)
        # for obses, rets in zip(obs.view(self._meta_batch_size, -1, self.O + self._latent_dim),
        #                        returns.view(self._meta_batch_size, -1, )):
        #     for dataset in self._baseline_optimizer.get_minibatch(
        #             obses, rets):
        #         self._train_value_function(*dataset)
        #self._train_value_function(obs.view(self._meta_batch_size, -1, self.O + self._latent_dim), returns.view(self._meta_batch_size, -1, ))


    def _train_policy(self, obs, actions, advantages):
        r"""Train the policy.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated mean scalar value of policy loss (float).
        """
        self._policy_optimizer.zero_grad()
        loss = self._compute_loss_with_adv(obs.cpu(), actions.cpu(), advantages.cpu())
        loss.backward()
        self._policy_optimizer.step(
            f_loss=lambda: self._compute_loss_with_adv(obs.cpu(), actions.cpu(), advantages.cpu()),
            f_constraint=lambda: self._compute_kl_constraint(obs.cpu()))

        return loss


    def _train_value_function(self, obs, returns):
        r"""Train the value function.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            returns (torch.Tensor): Acquired returns
                with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated mean scalar value of value function loss
                (float).
        """

        self._baseline_optimizer.zero_grad()
        #losses = []
        #for o, r in zip(obs, returns):
        #   loss = self._baseline.compute_loss(o, r)
        #   losses.append(loss)
        loss = self._baseline.compute_loss(obs, returns)
        #loss = torch.stack(losses).mean()
        loss.backward()
        self._baseline_optimizer.step()

        return


    def _compute_policy_entropy(self, obs):

        r"""Compute entropy value of probability distribution.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
        Returns:
            torch.Tensor: Calculated entropy values given observation
                with shape :math:`(N, P)`.
        """

        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = self._policy.inner_forward(obs)[0].entropy()
        else:
            policy_entropy = self._policy.inner_forward(obs)[0].entropy()

        # This prevents entropy from becoming negative for small policy std
        #if self._use_softplus_entropy:
        #    policy_entropy = F.softplus(policy_entropy)

        return policy_entropy


    def _compute_advantage(self, rewards, valids, baselines):
        r"""Compute advantages with Generalized Advantage Estimation (GAE).
        Notes: T is the maximum episode length
        Args:
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, T)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, T)`.
        Returns:
            torch.Tensor: Calculated advantage values given rewards and
                baselines with shape :math:`(N \dot [T], )`.
        """
        advantages = compute_advantages(self.gamma, self._gae_lambda,
                                        self.T, baselines,
                                        rewards)

        advantage_flat = torch.cat(filter_valids(advantages, valids))

        if self._center_adv:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat


    def _compute_loss_with_adv(self, obs, actions, advantages):
        r"""Compute mean value of loss.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(X, N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(X, N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(X, N \dot [T], )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(X, N \dot [T], )`.
        Returns:
            torch.Tensor: Calculated negative mean scalar value of objective.
        """

        objectives = []
        #Loop over meta_batch_size (the different tasks)
        for obses, acts, advs in zip(obs, actions, advantages):
            objective = self._compute_objective(advs, obses, acts)
            if self._entropy_regularzied:
                policy_entropies = self._compute_policy_entropy(obses)
                objective += self._policy_ent_coeff * policy_entropies
            objectives.append(-objective.mean())
        
        objectives = torch.stack(objectives)

        #if self._entropy_regularzied:
        #    policy_entropies = self._compute_policy_entropy(obs.view(-1, self.T, self.O + self._latent_dim))
        #    objectives += self._policy_ent_coeff * policy_entropies

        return objectives.mean().cpu()

        # objectives = self._compute_objective(
        #     advantages.view(-1, ),
        #     obs.view(-1, self.O + self._latent_dim),
        #     actions.view(-1, self.A))

        # if self._entropy_regularzied:
        #     policy_entropies = self._compute_policy_entropy(obs)
        #     objectives += self._policy_ent_coeff * policy_entropies

        # return -objectives.mean()


    def _compute_kl_constraint(self, obs):
        r"""Compute KL divergence.
        Compute the KL divergence between the old policy distribution and
        current policy distribution.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
        Returns:
            torch.Tensor: Calculated mean scalar value of KL divergence
                (float).
        """

        kl_constraints = []

        for obses in obs.to(global_device()):
            with torch.no_grad():
                old_dist = self._old_policy.inner_forward(obses)[0]

            new_dist = self._policy.inner_forward(obses)[0]

            kl_constraint = torch.distributions.kl.kl_divergence(
                old_dist, new_dist)

            kl_constraints.append(kl_constraint.mean())
        
        kl_constraints = torch.stack(kl_constraints)

        # with torch.no_grad():
        #     old_dist = self._old_policy.inner_forward(obs.view(-1, self.O + self._latent_dim))[0]

        # new_dist = self._policy.inner_forward(obs.view(-1, self.O + self._latent_dim))[0]

        # kl_constraint = torch.distributions.kl.kl_divergence(
        #     old_dist, new_dist)

        # return kl_constraint.mean()

        return kl_constraints.mean().cpu()


    def _compute_objective(self, advantages, obs, actions):
        r"""Compute TRPO objective value.
        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.
        """

        obs = obs.to(global_device())
        actions = actions.to(global_device())
        advantages = advantages.to(global_device())

        with torch.no_grad():
            old_ll = self._old_policy.inner_forward(obs)[0].log_prob(actions)

        new_ll = self._policy.inner_forward(obs)[0].log_prob(actions)
        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        return surrogate.cpu()


    def _obtain_samples(self,
                        trainer,
                        itr,
                        num_samples):

        """Obtain samples.
        Args:
            trainer (Trainer): Trainer.
            itr (int): Index of iteration (epoch).
            num_samples (int): Number of samples to obtain.
        """

        total_samples = 0
        while total_samples < num_samples:
            paths = trainer.obtain_samples(itr, num_samples,
                                           self._policy,
                                           self._env[self._task_idx])
            total_samples += sum([len(path['rewards']) for path in paths])

            for path in paths:

                p = {
                    'observations':
                    path['observations'],
                    'actions':
                    path['actions'],
                    'actions_log_prob':
                    path['agent_infos']['log_prob'].reshape(-1, 1),
                    'rewards':
                    path['rewards'].reshape(-1, 1),
                    'success':
                    path['env_infos']['success'].reshape(-1, 1),
                    'next_observations':
                    path['next_observations'],
                    'dones':
                    np.array([
                        step_type == StepType.TERMINAL
                        for step_type in path['step_types']
                    ]).reshape(-1, 1)
                }
                self._replay_buffers[self._task_idx].add_path(p)

        return


    def _obtain_experts(self, experts_dir):

        """ Get expert trajectories and store in buffer
        Args:
            experts_dir (String) - Directory with experts trajectories
        """

        with open(experts_dir, 'rb') as f:
            tasks = pickle.load(f)

        for i in range(self._num_train_tasks):
            good = False
            for j, path in enumerate(tasks[i]):
                good = True
                p = {
                    'observations':
                    np.array(path['observations']),
                    'actions':
                    np.array(path['actions']),
                    'rewards':
                    np.array(path['rewards']).reshape(-1, 1)}
                    #'success':
                    #np.array(path['success']).reshape(-1, 1)}
                
                self._expert_traj_buffers[i].add_path(p)
            if not good:
                path = tasks[0][0]
                p = {
                    'observations':
                    np.array(path['observations']),
                    'actions':
                    np.array(path['actions']),
                    'rewards':
                    np.array(path['rewards']).reshape(-1, 1)}
                    #'success':
                    #np.array(path['success']).reshape(-1, 1)}
                
                self._expert_traj_buffers[i].add_path(p)

        return


    def _sample_data(self, indices):

        """Sample batch of training data from a list of tasks.
        Args:
            indices (list): List of task indices to sample from.
        Returns:
            torch.Tensor: Obervations, with shape :math:`(X, N, T, O^*)` where X
                is the number of tasks. N is batch size. T is max episode length.
            torch.Tensor: Next obervations, with shape :math:`(X, N, T, O^*)`.
            torch.Tensor: Actions, with shape :math:`(X, N, T, A^*)`.
            torch.Tensor: Next actions, with shape :math:`(X, N, T, A^*)`.
            torch.Tensor: Actions log prob, with shape :math:`(X, N, T, A^*)`.
            torch.Tensor: Rewards, with shape :math:`(X, N, T, 1)`.
            torch.Tensor: Dones, with shape :math:`(X, N, T, 1)`.
        """

        initialized = False
        for idx in range(self._meta_batch_size):
            batch_initialized = False
            for _ in range(self._batch_size):
                path = self._replay_buffers[idx].sample_path()
                no1 = np.roll(path['observations'], -1)
                no1[:, -1] = 0.0
                path['next_observation'] = no1
                na1 = np.roll(path['actions'], -1)
                na1[:, -1] = 0.0
                path['next_action'] = na1

                good = (path['success'].reshape(-1, ).nonzero()[0])
                if good.shape[0] == 0:
                    length = self.T
                else:
                    length = good[0]
                path['length'] = np.expand_dims(np.array([length]), axis=0)

                if not batch_initialized:
                    o = path['observations'][np.newaxis]
                    a = path['actions'][np.newaxis]
                    no = path['next_observations'][np.newaxis]
                    na = path['next_action'][np.newaxis]
                    log_pi = path['actions_log_prob'][np.newaxis]
                    r = path['rewards'][np.newaxis]
                    s = path['success'][np.newaxis]
                    l = path['length'][np.newaxis]
                    batch_initialized = True
                else:
                    o = np.vstack((o, path['observations'][np.newaxis]))
                    no = np.vstack((no, path['next_observations'][np.newaxis]))
                    a = np.vstack((a, path['actions'][np.newaxis]))
                    na = np.vstack((na, path['next_action'][np.newaxis]))
                    log_pi = np.vstack((log_pi, path['actions_log_prob'][np.newaxis]))
                    r = np.vstack((r, path['rewards'][np.newaxis]))
                    s = np.vstack((s, path['success'][np.newaxis]))
                    l = np.vstack((l, path['length'][np.newaxis]))

            if not initialized:
                all_o = o[np.newaxis]
                all_no = no[np.newaxis]
                all_a = a[np.newaxis]
                all_na = na[np.newaxis]
                all_log_pi = log_pi[np.newaxis]
                all_r = r[np.newaxis]
                all_s = s[np.newaxis]
                all_l = l[np.newaxis]
                initialized = True
            else:
                all_o = np.vstack((all_o, o[np.newaxis]))
                all_no = np.vstack((all_no, no[np.newaxis]))
                all_a = np.vstack((all_a, a[np.newaxis]))
                all_na = np.vstack((all_na, na[np.newaxis]))
                all_log_pi = np.vstack((all_log_pi, log_pi[np.newaxis]))
                all_r = np.vstack((all_r, r[np.newaxis]))
                all_s = np.vstack((all_s, s[np.newaxis]))
                all_l = np.vstack((all_l, l[np.newaxis]))


        observations = torch.as_tensor(all_o, device=global_device()).float()
        next_observations = torch.as_tensor(all_no, device=global_device()).float()
        actions = torch.as_tensor(all_a, device=global_device()).float()
        next_actions = torch.as_tensor(all_na, device=global_device()).float()
        actions_log_prob = torch.as_tensor(all_log_pi, device=global_device()).float()
        rewards = torch.as_tensor(all_r, device=global_device()).float()
        successes = torch.as_tensor(all_s, device=global_device()).float()
        lengths = torch.as_tensor(all_l, device=global_device()).int()

        return observations, next_observations, actions, next_actions, actions_log_prob, rewards, successes, lengths


    def _sample_experts(self, indices, n=1):

        """Sample batch of expert trajectories from a list of tasks.
        Args:
            indices (list): List of task indices to sample from.
        Returns:
            torch.Tensor: Obervations, with shape :math:`(X, N, T, O^*)` where X
                is the number of tasks. N is batch size. T is episode length
            torch.Tensor: Next obervations, with shape :math:`(X, N, T, O^*)`.
            torch.Tensor: Actions, with shape :math:`(X, N, T, A^*)`.
            torch.Tensor: Next actions, with shape :math:`(X, N, T, A^*)`.
        """

        initialized = False
        num_tasks = len(indices)
        for idx in indices:
            batch_initialized = False
            for _ in range(n):
                path = self._expert_traj_buffers[idx].sample_path()

                no1 = np.roll(path['observations'], -1)
                no1[:, -1] = 0.0
                path['next_observation'] = no1

                # good = (path['success'].reshape(-1, ).nonzero()[0])
                # if good.shape[0] == 0:
                #     length = self.T
                # else:
                #     length = good[0]
                # path['length'] = np.expand_dims(np.array([length]), axis=0)

                if not batch_initialized:
                    o = path['observations'][np.newaxis]
                    a = path['actions'][np.newaxis]
                    r = path['rewards'][np.newaxis]
                    #s = path['success'][np.newaxis]
                    #l = path['length'][np.newaxis]
                    no = path['next_observation'][np.newaxis]
                    batch_initialized = True
                else:
                    o = np.vstack((o, path['observations'][np.newaxis]))
                    no = np.vstack((no, path['next_observation'][np.newaxis]))
                    a = np.vstack((a, path['actions'][np.newaxis]))
                    r = np.vstack((r, path['rewards'][np.newaxis]))
                    #s = np.vstack((s, path['success'][np.newaxis]))
                    #l = np.vstack((l, path['length'][np.newaxis]))

            if not initialized:
                all_o = o[np.newaxis]
                all_no = no[np.newaxis]
                all_a = a[np.newaxis]
                all_r = r[np.newaxis]
                #all_s = s[np.newaxis]
                #all_l = l[np.newaxis]
                initialized = True
            else:
                all_o = np.vstack((all_o, o[np.newaxis]))
                all_no = np.vstack((all_no, no[np.newaxis]))
                all_a = np.vstack((all_a, a[np.newaxis]))
                all_r = np.vstack((all_r, r[np.newaxis]))
                #all_s = np.vstack((all_s, s[np.newaxis]))
                #all_l = np.vstack((all_l, l[np.newaxis]))

        observations = torch.as_tensor(
            all_o, device=global_device()).float().view(num_tasks, -1, self.T, self.O)
        next_observations = torch.as_tensor(
            all_no, device=global_device()).float().view(num_tasks, -1, self.T, self.O)
        actions = torch.as_tensor(
            all_a, device=global_device()).float().view(num_tasks, -1, self.T, self.A)
        rewards = torch.as_tensor(
            all_r, device=global_device()).float().view(num_tasks, -1, self.T, 1)
        #success = torch.as_tensor(
        #    all_s, device=global_device()).view(num_tasks, -1, self.T, 1)
        #lengths = torch.as_tensor(
        #    all_l, device=global_device()).int().view(num_tasks, -1, 1)

        return observations, next_observations, actions, rewards#, success, lengths

    @classmethod
    def get_env_spec(cls, env_spec, latent_dim, module, max_episode_length):

        """Get environment specs of encoder with latent dimension.
        Args:
            env_spec (EnvSpec): Environment specification.
            latent_dim (int): Latent dimension.
            module (str): Module to get environment specs for.
        Returns:
            InOutSpec: Module environment specs with latent dimension.
        """

        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        if module == 'encoder':
            in_dim = max_episode_length*(obs_dim + action_dim)
            out_dim = latent_dim * 2
            in_space = akro.Box(low=-1, high=1, shape=(in_dim, ), dtype=np.float32)
            out_space = akro.Box(low=-1,
                                 high=1,
                                 shape=(out_dim, ),
                                 dtype=np.float32)
        elif module == 'vf':
            out_dim = 1
            in_space = akro.Box(low=np.concatenate((env_spec.observation_space.low, np.zeros(latent_dim))), high=np.concatenate((env_spec.observation_space.high, np.ones(latent_dim))), dtype=np.float64)
            #in_space = akro.Box(low=env_spec.observation_space.low, high=env_spec.observation_space.high, dtype=np.float32)
            out_space = akro.Box(low=-np.inf,
                                 high=np.inf,
                                 shape=(out_dim, ),
                                 dtype=np.float64)
        if module == 'encoder':
            spec = InOutSpec(in_space, out_space)
        elif module == 'vf':
            spec = EnvSpec(in_space, out_space)

        return spec

    @classmethod
    def augment_env_spec(cls, env_spec, latent_dim):

        """Augment environment by a size of latent dimension.
        Args:
            env_spec (EnvSpec): Environment specs to be augmented.
            latent_dim (int): Latent dimension.
        Returns:
            EnvSpec: Augmented environment specs.
        """

        aug_obs = akro.Box(low=np.concatenate((env_spec.observation_space.low, np.zeros(latent_dim))),
                           high=np.concatenate((env_spec.observation_space.high, np.ones(latent_dim))),
                           dtype=np.float32)
        aug_act = akro.Box(low=env_spec.action_space.low,
                           high=env_spec.action_space.high,
                           dtype=np.float32)
        return EnvSpec(aug_obs, aug_act)

    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')


    def sample_batch(self, *args, batch_size):
        # sample a batch for each task
        if len(args[0].size()) > 3:
            N = args[0].size(1)
        else:
            N = args[0].size(0)
        batch_idxs = np.random.randint(0, N, batch_size)  # trajectories are negatives
        if len(args[0].size()) > 3:
            return [data[:, batch_idxs, ...] for data in args]
        return [data[batch_idxs] for data in args]


    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        return torch.sum(
            -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + np.log(2.*np.pi)), 
            axis=raxis)


    def __getstate__(self):
        """Object.__getstate__.
        Returns:
            dict: the state to be pickled for the instance.
        """
        data = self.__dict__.copy()
        del data['_replay_buffers']
        del data['_expert_traj_buffers']
        return data


    def __setstate__(self, state):
        """Object.__setstate__.
        Args:
            state (dict): unpickled state.
        """
        self.__dict__.update(state)
        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }

        self._expert_traj_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }

        self._is_resuming = True

    @property
    def networks(self):
        """Return all the networks within the model.
        Returns:
            list: A list of networks.
        """
        return self._policy.networks + [self._vf] + [self._baseline]


    def to(self, device=None):
        """Put all the networks within the model on device.
        Args:
            device (str): ID of GPU or CPU.
        """
        device = device or global_device()
        for net in self.networks:
            net.to(device)

        self.networks[0]._module.to(device)


class PEMIRLWorker(DefaultWorker):
    """A worker class used in sampling for PEMIRL.
    It stores context and resamples belief in the policy every step.
    Args:
        seed (int): The seed to use to intialize random number generators.
        max_episode_length(int or float): The maximum length of episodes which
            will be sampled. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.
    """

    def __init__(self,
                 *,
                 seed,
                 max_episode_length,
                 worker_number):
        self._episode_info = None
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def start_episode(self):
        """Begin a new episode."""
        self._eps_length = 0
        self._prev_obs, self._episode_info = self.env.reset()

    def step_episode(self):
        """Take a single time-step in the current episode.
        Returns:
            bool: True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.
        """
        if self._eps_length < self._max_episode_length:
            #Get action
            a, agent_info = self.agent.get_action(self._prev_obs)
            a = a.squeeze()
            es = self.env.step(a)
            #Store observations
            self._observations.append(self._prev_obs)
            self._env_steps.append(es)

            #Store mean and variance of distribution
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if not es.last:
                self._prev_obs = es.observation
                return False

        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)

        return True

    def rollout(self):
        """Sample a single episode of the agent in the environment.
        Returns:
            EpisodeBatch: The collected episode.
        """
        #self.agent.sample_from_belief()
        self.start_episode()
        while not self.step_episode():
            pass
        return self.collect_episode()

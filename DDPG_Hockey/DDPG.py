import torch
import torch.nn as nn
# from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle
import hockey.hockey_env as h_env
import random  # for RND

# from pink import PinkActionNoise
# install with pip install pink-noise-rl
# from https://github.com/martius-lab/pink-noise-rl
# but not sure how to use it here

import memory as mem
from feedforward import Feedforward
# from . import memory as mem
# from .feedforward import Feedforward

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)


class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100, 100],
                 learning_rate=0.0002):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate,
                                          eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets):  # all arguments should be torch tensors
        self.train()  # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass

        pred = self.Q_value(observations, actions)
        # Compute Loss
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(torch.hstack([observations, actions]))


class OUNoise():
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (- self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)


# code works but we have to check if it makes sense and if it improves something
class ColoredNoise:
    def __init__(self, shape, beta: float = 1.0, dt: float = 1e-2):
        """
        Frequency-domain noise generator.

        Args:
            shape (tuple): Shape of the noise sequence.
            beta (float): Spectral slope parameter.
                          beta = 0 --> white noise
                          beta = 1 --> pink noise
                          beta = 2 --> OU noise
            dt (float): Time step for scaling.
        """
        self._shape = shape
        self._beta = beta
        self._dt = dt
        self.reset()

    def __call__(self) -> np.ndarray:
        """
        Generate a single step of noise.
        """
        # Generate frequency components
        freqs = np.fft.rfftfreq(self._shape, d=self._dt)  # Frequencies for Fourier transform
        # Scale by f^(-β/2), avoiding divide-by-zero

        scale = np.zeros_like(freqs)  # Initialize scale with zeros
        nonzero_freqs = freqs > 0  # Identify nonzero frequencies
        scale[nonzero_freqs] = freqs[nonzero_freqs] ** (-self._beta / 2)

        real_part = np.random.normal(0, 1, len(freqs)) * scale  # Real part
        imag_part = np.random.normal(0, 1, len(freqs)) * scale  # Imaginary part

        # Combine into a complex array for inverse FFT
        spectrum = real_part + 1j * imag_part

        # Transform back to time domain
        noise = np.fft.irfft(spectrum, n=self._shape)
        self.noise_prev = noise  # Store last noise value (optional, for reset compatibility)
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)


# RND Module
class RNDModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[64, 64]):
        super().__init__()
        self.predictor = Feedforward(input_dim, hidden_sizes, output_dim)
        self.target = Feedforward(input_dim, hidden_sizes, output_dim)
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, state):
        target_output = self.target(state)
        predicted_output = self.predictor(state)
        exploration_bonus = ((target_output - predicted_output) ** 2).mean()
        return exploration_bonus


# Normalization class for RND
class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        new_var = (self.count * self.var + batch_count * batch_var +
                   delta**2 * self.count * batch_count / total_count) / total_count

        self.mean, self.var, self.count = new_mean, new_var, total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class TD3():
    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = self._action_space.shape[0]
        self._config = {
            "eps": 0.1,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128, 128],
            "hidden_sizes_critic": [128, 128, 64],
            "update_target_every": 100,
            "use_target_net": True,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "colNoise": False
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        self._colNoise = self._config['colNoise']

        # pink noise
        if self._colNoise:
            self.action_noise = ColoredNoise((self._action_n))
        # OU Noise (default)
        else:
            self.action_noise = OUNoise((self._action_n))

        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Q Networks
        self.Q1 = QFunction(observation_dim=self._obs_dim,
                            action_dim=self._action_n,
                            hidden_sizes=self._config["hidden_sizes_critic"],
                            learning_rate=self._config["learning_rate_critic"])
        self.Q2 = QFunction(observation_dim=self._obs_dim,
                            action_dim=self._action_n,
                            hidden_sizes=self._config["hidden_sizes_critic"],
                            learning_rate=self._config["learning_rate_critic"])

        # Target Q Networks
        self.Q1_target = QFunction(observation_dim=self._obs_dim,
                                   action_dim=self._action_n,
                                   hidden_sizes=self._config["hidden_sizes_critic"],
                                   learning_rate=0)
        self.Q2_target = QFunction(observation_dim=self._obs_dim,
                                   action_dim=self._action_n,
                                   hidden_sizes=self._config["hidden_sizes_critic"],
                                   learning_rate=0)

        # Policy Network
        self.policy = Feedforward(input_size=self._obs_dim,
                                  hidden_sizes=self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun=torch.nn.ReLU(),
                                  output_activation=torch.nn.Tanh())
        self.policy_target = Feedforward(input_size=self._obs_dim,
                                         hidden_sizes=self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun=torch.nn.ReLU(),
                                         output_activation=torch.nn.Tanh())

        self._copy_nets()

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self._config["learning_rate_actor"],
                                          eps=0.000001)
        self.train_iter = 0

    def _copy_nets(self):
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        action = self.policy.predict(observation) + eps * self.action_noise()
        action = np.clip(action, self._action_space.low, self._action_space.high)
        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.Q1.state_dict(), self.Q2.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        self.Q1.load_state_dict(state[0])
        self.Q2.load_state_dict(state[1])
        self.policy.load_state_dict(state[2])
        self._copy_nets()

    def reset(self):
        self.action_noise.reset()

    def train(self, iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = []
        self.train_iter += 1

        if self.train_iter % self._config["update_target_every"] == 0:
            self._copy_nets()

        for i in range(iter_fit):
            data = self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:, 0]))  # s_t
            a = to_torch(np.stack(data[:, 1]))  # a_t
            rew = to_torch(np.stack(data[:, 2])[:, None])  # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:, 3]))  # s_t+1
            done = to_torch(np.stack(data[:, 4])[:, None])  # done signal  (batchsize,1)

            # ### TD3 ###
            # Trick One: Clipped Double-Q Learning.
            # TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form
            # the targets in the Bellman error loss functions.
            # Trick Two: “Delayed” Policy Updates. 
            # TD3 updates the policy (and target networks) less frequently than the Q-function. The paper recommends one
            # policy update for every two Q-function updates.
            # Trick Three: Target Policy Smoothing. 
            # TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by
            # smoothing out Q along changes in action.
            
            with torch.no_grad():
                # Generate noise for target policy smoothing
                noise = (torch.randn_like(a) * self._config["policy_noise"]).clamp(-self._config["noise_clip"],
                                                                                   self._config["noise_clip"])
                
                # Compute the target action with added noise and clamp it within the action space bounds
                a_prime = (self.policy_target(s_prime) + noise).clamp(torch.tensor(self._action_space.low),
                                                                      torch.tensor(self._action_space.high))
                
                # Compute the target Q-values using the target Q-networks
                q1_prime = self.Q1_target.Q_value(s_prime, a_prime)
                q2_prime = self.Q2_target.Q_value(s_prime, a_prime)
                
                # Use the minimum of the two Q-values to form the target
                q_prime = torch.min(q1_prime, q2_prime)
                
                # Compute the TD target
                td_target = rew + self._config["discount"] * (1.0 - done) * q_prime

            # Optimize the Q objectives
            q1_loss = self.Q1.fit(s, a, td_target)
            q2_loss = self.Q2.fit(s, a, td_target)

            # Optimize actor objective
            if self.train_iter % self._config["policy_freq"] == 0:
                self.optimizer.zero_grad()
                q = torch.min(self.Q1.Q_value(s, self.policy(s)), self.Q2.Q_value(s, self.policy(s)))
                actor_loss = -torch.mean(q)
                actor_loss.backward()
                losses.append(
                    (q1_loss, q2_loss, actor_loss.item() if self.train_iter % self._config["policy_freq"] == 0 else 0))

        return losses


# DDPG Agent
class DDPGAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.1,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128, 128],
            "hidden_sizes_critic": [128, 128, 64],
            "update_target_every": 100,
            "use_target_net": True,
            "colNoise": False
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        self._colNoise = self._config['colNoise']

        # pink noise
        if self._colNoise:
            self.action_noise = ColoredNoise((self._action_n))
        # OU Noise (default)
        else:
            self.action_noise = OUNoise((self._action_n))

        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Q Network
        self.Q = QFunction(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           hidden_sizes=self._config["hidden_sizes_critic"],
                           learning_rate=self._config["learning_rate_critic"])
        # target Q Network
        self.Q_target = QFunction(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  hidden_sizes=self._config["hidden_sizes_critic"],
                                  learning_rate=0)

        self.policy = Feedforward(input_size=self._obs_dim,
                                  hidden_sizes=self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun=torch.nn.ReLU(),
                                  output_activation=torch.nn.Tanh())
        self.policy_target = Feedforward(input_size=self._obs_dim,
                                         hidden_sizes=self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun=torch.nn.ReLU(),
                                         output_activation=torch.nn.Tanh())

        self._copy_nets()

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self._config["learning_rate_actor"],
                                          eps=0.000001)
        self.train_iter = 0

    def _copy_nets(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        #
        action = self.policy.predict(observation) + eps*self.action_noise()  # action in -1 to 1 (+ noise)
        action = self._action_space.low + (action + 1.0) / 2.0 * (self._action_space.high - self._action_space.low)
        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
        self._copy_nets()

    def reset(self):
        self.action_noise.reset()

    def train(self, iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = []
        self.train_iter += 1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._copy_nets()
        for i in range(iter_fit):

            # sample from the replay buffer
            data = self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:, 0]))  # s_t
            a = to_torch(np.stack(data[:, 1]))  # a_t
            rew = to_torch(np.stack(data[:, 2])[:, None])  # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:, 3]))  # s_t+1
            done = to_torch(np.stack(data[:, 4])[:, None])  # done signal  (batchsize,1)

            if self._config["use_target_net"]:
                q_prime = self.Q_target.Q_value(s_prime, self.policy_target.forward(s_prime))
            else:
                q_prime = self.Q.Q_value(s_prime, self.policy.forward(s_prime))
            # target
            gamma = self._config['discount']
            td_target = rew + gamma * (1.0-done) * q_prime

            # optimize the Q objective
            fit_loss = self.Q.fit(s, a, td_target)

            # optimize actor objective
            self.optimizer.zero_grad()
            q = self.Q.Q_value(s, self.policy.forward(s))
            actor_loss = -torch.mean(q)
            actor_loss.backward()
            self.optimizer.step()

            losses.append((fit_loss, actor_loss.item()))

        return losses


# a trained Opponent (DDPG Agent) to play against in the Hockey environment
class DDPGOpponent():
  def __init__(self, keep_mode=True):
    self.keep_mode = keep_mode

    # env_name = "RND_Hockey"
    # alg = "pinkNoiseRND"
    # eps = 0.1
    # ts = 32
    # lr = 0.0001
    # s = None
    # episodes = 2000

    checkpoint = "../../agents/DDPG_RND_Hockey_2000-m2005.0-eps0.1-t32-l0.0001-s1.pth"
    env = h_env.HockeyEnv()
    self.agent = DDPGAgent(env.observation_space, env.action_space)
    self.agent.restore_state(torch.load(checkpoint))

  def act(self, obs):
    action = self.agent.act(obs)
    return action


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps', action='store',  type='float',
                         dest='eps', default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train', action='store',  type='int',
                         dest='train', default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr', action='store',  type='float',
                         dest='lr', default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes', action='store',  type='float',
                         dest='max_episodes', default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update', action='store',  type='float',
                         dest='update_every', default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed', action='store',  type='int',
                         dest='seed', default=None,
                         help='random seed (default %default)')
    optParser.add_option('-a', '--algorithm', action='store', type='string',
                         dest='alg', default="pure",
                         help='algorithm modification (default %default)')
    optParser.add_option('-p', '--policy', action='store', type='string',
                         dest='pol', default="DDPG-default",
                         help='policy /strategy (DDPG or TD3) (default %default)')
    opts, args = optParser.parse_args()
    # ############# Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous = True)
    if env_name == "Hockey":
        env = h_env.HockeyEnv()
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = opts.max_episodes  # max training episodes
    max_timesteps = 2000         # max timesteps in one episode

    train_iter = opts.train      # update networks for given batched after every episode
    eps = opts.eps               # noise of DDPG policy
    lr = opts.lr                 # learning rate of DDPG policy
    random_seed = opts.seed
    alg = opts.alg               # modification of algorithm
    pol = opts.pol               # policy/strategy

    # activate modifications
    if alg == "pure":
        act_pink = False
        act_RND = False
    elif alg == "pinkNoise":
        act_pink = True
        act_RND = False
    elif alg == "RND":
        act_pink = False
        act_RND = True
    elif alg == "pinkNoiseRND":
        act_pink = True
        act_RND = True
    else:
        raise ValueError("Unknown algorithm modification")

    # Initialization of RND-Module
    if act_RND:
        rnd = RNDModule(input_dim=env.observation_space.shape[0], output_dim=16)  # Parameter: Output dimension
        rnd_optimizer = torch.optim.Adam(rnd.parameters(), lr=0.005)  # Parameter: Learning rate
    #############################################

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    if pol == "TD3":
        # Initialize TD3 Agent
        agent = TD3(env.observation_space, env.action_space, eps=eps, learning_rate_actor=lr,
                    update_target_every=opts.update_every, colNoise=act_pink)
    else:  # so pol=="DDPG" is default
        agent = DDPGAgent(env.observation_space, env.action_space, eps=eps, learning_rate_actor=lr,
                          update_target_every=opts.update_every, colNoise=act_pink)
    
    opponent = h_env.BasicOpponent()

    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    # RND variables
    counter = 1
    rnd_scale = 2
    rnd_states = []
    batch_size_rnd = 32
    rnd_bonus_stats = RunningMeanStd()
    rnd_threshold = 0.05

    def save_statistics():
        with open(f"./results/{pol}_{alg}_{env_name}-m{max_episodes}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}"
                  f"-stat.pkl", 'wb') as f:
            pickle.dump({"rewards": rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    # training loop
    for i_episode in range(1, int(max_episodes)+1):
        ob, _info = env.reset()
        obs_agent2 = env.obs_agent_two()

        agent.reset()
        total_reward = 0

        # RND counter
        if i_episode % 20 == 0:
            counter += 1

        for t in range(max_timesteps):
            timestep += 1
            # done = False

            a1 = agent.act(ob)
            a2 = opponent.act(obs_agent2)
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
            
            total_reward += reward

            # RND
            if act_RND:
                # Calculate the RND exploration bonus
                s = torch.from_numpy(np.array(ob, dtype=np.float32)).to(device)
                exploration_bonus = rnd.forward(s).item()
                rnd_states.append(s)

                rnd_bonus_stats.update([exploration_bonus])
                normalized_bonus = rnd_bonus_stats.normalize(exploration_bonus)

                if exploration_bonus > rnd_threshold:  # if state is unknown -> explore
                    reward += rnd_scale * normalized_bonus

                # Limitation of the buffer size to avoid storage problems
                if len(rnd_states) > 1000:
                    rnd_state_to_remove = random.choice(rnd_states)
                    rnd_states = [state for state in rnd_states if not (torch.equal(state, rnd_state_to_remove))]

                if i_episode % (1 * counter) == 0:
                    # Training RND-Module batch-wise
                    if len(rnd_states) >= batch_size_rnd:
                        # Select states batch-wise
                        rnd_states.sort(key=lambda x: rnd.forward(x).item(), reverse=True)
                        states_batch = torch.stack(rnd_states[:batch_size_rnd]).to(
                            device)  # train the most difficult states

                        # states_batch = torch.stack(rnd_states[:batch_size_rnd]).to(device)
                        # Target and predictor
                        target_output = rnd.target(states_batch)
                        predicted_output = rnd.predictor(states_batch)
                        # RND loss and optimizer
                        # Different loss functions: L1Loss, MSELoss(), CrossEntropyLoss,
                        # see https://pytorch.org/docs/stable/nn.html
                        loss_rnd = nn.SmoothL1Loss()(predicted_output, target_output)
                        rnd_optimizer.zero_grad()
                        loss_rnd.backward()
                        rnd_optimizer.step()
                        # Delete from list
                        del rnd_states[:batch_size_rnd]
            
            agent.store_transition((ob, a1, reward, ob_new, done))
            ob = ob_new
            obs_agent2 = env.obs_agent_two()

            if done or trunc:
                break

        # print("total reward after 250 steps:", total_reward)
        losses.extend(agent.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(agent.state(), f'./results/{pol}_{alg}_{env_name}_{i_episode}_m{max_episodes}-eps{eps}'
                                      f'-t{train_iter}-l{lr}-s{random_seed}.pth')
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
    save_statistics()


if __name__ == '__main__':
    main()

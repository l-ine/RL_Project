import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle

# from pink import PinkActionNoise
# install with pip install pink-noise-rl
# from https://github.com/martius-lab/pink-noise-rl
# but not sure how to use it here


import memory as mem
from feedforward import Feedforward

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
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=1)
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass

        pred = self.Q_value(observations,actions)
        # Compute Loss
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(torch.hstack([observations,actions]))

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
            + self._theta * ( - self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)


import numpy as np


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
            param.requires_grad = False  # Target network is fixed

    def forward(self, state):
        target_output = self.target(state)
        predicted_output = self.predictor(state)
        exploration_bonus = ((target_output - predicted_output) ** 2).mean()
        return exploration_bonus


class TD3():
    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible '
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.'
                                   ' (Require Box)'.format(action_space, self))

        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_n = action_space.shape[0]
        self._action_space = action_space

        self._config = {
            "eps": 0.1,  # Epsilon: noise strength to add to policy
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
            print(self.action_noise)
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

        # if self.train_iter % self._config["update_target_every"] == 0:
        #    self._copy_nets()
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._copy_nets()

        for i in range(iter_fit):
            data = self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:, 0]))  # s_t
            a = to_torch(np.stack(data[:, 1]))  # a_t
            rew = to_torch(np.stack(data[:, 2])[:, None])  # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:, 3]))  # s_t+1
            done = to_torch(np.stack(data[:, 4])[:, None])  # done signal  (batchsize,1)

            with torch.no_grad():
                # Trick 3: Target Policy Smoothing.
                # TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by
                # smoothing out Q along changes in action.

                # Generate noise for target policy smoothing
                noise = (torch.randn_like(a) * self._config["policy_noise"]).clamp(-self._config["noise_clip"],
                                                                                   self._config["noise_clip"])

                # Compute the target action with added noise and clamp it within the action space bounds
                a_prime = (self.policy_target(s_prime) + noise).clamp(torch.tensor(self._action_space.low),
                                                                      torch.tensor(self._action_space.high))

                # Trick 1: Clipped Double-Q Learning.
                # TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to
                # form the targets in the Bellman error loss functions.
                # Compute the target Q-values using the target Q-networks
                q1_prime = self.Q1_target.Q_value(s_prime, a_prime)
                q2_prime = self.Q2_target.Q_value(s_prime, a_prime)

                # Use the minimum of the two Q-values to form the target
                q_prime = torch.min(q1_prime, q2_prime)

            # Compute the TD target
            gamma = self._config['discount']
            td_target = rew + gamma * (1.0 - done) * q_prime

            # Optimize the Q objectives
            q1_loss = self.Q1.fit(s, a, td_target)
            q2_loss = self.Q2.fit(s, a, td_target)

            # Trick 2: “Delayed” Policy Updates.
            # TD3 updates the policy (and target networks) less frequently than the Q-function, recommended: one
            # policy update for every two Q-function updates.

            # Optimize actor objective
            if self.train_iter % self._config["policy_freq"] == 0:
                self.optimizer.zero_grad()
                q = torch.min(self.Q1.Q_value(s, self.policy(s)), self.Q2.Q_value(s, self.policy(s)))
                actor_loss = -torch.mean(q)
                actor_loss.backward()
                self.optimizer.step()
                losses.append(
                    (q1_loss, q2_loss, actor_loss.item() if self.train_iter % self._config["policy_freq"] == 0 else 0))

        return losses


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
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.1,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
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
                            hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_critic"])
        # target Q Network
        self.Q_target = QFunction(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = 0)

        self.policy = Feedforward(input_size=self._obs_dim,
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = torch.nn.Tanh())
        self.policy_target = Feedforward(input_size=self._obs_dim,
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = torch.nn.Tanh())

        self._copy_nets()

        self.optimizer=torch.optim.Adam(self.policy.parameters(),
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
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._copy_nets()
        for i in range(iter_fit):

            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:,0])) # s_t
            a = to_torch(np.stack(data[:,1])) # a_t
            rew = to_torch(np.stack(data[:,2])[:,None]) # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:,3])) # s_t+1
            done = to_torch(np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)

            if self._config["use_target_net"]:
                q_prime = self.Q_target.Q_value(s_prime, self.policy_target.forward(s_prime))
            else:
                q_prime = self.Q.Q_value(s_prime, self.policy.forward(s_prime))
            # target
            gamma=self._config['discount']
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


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='float',
                         dest='max_episodes',default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update',action='store',  type='float',
                         dest='update_every',default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=None,
                         help='random seed (default %default)')
    optParser.add_option('-p', '--policy', action='store', type='string',
                         dest='pol', default="DDPG-default",
                         help='policy /strategy (DDPG or TD3) (default %default)')
    optParser.add_option('-a', '--algorithm', action='store', type='string',
                         dest='alg', default="DDPG-default",
                         help='algorithm modification (default %default)')
    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    pol = opts.pol  # policy/strategy
    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous = True)
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = opts.max_episodes # max training episodes
    max_timesteps = 2000         # max timesteps in one episode

    train_iter = opts.train      # update networks for given batched after every episode
    eps = opts.eps               # noise of DDPG policy
    lr  = opts.lr                # learning rate of DDPG policy
    random_seed = opts.seed
    alg = opts.alg               # modification of algorithm

    # activate modifications
    if alg == "DDPG-default":
        act_pink = False
        act_RND = False
    elif alg == "pinkNoise":
        act_pink = True
        act_RND = False
    elif alg == "pinkNoiseRND":
        act_pink = True
        act_RND = False
    else:
        act_pink, act_RND = False

    # Initialization of RND-Module
    if act_RND:
        rnd = RNDModule(input_dim=env.observation_space.shape[0], output_dim=16)  # output_dim kann angepasst werden
        rnd_optimizer = torch.optim.Adam(rnd.parameters(), lr=0.001)  # Optional: Learning rate für das RND-Modul
    #############################################


    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)


    if pol == "TD3":
        # Initialize TD3 Agent
        agent = TD3(env.observation_space, env.action_space, eps=eps, learning_rate_actor=lr,
                    update_target_every=opts.update_every, colNoise=act_pink)
        # agent.restore_state(torch.load(checkpoint_agent, weights_only=True))
    else:  # so pol=="DDPG" is default
        agent = DDPGAgent(env.observation_space, env.action_space, eps=eps, learning_rate_actor=lr,
                          update_target_every=opts.update_every, colNoise=act_pink)
        #agent.restore_state(torch.load(checkpoint_agent, weights_only=True))
        #print(f"agent restored: {checkpoint_agent}")
    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(f"./results/{pol}_{alg}_{env_name}-m{max_episodes}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    # training loop
    for i_episode in range(1, int(max_episodes)+1):
        ob, _info = env.reset()
        agent.reset()
        total_reward=0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a = agent.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward += reward

            if act_RND:
                # Calculate the RND exploration bonus
                s = torch.from_numpy(np.array(ob, dtype=np.float32)).to(device)
                exploration_bonus = rnd.forward(s)
                reward += exploration_bonus.item()

            agent.store_transition((ob, a, reward, ob_new, done))
            ob=ob_new

            if act_RND:
                # Training RND-Module
                loss_rnd = nn.MSELoss()
                target_output = rnd.target(s)
                predicted_output = rnd.predictor(s)
                loss = loss_rnd(predicted_output, target_output)
                rnd_optimizer.zero_grad()
                loss.backward()
                rnd_optimizer.step()

            if done or trunc: break

        losses.extend(agent.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(agent.state(), f'./results/{pol}_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth')
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
    save_statistics()

if __name__ == '__main__':
    main()

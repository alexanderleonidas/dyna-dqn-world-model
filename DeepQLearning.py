import random
import math
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple, deque
from globals import DEVICE, torch

# Experience Replay
# Stores the transitions (s, a, s', r) in a replay memory which is used to sample from and train the agent
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    Standard DQN Network Architecture
    Inputs: State vector and Action vector.
    Outputs: Predicted Q-values for each action.
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine the next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)


class WorldModel(nn.Module):
    """
    Learns the environment transition dynamics: P(s', r | s, a)
    Inputs: Concatenated State vector and One-Hot encoded action vector.
    Outputs: Predicted Next State (8 dimensions) + Predicted Reward (1 dimension).
    """

    def __init__(self, n_observations, n_actions):
        super(WorldModel, self).__init__()
        self.n_actions = n_actions
        # Input features = state dimensions + action dimensions
        self.fc1 = nn.Linear(n_observations + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_next_state = nn.Linear(256, n_observations)
        self.fc_reward = nn.Linear(256, 1)

    def forward(self, state, action_indices):
        # Convert action indices to one-hot encodings
        action_one_hot = f.one_hot(action_indices.squeeze(-1), num_classes=self.n_actions).float()
        x = torch.cat([state, action_one_hot], dim=-1)

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))

        pred_next_state = self.fc_next_state(x)
        pred_reward = self.fc_reward(x)
        return pred_next_state, pred_reward

class DQNAgent:
    # discount_factor: float, learning_rate: float = 1e-3, initial_epsilon:float = 1.0, final_epsilon:float = 0.01, epsilon_decay:float = 2500, tau:float = 0.005
    def __init__(self, env: gym.Env, hyperparameters: dict):
        self.env = env
        self.name = 'DQNAgent' if not hyperparameters.get('use_dyna', False) else 'DynaDQNAgent'
        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.n

        # Unpack standard hyperparameters
        self.seed = hyperparameters['seed']
        self.discount_factor = hyperparameters['discount_factor']
        self.initial_epsilon = hyperparameters['initial_epsilon']
        self.epsilon = self.initial_epsilon
        self.final_epsilon = hyperparameters['final_epsilon']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.tau = hyperparameters['tau']
        self.memory_capacity = hyperparameters['memory_capacity']
        self.memory_batch_size = hyperparameters['batch_size']
        self.steps_done = 0

        # Base Networks
        self.policy_net = DQN(self.obs_space_dims, self.action_space_dims).to(DEVICE)
        self.target_net = DQN(self.obs_space_dims, self.action_space_dims).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=hyperparameters['learning_rate'])
        self.memory = ReplayMemory(self.memory_capacity)

        # --- DYNA-SPECIFIC PROPERTIES ---
        self.use_dyna = hyperparameters.get('use_dyna', False)
        self.latest_wm_loss = 0.0  # Tracking metric for analysis

        if self.use_dyna:
            self.planning_steps = hyperparameters.get('planning_steps', 10)
            self.world_model = WorldModel(self.obs_space_dims, self.action_space_dims).to(DEVICE)
            self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=hyperparameters.get('wm_lr', 1e-3))
            self.wm_criterion = nn.MSELoss()

    def store_transition(self, state: torch.Tensor|None, action: torch.Tensor, next_state: torch.Tensor|None, reward: torch.Tensor):
        self.memory.push(state, action, next_state, reward)

    def sample_action(self, observation: torch.Tensor):
        sample = random.random()
        self.steps_done += 1
        if sample > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(observation).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=DEVICE, dtype=torch.long)

    def decay_epsilon(self):
        self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * math.exp(-1. * self.steps_done / self.epsilon_decay)

    def optimize_model(self, custom_batch=None):
        """Standard DQN Optimization loop; accepts custom batches for simulated planning"""
        if custom_batch is not None:
            state_batch, action_batch, next_state_batch, reward_batch, non_final_mask = custom_batch
            batch_size = state_batch.size(0)
        else:
            if len(self.memory) < self.memory_batch_size:
                return
            transitions = self.memory.sample(self.memory_batch_size)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts a batch-array of Transitions to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)

            # Handle empty non-final next states edge case safely
            non_final_next_states = [s for s in batch.next_state if s is not None]
            if len(non_final_next_states) > 0:
                non_final_next_states = torch.cat(non_final_next_states)
            else:
                non_final_next_states = torch.empty(0, self.obs_space_dims, device=DEVICE)

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            batch_size = self.memory_batch_size

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=DEVICE)
        if custom_batch is not None:
            # For simulated updates, next_state_batch contains all states; mask tracks terminal conditions
            with torch.no_grad():
                if non_final_mask.any():
                    next_state_values[non_final_mask] = self.target_net(next_state_batch[non_final_mask]).max(1).values
        else:
            with torch.no_grad():
                if len(non_final_next_states) > 0:
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch.squeeze(-1)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train_world_model(self):
        """Train the predictive World Model using real trajectories from the replay buffer"""
        if len(self.memory) < self.memory_batch_size:
            return 0.0

        transitions = self.memory.sample(self.memory_batch_size)
        batch = Transition(*zip(*transitions))

        # We only learn transitions from valid states containing actual next states
        valid_indices = [i for i, ns in enumerate(batch.next_state) if ns is not None]
        if len(valid_indices) == 0:
            return 0.0

        state_batch = torch.cat([batch.state[i] for i in valid_indices])
        action_batch = torch.cat([batch.action[i] for i in valid_indices])
        next_state_batch = torch.cat([batch.next_state[i] for i in valid_indices])
        reward_batch = torch.cat([batch.reward[i] for i in valid_indices]).unsqueeze(-1)

        # Forward pass through model world
        pred_next_states, pred_rewards = self.world_model(state_batch, action_batch)

        # Calculate loss for each component
        state_loss = self.wm_criterion(pred_next_states, next_state_batch)
        reward_loss = self.wm_criterion(pred_rewards, reward_batch)
        total_wm_loss = state_loss + reward_loss

        # Step optimizer
        self.wm_optimizer.zero_grad()
        total_wm_loss.backward()
        self.wm_optimizer.step()

        return total_wm_loss.item()

    def update(self):
        """Update the Q-Network and the target network. If Dyna is enabled, also update the world model and perform simulated planning updates."""
        # 1. Optimize Q-network via a standard model-free step
        self.optimize_model()

        # 2. If Dyna is active, update the world model and run imaginary planning steps
        if self.use_dyna and len(self.memory) >= self.memory_batch_size:
            self.latest_wm_loss = self.train_world_model()

            # Simulated Planning Loops
            for _ in range(self.planning_steps):
                # Sample random previous real states and actions from the replay buffer
                samples = self.memory.sample(self.memory_batch_size)
                sim_states = torch.cat([s.state for s in samples])

                # For high exploration of actions, we can sample uniform random choices
                # sim_actions = torch.randint(0, self.action_space_dims, (self.memory_batch_size, 1), device=DEVICE)
                # Alternatively, we can sample from the current policy network to explore more relevant state action pairs
                sim_actions = self.policy_net(sim_states).max(1).indices.view(self.memory_batch_size, 1)


                # Generate simulated experiences using the learned environment dynamics
                with torch.no_grad():
                    pred_next_states, pred_rewards = self.world_model(sim_states, sim_actions)

                # Standard Dyna implies continuing assumptions (assume non-terminal during planning)
                sim_non_final_mask = torch.ones(self.memory_batch_size, dtype=torch.bool, device=DEVICE)

                # Update Q-values using simulated experience
                sim_batch = (sim_states, sim_actions, pred_next_states, pred_rewards, sim_non_final_mask)
                self.optimize_model(custom_batch=sim_batch)

        # Soft target update: θ′ ← τ θ + (1 − τ)θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


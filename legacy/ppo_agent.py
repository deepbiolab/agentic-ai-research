"""
PPO Agent with Agentic AI Framework

This module implements a Proximal Policy Optimization (PPO) agent following the
agentic AI paradigm, emphasizing:
- Perception: Processing visual input from the environment
- Memory: Storing and utilizing past experiences
- Learning: Updating policy through PPO algorithm
- Planning: Computing future rewards and optimizing for long-term gains
- Decision-making: Using a policy network to select actions
- Action: Interacting with the environment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from enum import Enum
import gymnasium as gym
import ale_py

# Register Atari environments
gym.register_envs(ale_py)

# Define action space for Pong
class Action(Enum):
    NOOP = 0         # No operation
    FIRE = 1         # Fire
    RIGHT = 2        # Move right
    LEFT = 3         # Move left
    RIGHTFIRE = 4    # Move right and fire
    LEFTFIRE = 5     # Move left and fire


###################
# PERCEPTION MODULE
###################

class PerceptionModule:
    """
    Handles the processing of raw sensory input (game frames) into a format
    suitable for the agent's decision-making process.
    """
    
    @staticmethod
    def preprocess(image, bkg_color=np.array([144, 72, 17])):
        """
        Preprocess a single game frame by cropping, downsampling, and normalizing.
        
        Args:
            image (np.ndarray): The input image (game frame) as a NumPy array.
            bkg_color (np.ndarray): The RGB background color to subtract.
            
        Returns:
            np.ndarray: The processed image, normalized to [0, 1].
        """
        # Crop the image to remove irrelevant parts (e.g., score and borders)
        cropped_image = image[34:-16, :]
        # Downsample the image by taking every second pixel (both rows and columns)
        downsampled_image = cropped_image[::2, ::2]
        # Subtract the background color
        adjusted_image = downsampled_image - bkg_color
        # Convert to grayscale by taking the mean across the color channels
        grayscale_image = np.mean(adjusted_image, axis=-1)
        # Normalize pixel values to the range [0, 1]
        normalized_image = grayscale_image / 255.0
        
        return normalized_image
    
    @staticmethod
    def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
        """
        Convert outputs of ParallelEnv to inputs for tensor processing.
        
        Args:
            images (list or np.ndarray): Batch of input images (game frames).
            bkg_color (np.ndarray): The RGB background color to subtract.
            
        Returns:
            torch.Tensor: The processed batch of images as a tensor, normalized to [0, 1].
        """
        # Ensure images are in a NumPy array
        # shape: (time_steps, height, width, channel)
        batch_images = np.asarray(images)
        
        # If the input has less than 5 dimensions, expand the dimensions
        # shape: (time_steps, batch, height, width, channels)
        if len(batch_images.shape) < 5:
            batch_images = np.expand_dims(batch_images, 1)
            
        # Process each image in the batch using logic from the preprocess function
        # shape: (time_steps, batch, height, width, channels)
        cropped_images = batch_images[:, :, 34:-16, :, :]  # Crop the images
        downsampled_images = cropped_images[:, :, ::2, ::2, :]  # Downsample the images
        adjusted_images = downsampled_images - bkg_color  # Subtract the background color
        
        # Convert to grayscale and normalize pixel values to [0, 1]
        # shape: (time_steps, batch, height, width)
        grayscale_images = np.mean(adjusted_images, axis=-1)  
        normalized_images = grayscale_images / 255.0
        
        # Rearrange the batch dimension to match the expected input format
        # shape: (batch, time_steps, height, width) or in other way
        #        (batch, channel, height, width)
        batch_input = torch.from_numpy(normalized_images).float()
        batch_input = batch_input.permute(1, 0, 2, 3)
        return batch_input


###################
# DECISION-MAKING MODULE
###################

class PolicyNetwork(nn.Module):
    """
    Neural network that maps processed observations to action probabilities,
    representing the agent's decision-making process.
    """
    
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
        self.RIGHTFIRE = Action.RIGHTFIRE.value
        self.LEFTFIRE = Action.LEFTFIRE.value
        
        # 80x80x2 input
        # Conv layers with progressively decreasing spatial dimensions
        # First convolutional layer: input 80x80x2 -> output 20x20x32
        self.conv1 = nn.Conv2d(2, 32, kernel_size=4, stride=4)
        
        # Second convolutional layer: input 20x20x32 -> output 9x9x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        
        # Third convolutional layer: input 9x9x64 -> output 7x7x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Flatten the output of the conv layers
        self.size = 7 * 7 * 64  # Flattened size after conv layers
        
        # Fully connected layers: progressively decreasing sizes
        self.fc1 = nn.Linear(self.size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 1)          # Final fully connected layer (output)
        
    def forward(self, x):
        # Pass through the convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the tensor
        x = x.view(-1, self.size)
        
        # Pass through the fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        
        # the output is the probability of moving right, P(right)
        # so, for moving left, have P(left) = 1-P(right)
        return x
    
    def get_action_probs(self, states, actions):
        """
        Get probabilities for given states and actions.
        Useful for calculating policy gradients.
        
        Args:
            states: Preprocessed state tensor
            actions: Tensor of actions taken
            
        Returns:
            Tensor of action probabilities
        """
        probs = self(states).squeeze()
        probs = probs.view(*actions.shape)
        action_probs = torch.where(
            actions == self.RIGHTFIRE,
            probs,
            1.0 - probs
        )
        return action_probs


###################
# ACTION MODULE
###################

class ActionModule:
    """
    Responsible for selecting actions based on the policy's output and
    executing them in the environment.
    """
    
    @staticmethod
    def get_random_action(n, policy_network=None):
        """Generate random actions for exploration"""
        return np.random.choice(
            [Action.RIGHTFIRE.value, Action.LEFTFIRE.value],
            size=n
        )
    
    @staticmethod
    @torch.no_grad()
    def select_action(policy_network, frames, perception_module, mode='inference', device='cpu'):
        """
        Select actions based on policy for either inference or trajectory collection.
        
        Args:
            policy_network: The neural network policy
            frames: Tuple of (frame1, frame2) or preprocessed tensor
            perception_module: Module for processing raw observations
            mode: Either 'inference' for single action or 'collect' for trajectory collection
            device: Device to run computations on
            
        Returns:
            For mode='inference': single action value
            For mode='collect': tuple of (states, actions, action_probs)
        """
        # Ensure frames are on the correct device
        if isinstance(frames, tuple):
            states = perception_module.preprocess_batch(frames).to(device)
        else:
            states = frames.to(device)
        
        # Get action probabilities
        probs = policy_network(states).squeeze().detach().cpu().numpy()
        
        if mode == 'inference':
            # Single instance inference
            action = policy_network.RIGHTFIRE if np.random.random() < probs else policy_network.LEFTFIRE
            return action
        
        elif mode == 'collect':
            # Batch processing for trajectory collection
            n = probs.shape[0]
            
            # Generate random values for all instances
            random_values = np.random.rand(n)
            
            # Select actions based on probabilities
            actions = np.where(
                random_values < probs,
                policy_network.RIGHTFIRE,
                policy_network.LEFTFIRE
            )
            
            # Calculate action probabilities
            action_probs = np.where(
                actions == policy_network.RIGHTFIRE,
                probs,
                1.0 - probs
            )
            
            return states, actions, action_probs
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'inference' or 'collect'")


###################
# MEMORY MODULE
###################

class MemoryModule:
    """
    Manages the collection and storage of experiences (trajectories)
    for learning and policy updates.
    """
    
    @staticmethod
    def perform_random_steps(env, nrand, parallel=False):
        """
        Perform a number of random steps in the environment to initialize the game.
        Supports both single and parallel environments.
        
        Args:
            env: The game environment (single or parallel)
            nrand (int): Number of random steps to perform
            parallel (bool): Whether the environment is parallel
            
        Returns:
            tuple: The last two frames after performing random steps
        """
        # Reset environment
        env.reset()
        
        # Get environment size (1 for single env, n for parallel envs)
        n = len(env.ps) if parallel else 1
        
        # Unified action definitions
        fire_action = np.full(n, Action.FIRE.value, dtype=np.int32)
        noop_action = np.full(n, Action.NOOP.value, dtype=np.int32)
        
        # Start the game with a FIRE action
        env.step(fire_action.item() if n == 1 else fire_action)
        
        # Initialize frames
        frames1, frames2 = None, None
        
        # Perform random steps
        for _ in range(nrand):
            # Get and format random action
            action = ActionModule.get_random_action(n)
            frames1, _, dones, *_ = env.step(action.item() if n == 1 else action)
            frames2, _, dones, *_ = env.step(noop_action.item() if n == 1 else noop_action)
            
            # Check termination condition
            if (dones if n == 1 else dones.any()):
                break
        
        return frames1, frames2
    
    @staticmethod
    def collect_trajectories(envs, policy_network, perception_module, action_module, max_t=200, nrand=5, device='cpu'):
        """
        Collect trajectories from parallel environments for learning.
        
        Args:
            envs: Parallel environment instances
            policy_network: Policy network for action selection
            perception_module: Module for processing observations
            action_module: Module for selecting actions
            max_t: Maximum trajectory length
            nrand: Number of random steps at the beginning
            device: Device to run computations on
            
        Returns:
            tuple: (states, actions, probs, rewards) for learning
        """
        # number of parallel instances
        n = len(envs.ps)
        
        # initialize returning lists and start the game!
        states = torch.zeros(max_t, n, 2, 80, 80, device=device)
        rewards = torch.zeros(max_t, n, device=device)
        probs = torch.zeros(max_t, n, device=device)
        actions = torch.zeros(max_t, n, dtype=torch.int8, device=device)
        
        # Initialize the game and perform random steps
        frame1, frame2 = MemoryModule.perform_random_steps(envs, nrand, parallel=True)
        
        for t in range(max_t):
            # Predict actions
            state, action, actions_prob = action_module.select_action(
                policy_network, (frame1, frame2), perception_module, mode='collect', device=device
            )
            
            # Advance the game, we take one action and skip game forward
            frame1, reward1, done, _ = envs.step(action)
            frame2, reward2, done, _ = envs.step([Action.NOOP.value] * n)
            
            reward = reward1 + reward2
            
            # store the result
            states[t] = state
            rewards[t] = torch.from_numpy(reward)
            probs[t] = torch.from_numpy(actions_prob)
            actions[t] = torch.from_numpy(action)
            
            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if done.any():
                break
        
        # Convert time steps dimension into batch for vectorize inference
        states = states.view(-1, 2, 80, 80)
        
        return states, actions, probs, rewards


###################
# PLANNING MODULE
###################

class PlanningModule:
    """
    Responsible for computing future rewards and optimizing for long-term gains.
    """
    
    @staticmethod
    def compute_future_rewards(rewards, gamma=0.99, device='cpu'):
        """
        Compute future rewards using discount factors and matrix multiplication.
        
        Args:
            rewards: Tensor of rewards [T]
            gamma: Discount factor
            device: Device to run computations on
            
        Returns:
            future_rewards: Tensor of future rewards [T]
        """
        T = len(rewards)
        indices = torch.arange(T).to(device=rewards.device)
        discounts_matrix = gamma ** (indices.unsqueeze(0) - indices.unsqueeze(1)).clamp_min(0)
        mask = torch.triu(torch.ones(T, T)).to(device=rewards.device)
        future_rewards = torch.matmul(discounts_matrix * mask, rewards)
        
        # Normalize rewards for stable learning
        mean = future_rewards.mean(dim=1, keepdim=True)
        std = future_rewards.std(dim=1, keepdim=True)
        future_rewards = (future_rewards - mean) / (std + 1e-10)
        
        return future_rewards


###################
# LEARNING MODULE
###################

class LearningModule:
    """
    Implements the PPO algorithm for updating the agent's policy based on
    collected experiences.
    """
    
    @staticmethod
    def clipped_surrogate(
        policy_network, states, actions, old_probs, rewards, 
        planning_module, gamma=0.995, beta=0.01, epsilon=0.1
    ):
        """
        Compute the PPO clipped surrogate objective function.
        
        Args:
            policy_network: The neural network policy
            states: Tensor of states
            actions: Tensor of actions taken
            old_probs: Tensor of action probabilities from old policy
            rewards: Tensor of rewards received
            planning_module: Module for computing future rewards
            gamma: Discount factor
            beta: Entropy coefficient for exploration
            epsilon: Clipping parameter
            
        Returns:
            torch.Tensor: The PPO objective value
        """
        # Compute and normalize future rewards
        future_rewards = planning_module.compute_future_rewards(rewards, gamma=gamma)
        
        # Convert states to probability
        # probs shape: (time_steps, batch_size)
        probs = policy_network.get_action_probs(states, actions)
        
        # Compute policy gradient
        # (1/B)(1/T)∑∑ π_θ'(a_t|s_t)/ π_θ(a_t|s_t) * R_t^future
        # g shape: (time_steps, batch_size)
        ratio = probs / old_probs
        clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        g = torch.min(ratio * future_rewards, clip * future_rewards)
        
        # Regularization term
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(
            probs * torch.log(old_probs + 1.0e-10)
            + (1.0 - probs) * torch.log(1.0 - old_probs + 1.0e-10)
        )
        
        return torch.mean(g + beta * entropy)
    
    @staticmethod
    def train_ppo(
        envs,
        policy_network,
        optimizer,
        perception_module,
        action_module,
        memory_module,
        planning_module,
        n_episodes=500,
        gamma=0.99,
        epsilon=0.1,
        beta=0.01,
        max_t=320,
        window=20,
        epoch=4,
        checkpoint_path="checkpoint.pth",
        device='cpu'
    ):
        """
        Train the agent using the PPO algorithm.
        
        Args:
            envs: Parallel environment instances
            policy_network: Policy network for action selection
            optimizer: Optimizer for policy network
            perception_module: Module for processing observations
            action_module: Module for selecting actions
            memory_module: Module for collecting trajectories
            planning_module: Module for computing future rewards
            n_episodes: Number of episodes to train
            gamma: Discount factor
            epsilon: Clipping parameter
            beta: Entropy coefficient for exploration
            max_t: Maximum trajectory length
            window: How often to print progress
            epoch: Number of policy updates on history trajectories
            checkpoint_path: Path to save the model checkpoint
            device: Device to run computations on
            
        Returns:
            list: List of mean rewards per episode
        """
        n = len(envs.ps)
        
        # keep track of progress
        scores = []
        best_score = float("-inf")
        for i_episode in range(1, n_episodes + 1):
            # collect trajectories
            states, actions, probs, rewards = memory_module.collect_trajectories(
                envs, policy_network, perception_module, action_module, 
                max_t=max_t, device=device
            )
            
            # gradient ascent step
            for _ in range(epoch):
                pg = LearningModule.clipped_surrogate(
                    policy_network,
                    states,
                    actions,
                    probs,
                    rewards,
                    planning_module,
                    gamma=gamma,
                    epsilon=epsilon,
                    beta=beta,
                )
                loss = -pg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # reduce exploration over time
            beta *= 0.995
            
            # reduce clip factor over time
            epsilon *= 0.999
            
            # calculate total rewards
            total_rewards = rewards.sum(dim=0)
            
            # display progress
            avg_score = total_rewards.mean().item()
            
            # store mean reward on batch
            scores.append(avg_score)
            
            if avg_score > best_score:
                best_score = avg_score
                torch.save(policy_network.state_dict(), checkpoint_path)
                print(f"Checkpoint saved with new best score: {best_score:.2f}")
            
            if i_episode % window == 0:
                print(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}")
                print(f"Total Rewards on {n} envs: {total_rewards.cpu().numpy()}")
        
        return scores


###################
# AGENT INTEGRATION
###################

class PPOAgent:
    """
    Integrates all modules into a cohesive agent that can perceive, learn,
    remember, plan, decide, and act in an environment.
    """
    
    def __init__(self, env_name='PongDeterministic-v4', n_envs=8, seed=42, device=None):
        """
        Initialize the PPO agent with all its modules.
        
        Args:
            env_name: Name of the environment
            n_envs: Number of parallel environments
            seed: Random seed
            device: Device to run computations on
        """
        # Set device
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Initialize environments
        from parallel_env import ParallelEnv
        self.envs = ParallelEnv(env_name, n=n_envs, seed=seed)
        self.single_env = gym.make(env_name)
        
        # Initialize modules
        self.perception = PerceptionModule()
        self.policy_network = PolicyNetwork().to(self.device)
        self.action = ActionModule()
        self.memory = MemoryModule()
        self.planning = PlanningModule()
        self.learning = LearningModule()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-4)
    
    def train(self, n_episodes=500, checkpoint_path="checkpoint.pth"):
        """
        Train the agent using PPO algorithm.
        
        Args:
            n_episodes: Number of episodes to train
            checkpoint_path: Path to save the model checkpoint
            
        Returns:
            list: List of mean rewards per episode
        """
        return self.learning.train_ppo(
            envs=self.envs,
            policy_network=self.policy_network,
            optimizer=self.optimizer,
            perception_module=self.perception,
            action_module=self.action,
            memory_module=self.memory,
            planning_module=self.planning,
            n_episodes=n_episodes,
            checkpoint_path=checkpoint_path,
            device=self.device
        )
    
    def load(self, checkpoint_path="checkpoint.pth"):
        """Load a trained policy from a checkpoint."""
        self.policy_network.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
    
    def interact(self, max_t=2000, nrand=5):
        """
        Play a game using the trained policy and return frames for visualization.
        
        Args:
            max_t: Maximum number of timesteps
            nrand: Number of random steps at the beginning
            
        Returns:
            list: List of frames for visualization
        """
        # Initialize the game and perform random steps
        frame1, frame2 = self.memory.perform_random_steps(self.single_env, nrand)
        
        selected_frames = []
        for _ in range(max_t):
            # Select an action using the policy
            action = self.action.select_action(
                self.policy_network, 
                (frame1, frame2), 
                self.perception, 
                mode='inference',
                device=self.device
            )
            
            # Perform the action and a NOOP step
            frame1, _, done, *_ = self.single_env.step(action)
            frame2, _, done, *_ = self.single_env.step(Action.NOOP.value)
            
            # Store the frame
            selected_frames.append(frame1)
            
            # End the game if done
            if done:
                break
        
        self.single_env.close()
        return selected_frames


# Example usage
if __name__ == "__main__":
    # Create and train the agent
    agent = PPOAgent(n_envs=8)
    scores = agent.train(n_episodes=600)
    
    # Visualize training progress
    from plot_utils import plot_scores
    plot_scores(scores, rolling_window=20)
    
    # Play a game with the trained agent
    agent.load()
    frames = agent.interact()
    
    # Save animation
    from plot_utils import save_animation
    save_animation(frames)
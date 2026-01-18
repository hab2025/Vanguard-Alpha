"""
Reinforcement Learning Agent for Vanguard-Alpha
Deep RL agent for learning optimal trading strategies
"""

import numpy as np
import pandas as pd
import logging
import gymnasium as gym
from gymnasium import spaces
from config import (
    RL_TOTAL_TIMESTEPS, RL_LEARNING_RATE,
    RL_BATCH_SIZE, RL_GAMMA, RL_ENTROPY_COEF,
    MODELS_DIR
)
from utils import setup_logger

logger = setup_logger(__name__)

class TradingEnvironment(gym.Env):
    """Custom trading environment for RL agent"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        """
        Initialize trading environment
        
        Args:
            data: Historical market data
            initial_balance: Initial cash balance
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # State: [balance, shares_held, current_price, SMA_20, SMA_50, RSI, MACD]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        self.max_value = self.initial_balance
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state observation"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        row = self.data.iloc[self.current_step]
        
        obs = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held,
            row['Close'],
            row.get('SMA_20', row['Close']),
            row.get('SMA_50', row['Close']),
            row.get('RSI', 50),
            row.get('MACD', 0)
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute action
        if action == 1:  # Buy
            shares_to_buy = int(self.balance / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.shares_held += shares_to_buy
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price
                self.balance += proceeds
                self.shares_held = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        self.total_value = self.balance + (self.shares_held * current_price)
        reward = self.total_value - self.max_value
        
        if self.total_value > self.max_value:
            self.max_value = self.total_value
        
        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Get next observation
        observation = self._get_observation()
        
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_value': self.total_value,
            'current_price': current_price
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment state"""
        current_price = self.data.iloc[self.current_step]['Close']
        print(f"Step: {self.current_step}, Price: ${current_price:.2f}, "
              f"Balance: ${self.balance:.2f}, Shares: {self.shares_held}, "
              f"Total Value: ${self.total_value:.2f}")

class RLTradingAgent:
    """Reinforcement Learning trading agent"""
    
    def __init__(self, env: TradingEnvironment):
        """
        Initialize RL agent
        
        Args:
            env: Trading environment
        """
        self.env = env
        self.model = None
        
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # Wrap environment
            vec_env = DummyVecEnv([lambda: env])
            
            # Create PPO model
            self.model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=RL_LEARNING_RATE,
                n_steps=2048,
                batch_size=RL_BATCH_SIZE,
                gamma=RL_GAMMA,
                ent_coef=RL_ENTROPY_COEF,
                verbose=1
            )
            
            logger.info("RL Agent initialized with PPO")
            
        except ImportError:
            logger.warning("stable-baselines3 not available, RL agent disabled")
            self.model = None
    
    def train(self, total_timesteps: int = RL_TOTAL_TIMESTEPS):
        """
        Train the RL agent
        
        Args:
            total_timesteps: Total training timesteps
        """
        if self.model is None:
            logger.error("Model not initialized")
            return
        
        logger.info(f"Training RL agent for {total_timesteps} timesteps...")
        
        try:
            self.model.learn(total_timesteps=total_timesteps)
            logger.info("Training completed")
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
    
    def predict(self, observation):
        """
        Predict action for given observation
        
        Args:
            observation: Current state observation
            
        Returns:
            Predicted action
        """
        if self.model is None:
            return 0  # Default to hold
        
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, filename: str = "rl_trading_agent"):
        """
        Save trained model
        
        Args:
            filename: Model filename
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        filepath = f"{MODELS_DIR}/{filename}"
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filename: str = "rl_trading_agent"):
        """
        Load trained model
        
        Args:
            filename: Model filename
        """
        if self.model is None:
            logger.error("Model not initialized")
            return
        
        try:
            from stable_baselines3 import PPO
            filepath = f"{MODELS_DIR}/{filename}"
            self.model = PPO.load(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def evaluate(self, n_episodes: int = 10) -> dict:
        """
        Evaluate agent performance
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            logger.error("Model not initialized")
            return {}
        
        total_rewards = []
        final_values = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
            final_values.append(info['total_value'])
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'avg_return': (np.mean(final_values) - self.env.initial_balance) / self.env.initial_balance
        }

class SimpleQLearningAgent:
    """Simple Q-Learning agent (fallback if stable-baselines3 not available)"""
    
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize Q-Learning agent
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Simple Q-table (discretized states)
        self.q_table = {}
    
    def _discretize_state(self, state):
        """Discretize continuous state"""
        return tuple(np.round(state, 2))
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state):
        """Update Q-values"""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    """Test RL agent"""
    from data_fetcher import DataFetcher
    
    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_historical_data('AAPL', period='1y', interval='1d')
    
    if data.empty:
        logger.error("Failed to fetch data")
        return
    
    # Create environment
    env = TradingEnvironment(data, initial_balance=10000)
    
    # Create agent
    agent = RLTradingAgent(env)
    
    if agent.model is not None:
        # Train agent
        agent.train(total_timesteps=10000)
        
        # Evaluate
        results = agent.evaluate(n_episodes=5)
        
        print("\nEvaluation Results:")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Average Final Value: ${results['avg_final_value']:.2f}")
        print(f"Average Return: {results['avg_return']*100:.2f}%")
        
        # Save model
        agent.save()
    else:
        logger.info("Using simple Q-Learning agent")
        simple_agent = SimpleQLearningAgent(state_size=7, action_size=3)
        
        # Train for a few episodes
        for episode in range(100):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action = simple_agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                simple_agent.update(obs, action, reward, next_obs)
                obs = next_obs
                done = terminated or truncated
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Final Value: ${info['total_value']:.2f}")

if __name__ == "__main__":
    main()

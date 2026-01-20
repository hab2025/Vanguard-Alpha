"""
Vanguard-Alpha Advanced AI Engine v2.0
=======================================
محرك الذكاء الاصطناعي المتقدم

المكونات:
1. Transformer Predictor - للتنبؤ بالأسعار
2. PPO Agent - لاتخاذ قرارات التداول
3. Large Replay Buffer - 200,000 تجربة
4. GPU Optimization - دعم CUDA
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# =============== CONFIGURATION ===============

@dataclass
class AIConfig:
    """إعدادات AI"""
    # Model Architecture
    feature_dim: int = 20
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 0.0001
    batch_size: int = 64
    replay_buffer_size: int = 200000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    
    # PPO specific
    ppo_epochs: int = 10
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# =============== TRANSFORMER PREDICTOR ===============

class TransformerPredictor(nn.Module):
    """Transformer للتنبؤ بالأسعار"""
    
    def __init__(self, config: AIConfig):
        super(TransformerPredictor, self).__init__()
        
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Linear(config.feature_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(config.d_model, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # [price_up, price_down, price_stable]
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        x: (batch_size, sequence_length, feature_dim)
        """
        # Embed input
        x = self.input_embedding(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq, d_model)
        
        # Take last timestep
        x = x[:, -1, :]  # (batch, d_model)
        
        # Output layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return self.softmax(x)

class PositionalEncoding(nn.Module):
    """Positional Encoding للـ Transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (batch_size, sequence_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# =============== PPO AGENT ===============

class PPOAgent(nn.Module):
    """PPO Agent لاتخاذ قرارات التداول"""
    
    def __init__(self, config: AIConfig):
        super(PPOAgent, self).__init__()
        
        self.config = config
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(config.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [BUY, SELL, HOLD]
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(config.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        """Forward pass"""
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
    
    def get_action(self, state, deterministic=False):
        """اختيار action"""
        with torch.no_grad():
            action_probs, _ = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action = torch.multinomial(action_probs, 1).squeeze(-1)
            
            return action, action_probs
    
    def evaluate(self, states, actions):
        """تقييم actions"""
        action_probs, state_values = self.forward(states)
        
        # Get log probabilities
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_probs, state_values, entropy

# =============== REPLAY BUFFER ===============

class LargeReplayBuffer:
    """Replay Buffer كبير - 200,000 تجربة"""
    
    def __init__(self, capacity: int = 200000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.logger = logging.getLogger("ReplayBuffer")
    
    def push(self, state, action, reward, next_state, done):
        """إضافة تجربة"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """أخذ عينة عشوائية"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# =============== ADVANCED AI ENGINE ===============

class AdvancedAIEngine:
    """محرك AI المتقدم - Transformer + PPO"""
    
    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig()
        self.logger = logging.getLogger("AdvancedAIEngine")
        
        # Models
        self.transformer = TransformerPredictor(self.config).to(self.config.device)
        self.ppo_agent = PPOAgent(self.config).to(self.config.device)
        
        # Optimizers
        self.transformer_optimizer = optim.Adam(
            self.transformer.parameters(),
            lr=self.config.learning_rate
        )
        self.ppo_optimizer = optim.Adam(
            self.ppo_agent.parameters(),
            lr=self.config.learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = LargeReplayBuffer(self.config.replay_buffer_size)
        
        # Training stats
        self.training_steps = 0
        self.episode_rewards = []
        
        self.logger.info(f"🤖 Advanced AI Engine initialized on {self.config.device}")
        self.logger.info(f"📊 Replay buffer capacity: {self.config.replay_buffer_size:,}")
    
    def predict(self, features: Dict) -> Dict:
        """التنبؤ واتخاذ القرار"""
        
        try:
            # تحويل Features إلى tensor
            feature_vector = self._features_to_vector(features)
            state = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.config.device)
            
            # الحصول على action من PPO
            with torch.no_grad():
                action, action_probs = self.ppo_agent.get_action(state, deterministic=False)
            
            action_idx = action.item()
            confidence = action_probs[0, action_idx].item()
            
            # تحويل action إلى signal
            actions_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
            
            signal = {
                'action': actions_map[action_idx],
                'confidence': confidence,
                'action_probs': action_probs[0].cpu().numpy().tolist()
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def train_step(self, batch_size: int = 64):
        """خطوة تدريب واحدة"""
        
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).to(self.config.device)
        
        # PPO update
        for _ in range(self.config.ppo_epochs):
            # Evaluate actions
            log_probs, state_values, entropy = self.ppo_agent.evaluate(states, actions)
            
            # Calculate advantages (simplified)
            with torch.no_grad():
                _, next_values = self.ppo_agent(next_states)
                advantages = rewards + self.config.gamma * next_values.squeeze() * (1 - dones) - state_values.squeeze()
            
            # Calculate losses
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), rewards + self.config.gamma * next_values.squeeze() * (1 - dones))
            entropy_loss = -entropy.mean()
            
            total_loss = actor_loss + self.config.value_coef * critic_loss + self.config.entropy_coef * entropy_loss
            
            # Update
            self.ppo_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ppo_agent.parameters(), 0.5)
            self.ppo_optimizer.step()
        
        self.training_steps += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item()
        }
    
    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """تحويل Features إلى vector"""
        
        # قائمة الميزات بالترتيب
        feature_keys = [
            'price', 'price_change_pct',
            'price_sma_5', 'price_sma_20', 'price_sma_50',
            'price_ema_12', 'price_ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'volatility', 'bb_upper', 'bb_lower', 'bb_width',
            'volume', 'volume_sma_20', 'volume_ratio',
            'spread', 'order_flow'
        ]
        
        vector = []
        for key in feature_keys:
            value = features.get(key, 0.0)
            # Normalization (simple)
            if 'price' in key and 'change' not in key:
                value = value / 50000.0  # normalize price
            elif key == 'rsi':
                value = value / 100.0
            elif key == 'volatility':
                value = value * 50.0
            
            vector.append(value)
        
        return np.array(vector, dtype=np.float32)
    
    def save_models(self, path: str):
        """حفظ النماذج"""
        torch.save({
            'transformer': self.transformer.state_dict(),
            'ppo_agent': self.ppo_agent.state_dict(),
            'transformer_optimizer': self.transformer_optimizer.state_dict(),
            'ppo_optimizer': self.ppo_optimizer.state_dict(),
            'training_steps': self.training_steps
        }, path)
        self.logger.info(f"💾 Models saved to {path}")
    
    def load_models(self, path: str):
        """تحميل النماذج"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.transformer.load_state_dict(checkpoint['transformer'])
        self.ppo_agent.load_state_dict(checkpoint['ppo_agent'])
        self.transformer_optimizer.load_state_dict(checkpoint['transformer_optimizer'])
        self.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.logger.info(f"📂 Models loaded from {path}")

# =============== EXAMPLE USAGE ===============

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create AI engine
    config = AIConfig()
    ai_engine = AdvancedAIEngine(config)
    
    # Test prediction
    test_features = {
        'price': 42000.0,
        'price_change_pct': 0.001,
        'price_sma_5': 41950.0,
        'price_sma_20': 41900.0,
        'price_sma_50': 41800.0,
        'price_ema_12': 41980.0,
        'price_ema_26': 41920.0,
        'rsi': 55.0,
        'macd': 50.0,
        'macd_signal': 45.0,
        'macd_hist': 5.0,
        'volatility': 0.02,
        'bb_upper': 42500.0,
        'bb_lower': 41500.0,
        'bb_width': 1000.0,
        'volume': 1000.0,
        'volume_sma_20': 950.0,
        'volume_ratio': 1.05,
        'spread': 0.0005,
        'order_flow': 0.0
    }
    
    signal = ai_engine.predict(test_features)
    print(f"\n🤖 AI Signal: {signal}")
    
    # Test training (with dummy data)
    for i in range(100):
        state = np.random.randn(20)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(20)
        done = False
        
        ai_engine.replay_buffer.push(state, action, reward, next_state, done)
    
    if len(ai_engine.replay_buffer) >= 64:
        losses = ai_engine.train_step(batch_size=64)
        print(f"\n📊 Training losses: {losses}")
    
    print(f"\n✅ Replay buffer size: {len(ai_engine.replay_buffer):,}")
    print(f"✅ Training steps: {ai_engine.training_steps}")

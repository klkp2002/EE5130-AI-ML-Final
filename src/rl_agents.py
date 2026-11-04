"""
Reinforcement Learning Module for EE5130 MNIST Project.

Formalizes MNIST classification as a Contextual Bandit (One-Step MDP):
- State: 50-dimensional PCA feature vector of 28x28 handwritten digit.
- Action: {0, 1, 2, ..., 9} (the predicted digit).
- Reward: +1.0 for correct classification, -1.0 for incorrect.
- Episode Horizon: T=1 (terminal after 1 step).

Implements:
1. Q-Learning
2. SARSA
3. TD(0) (Actor-Critic / Baseline TD)
4. Monte Carlo (Sample Return)
With uniform MNISTBaseClassifier predict() API.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.classifiers import MNISTBaseClassifier
from src.features import FeaturePipeline
from src.utils import MODELS_DIR, set_seed


class ContextualBanditMNISTEnv:
    """One-step MDP environment for handwritten digit classification."""

    def __init__(self, X: np.ndarray, y: np.ndarray, feature_pipeline: Optional[FeaturePipeline] = None):
        self.X = X
        self.y = y
        self.feature_pipeline = feature_pipeline
        self.n_samples = len(X)
        self.current_idx = 0
        self.current_state: Optional[np.ndarray] = None
        self.current_label: Optional[int] = None

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        self.current_idx = rng.integers(0, self.n_samples)
        raw_x = self.X[self.current_idx]
        self.current_label = int(self.y[self.current_idx])

        if self.feature_pipeline is not None:
            self.current_state = self.feature_pipeline.transform_pca(raw_x.reshape(1, -1))[0]
        else:
            self.current_state = (raw_x.astype(np.float32) / 255.0)

        return self.current_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        reward = 1.0 if action == self.current_label else -1.0
        done = True
        info = {"true_label": self.current_label, "correct": (action == self.current_label)}
        # Terminal next state
        next_state = np.zeros_like(self.current_state)
        return next_state, reward, done, info


class BaseRLAgent(MNISTBaseClassifier):
    """Base class for contextual bandit RL agents."""

    def __init__(self, name: str, state_dim: int = 50, n_actions: int = 10, lr: float = 0.01, gamma: float = 0.99):
        super().__init__(name=name, family="Reinforcement Learning")
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        # Linear Q-function: W shape (state_dim, 10), b shape (10,)
        self.W = np.zeros((state_dim, n_actions), dtype=np.float32)
        self.b = np.zeros(n_actions, dtype=np.float32)
        self.feature_pipeline: Optional[FeaturePipeline] = None

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            return state @ self.W + self.b
        return state @ self.W + self.b

    def select_action(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        q = self.get_q_values(state)
        return int(np.argmax(q))

    @abstractmethod
    def update(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool, a_next: Optional[int] = None) -> float:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRLAgent":
        # Full training handled by train_bandit_agent script
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, f"{self.name} is not fitted."
        if self.feature_pipeline is not None:
            states = self.feature_pipeline.transform_pca(X)
        else:
            states = (X.astype(np.float32) / 255.0) if X.max() > 1.0 else X.astype(np.float32)

        q_vals = self.get_q_values(states)
        # Calibrated softmax over Q-values
        exp = np.exp(q_vals - np.max(q_vals, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)


class QLearningAgent(BaseRLAgent):
    """Q-Learning: off-policy temporal difference update."""

    def __init__(self, state_dim: int = 50, lr: float = 0.02):
        super().__init__(name="Q-Learning Agent", state_dim=state_dim, lr=lr)

    def update(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool, a_next: Optional[int] = None) -> float:
        current_q = float(s @ self.W[:, a] + self.b[a])
        # In one-step terminal episode: next_q = 0.0
        target = r if done else (r + self.gamma * np.max(self.get_q_values(s_next)))
        td_error = target - current_q

        # Gradient update for linear function approximation
        self.W[:, a] += self.lr * td_error * s
        self.b[a] += self.lr * td_error
        return td_error


class SARSAAgent(BaseRLAgent):
    """SARSA: on-policy temporal difference update."""

    def __init__(self, state_dim: int = 50, lr: float = 0.02):
        super().__init__(name="SARSA Agent", state_dim=state_dim, lr=lr)

    def update(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool, a_next: Optional[int] = None) -> float:
        current_q = float(s @ self.W[:, a] + self.b[a])
        target = r if done else (r + self.gamma * float(s_next @ self.W[:, a_next] + self.b[a_next]))
        td_error = target - current_q

        self.W[:, a] += self.lr * td_error * s
        self.b[a] += self.lr * td_error
        return td_error


class TD0Agent(BaseRLAgent):
    """TD(0) State-Value baseline actor-critic agent."""

    def __init__(self, state_dim: int = 50, lr: float = 0.02):
        super().__init__(name="TD(0) Agent", state_dim=state_dim, lr=lr)
        self.V_w = np.zeros(state_dim, dtype=np.float32)
        self.V_b = 0.0

    def update(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool, a_next: Optional[int] = None) -> float:
        # Estimate state value V(s)
        v_s = float(s @ self.V_w + self.V_b)
        td_error = r - v_s  # Target return is r since episode ends

        # Update value baseline
        self.V_w += self.lr * td_error * s
        self.V_b += self.lr * td_error

        # Update action preferences Q(s, a) using advantage td_error
        current_q = float(s @ self.W[:, a] + self.b[a])
        self.W[:, a] += self.lr * (r - current_q) * s
        self.b[a] += self.lr * (r - current_q)
        return td_error


class MonteCarloAgent(BaseRLAgent):
    """Monte Carlo agent updating policy from terminal sample returns."""

    def __init__(self, state_dim: int = 50, lr: float = 0.02):
        super().__init__(name="Monte Carlo Agent", state_dim=state_dim, lr=lr)

    def update(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool, a_next: Optional[int] = None) -> float:
        # In 1-step episode, empirical return G_t = r
        G = r
        current_q = float(s @ self.W[:, a] + self.b[a])
        error = G - current_q

        self.W[:, a] += self.lr * error * s
        self.b[a] += self.lr * error
        return error


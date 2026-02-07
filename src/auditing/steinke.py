"""Algorithm 2: Black-box auditing following Steinke et al. (2023).

Computes empirical privacy lower bounds using membership inference
on canary samples.

Reference:
    Steinke et al. "Privacy auditing with one (1) training run"
    (NeurIPS 2023)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import special
from scipy.optimize import brentq
from typing import Tuple, Optional


def compute_canary_scores(
    model: nn.Module,
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """Compute score function for each canary.
    
    Score = negative cross-entropy loss (higher = more likely memorized)
    
    Args:
        model: Trained model
        canary_images: Canary images [m, 3, 32, 32]
        canary_labels: Canary labels [m]
        device: Device
    
    Returns:
        Scores for each canary [m]
    """
    model = model.to(device)
    model.eval()
    
    canary_images = canary_images.to(device)
    canary_labels = canary_labels.to(device)
    
    with torch.no_grad():
        outputs = model(canary_images)
        criterion = nn.CrossEntropyLoss(reduction='none')
        losses = criterion(outputs, canary_labels)
        scores = -losses  # Higher score = lower loss = more memorized
    
    return scores.cpu()


def make_guesses(
    scores: torch.Tensor,
    in_mask: torch.Tensor,
    k_plus: int,
    k_minus: int
) -> Tuple[torch.Tensor, int]:
    """Make membership guesses based on scores.
    
    Args:
        scores: Score for each canary [m]
        in_mask: True labels (True if canary was in training)
        k_plus: Number of positive guesses (high scores → guess IN)
        k_minus: Number of negative guesses (low scores → guess OUT)
    
    Returns:
        Tuple of (guesses tensor, number of correct guesses)
        guesses[i] = 1 if guessing IN, -1 if guessing OUT, 0 if abstain
    """
    m = len(scores)
    guesses = torch.zeros(m, dtype=torch.long)
    
    # Sort scores
    sorted_indices = torch.argsort(scores, descending=True)
    
    # Top k+ scores → guess IN
    for i in range(k_plus):
        guesses[sorted_indices[i]] = 1
    
    # Bottom k- scores → guess OUT
    for i in range(k_minus):
        guesses[sorted_indices[m - 1 - i]] = -1
    
    # Count correct guesses
    # Correct if: guess=1 and in_mask=True, or guess=-1 and in_mask=False
    true_labels = in_mask.long() * 2 - 1  # True→1, False→-1
    correct = ((guesses == 1) & (true_labels == 1)) | ((guesses == -1) & (true_labels == -1))
    num_correct = correct.sum().item()
    
    return guesses, num_correct


def theorem1_bound(
    r: int,  # Total number of guesses
    v: int,  # Number of correct guesses
    epsilon: float,
    delta: float,
    m: int  # Number of canaries
) -> float:
    """Compute the probability bound from Theorem 1.
    
    Returns the probability that an (ε,δ)-DP algorithm achieves
    at least v correct guesses out of r total guesses.
    """
    # f(v) = P[Binomial(r, e^ε/(e^ε+1)) ≥ v]
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    
    # Use survival function of binomial
    f_v = 1 - special.betainc(v, r - v + 1, p)
    
    # Compute the correction term
    correction = 0
    if delta > 0 and m > 0:
        max_term = 0
        for i in range(1, m + 1):
            f_vi = 1 - special.betainc(v - i, r - v + i + 1, p) if v - i > 0 else 1.0
            f_v_curr = 1 - special.betainc(v, r - v + 1, p)
            term = (f_vi - f_v_curr) / i
            max_term = max(max_term, term)
        correction = 2 * m * delta * max_term
    
    return f_v + correction


def find_empirical_epsilon(
    num_correct: int,
    num_guesses: int,
    num_canaries: int,
    tau: float = 0.05,  # Probability threshold
    delta: float = 1e-5,
    epsilon_max: float = 20.0
) -> float:
    """Binary search for the largest ε satisfying Theorem 1.
    
    Args:
        num_correct: Number of correct guesses
        num_guesses: Total number of guesses (k+ + k-)
        num_canaries: Total number of canaries (m)
        tau: Probability threshold (typically 0.05)
        delta: Privacy parameter δ
        epsilon_max: Maximum epsilon to search
    
    Returns:
        Empirical epsilon lower bound
    """
    if num_correct <= num_guesses // 2:
        # No evidence of privacy leakage
        return 0.0
    
    def objective(eps):
        bound = theorem1_bound(num_guesses, num_correct, eps, delta, num_canaries)
        return bound - tau
    
    # Binary search for largest epsilon where bound < tau
    try:
        # Find where bound = tau
        eps_low, eps_high = 0.0, epsilon_max
        
        # Check if any epsilon works
        if objective(eps_low) < 0:
            return 0.0
        if objective(eps_high) > 0:
            return epsilon_max
        
        epsilon = brentq(objective, eps_low, eps_high)
        return epsilon
    except ValueError:
        return 0.0


def audit_steinke(
    model: nn.Module,
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    in_mask: torch.Tensor,
    k_plus: Optional[int] = None,
    k_minus: Optional[int] = None,
    tau: float = 0.05,
    delta: float = 1e-5,
    device: str = "cuda"
) -> Tuple[float, int, int]:
    """Algorithm 2: Steinke et al. auditing procedure.
    
    Args:
        model: Trained model to audit
        canary_images: All canary images
        canary_labels: All canary labels
        in_mask: Boolean mask indicating which canaries were in training
        k_plus: Number of positive guesses (default: m/4)
        k_minus: Number of negative guesses (default: m/4)
        tau: Probability threshold
        delta: Privacy parameter
        device: Device
    
    Returns:
        Tuple of (empirical_epsilon, num_correct, num_guesses)
    """
    m = len(canary_images)
    
    # Default: make m/4 guesses in each direction
    if k_plus is None:
        k_plus = m // 4
    if k_minus is None:
        k_minus = m // 4
    
    # Compute scores
    scores = compute_canary_scores(model, canary_images, canary_labels, device)
    
    # Make guesses
    guesses, num_correct = make_guesses(scores, in_mask, k_plus, k_minus)
    num_guesses = k_plus + k_minus
    
    # Find empirical epsilon
    epsilon = find_empirical_epsilon(
        num_correct, num_guesses, m, tau, delta
    )
    
    return epsilon, num_correct, num_guesses

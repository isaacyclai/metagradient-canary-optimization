"""Algorithm 3+4: Black-box auditing following Mahloujifar et al. (2024).

Computes empirical privacy lower bounds using paired canary scoring
and f-DP analysis.

Reference:
    Mahloujifar et al. "Auditing f-Differential Privacy in One Run"
    (arXiv:2410.22235)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import special
from scipy.optimize import brentq
from typing import Tuple, List, Optional


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


def create_canary_pairs(
    in_mask: torch.Tensor,
    seed: Optional[int] = None
) -> List[Tuple[int, int]]:
    """Create random pairings between IN and OUT canaries.
    
    Each pair contains one canary from C_IN and one from C_OUT.
    
    Args:
        in_mask: Boolean mask for C_IN
        seed: Random seed
    
    Returns:
        List of (in_idx, out_idx) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    in_indices = torch.where(in_mask)[0].numpy()
    out_indices = torch.where(~in_mask)[0].numpy()
    
    np.random.shuffle(in_indices)
    np.random.shuffle(out_indices)
    
    num_pairs = min(len(in_indices), len(out_indices))
    pairs = [(int(in_indices[i]), int(out_indices[i])) for i in range(num_pairs)]
    
    return pairs


def make_paired_guesses(
    scores: torch.Tensor,
    pairs: List[Tuple[int, int]],
    k: int
) -> Tuple[List[int], int]:
    """Make membership guesses based on paired scores.
    
    For each pair, predict the higher-scored canary was in training.
    Only make guesses for the top k pairs by score difference.
    
    Args:
        scores: Score for each canary [m]
        pairs: List of (in_idx, out_idx) pairs
        k: Number of pairs to make guesses on
    
    Returns:
        Tuple of (guesses list, number of correct guesses)
        guesses[i] = 1 if correctly guessing the IN canary
    """
    # Compute score differences for each pair
    diffs = []
    for in_idx, out_idx in pairs:
        diff = abs(scores[in_idx].item() - scores[out_idx].item())
        diffs.append(diff)
    
    # Sort pairs by score difference (descending)
    sorted_pair_indices = np.argsort(diffs)[::-1]
    
    # Make guesses on top k pairs
    num_correct = 0
    guesses = []
    
    for i in range(min(k, len(pairs))):
        pair_idx = sorted_pair_indices[i]
        in_idx, out_idx = pairs[pair_idx]
        
        # Guess the higher-scored one was in training
        if scores[in_idx] > scores[out_idx]:
            # Correctly predicted IN canary
            guesses.append(1)
            num_correct += 1
        else:
            # Incorrectly predicted OUT canary
            guesses.append(-1)
    
    return guesses, num_correct


def fdp_function(x: float, epsilon: float) -> float:
    """The f-DP trade-off function for (ε,δ)-DP.
    
    f(x) = max(0, 1 - δ - e^ε * x)
    
    For pure approximate DP: f̄(x) = e^ε * x + δ
    """
    return np.exp(epsilon) * x


def fdp_inverse(y: float, epsilon: float) -> float:
    """Inverse of fdp_function."""
    return y / np.exp(epsilon)


def algorithm4_bound(
    m: int,  # Number of pairs
    k: int,  # Number of guesses
    k_prime: int,  # Number of correct guesses
    tau: float,  # Probability threshold
    epsilon: float,
    delta: float = 0.0,
    s: int = 2  # Alphabet size (2 for binary IN vs OUT)
) -> bool:
    """Algorithm 4: Upper bound probability of correct guesses.
    
    Returns True if the probability of making k' correct guesses
    (out of k) is less than tau under the given f-DP function.
    
    Args:
        m: Total number of canary pairs
        k: Number of guesses made
        k_prime: Number of correct guesses
        tau: Probability threshold
        epsilon: Privacy parameter
        delta: Privacy parameter (default 0 for GDP)
        s: Alphabet size
    
    Returns:
        True if probability of k' correct guesses is bounded by tau
    """
    if k_prime <= 0:
        return True
    
    # Constants from Algorithm 4
    c = tau  # Simplified
    c_prime = tau
    
    # Initialize arrays
    h = np.zeros(k_prime + 1)
    r = np.zeros(k_prime + 1)
    
    r[k_prime] = tau * c / m if m > 0 else tau
    h[k_prime] = tau * (c_prime - c) / m if m > 0 else 0
    
    # Backward iteration
    for i in range(k_prime - 1, -1, -1):
        h[i] = (s - 1) * fdp_inverse(r[i + 1], epsilon)
        if k > i:
            r[i] = r[i + 1] + (i / (k - i)) * (h[i] - h[i + 1]) if k != i else r[i + 1]
        else:
            r[i] = r[i + 1]
    
    # Check condition
    return r[0] + h[0] >= k / m if m > 0 else True


def find_empirical_epsilon_mahloujifar(
    num_correct: int,
    num_guesses: int,
    num_pairs: int,
    tau: float = 0.05,
    delta: float = 1e-5,
    epsilon_max: float = 20.0
) -> float:
    """Binary search for the largest ε satisfying Algorithm 4.
    
    Args:
        num_correct: Number of correct guesses
        num_guesses: Number of guesses made
        num_pairs: Total number of canary pairs
        tau: Probability threshold
        delta: Privacy parameter
        epsilon_max: Maximum epsilon to search
    
    Returns:
        Empirical epsilon lower bound
    """
    if num_correct <= num_guesses // 2:
        return 0.0
    
    def passes_test(eps):
        return algorithm4_bound(num_pairs, num_guesses, num_correct, tau, eps, delta)
    
    # Binary search for largest epsilon where test passes
    eps_low, eps_high = 0.0, epsilon_max
    
    # Check boundaries
    if passes_test(eps_high):
        return epsilon_max
    if not passes_test(eps_low):
        return 0.0
    
    # Binary search
    while eps_high - eps_low > 0.01:
        eps_mid = (eps_low + eps_high) / 2
        if passes_test(eps_mid):
            eps_low = eps_mid
        else:
            eps_high = eps_mid
    
    return eps_low


def audit_mahloujifar(
    model: nn.Module,
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    in_mask: torch.Tensor,
    k: Optional[int] = None,
    tau: float = 0.05,
    delta: float = 1e-5,
    device: str = "cuda",
    seed: int = 42
) -> Tuple[float, int, int]:
    """Algorithm 3+4: Mahloujifar et al. auditing procedure.
    
    Args:
        model: Trained model to audit
        canary_images: All canary images
        canary_labels: All canary labels
        in_mask: Boolean mask indicating which canaries were in training
        k: Number of guesses to make (default: num_pairs/4)
        tau: Probability threshold
        delta: Privacy parameter
        device: Device
        seed: Random seed for pairing
    
    Returns:
        Tuple of (empirical_epsilon, num_correct, num_guesses)
    """
    # Create canary pairs
    pairs = create_canary_pairs(in_mask, seed)
    num_pairs = len(pairs)
    
    # Default: make guesses on 1/4 of pairs
    if k is None:
        k = max(1, num_pairs // 4)
    
    # Compute scores
    scores = compute_canary_scores(model, canary_images, canary_labels, device)
    
    # Make paired guesses
    guesses, num_correct = make_paired_guesses(scores, pairs, k)
    
    # Find empirical epsilon
    epsilon = find_empirical_epsilon_mahloujifar(
        num_correct, k, num_pairs, tau, delta
    )
    
    return epsilon, num_correct, k

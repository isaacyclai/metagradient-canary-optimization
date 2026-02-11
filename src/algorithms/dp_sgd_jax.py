"""DP-SGD training using JAX and jax_privacy-style implementation.

This module provides differentially private training using JAX's vmap
for efficient per-sample gradient computation, matching the approach
from the original paper which uses jax-privacy.

Reference:
    DeepMind jax_privacy: https://github.com/google-deepmind/jax_privacy
"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from flax.training import train_state
from flax import linen as nn
import optax
from typing import Tuple, Dict, Any, Optional, Callable
from functools import partial
import numpy as np
from tqdm import tqdm


def compute_rdp_subsampled_gaussian(q: float, sigma: float, alpha: float) -> float:
    """Compute RDP for a single step of subsampled Gaussian mechanism.
    
    Uses the analytical bound from Mironov 2017 "Rényi Differential Privacy".
    For subsampling rate q and noise multiplier sigma.
    
    Args:
        q: Subsampling rate (batch_size / dataset_size)
        sigma: Noise multiplier
        alpha: RDP order
    
    Returns:
        RDP value for a single step
    """
    if sigma == 0:
        return float('inf')
    if q == 0:
        return 0.0
    if alpha <= 1:
        return 0.0
    
    # For small q, use the binomial bound (Mironov 2017, Theorem 9)
    # This is the standard approach used by Opacus and TensorFlow Privacy
    
    # The tight bound for subsampled Gaussian comes from:
    # Mironov, I. "Rényi Differential Privacy" (2017)
    # Wang, Y. et al. "Subsampled Rényi Differential Privacy..." (2019)
    
    def log1mexp(x):
        """Compute log(1 - exp(x)) stably for x < 0."""
        if x < -1:
            return np.log1p(-np.exp(x))
        else:
            return np.log(-np.expm1(x))
    
    # For integer orders, use the closed-form bound
    if isinstance(alpha, int) or alpha == int(alpha):
        alpha = int(alpha)
        
        # Compute log of terms in the RDP sum
        log_terms = []
        for k in range(alpha + 1):
            # Binomial coefficient
            log_binom = (
                np.sum(np.log(np.arange(1, alpha + 1))) -
                np.sum(np.log(np.arange(1, k + 1))) -
                np.sum(np.log(np.arange(1, alpha - k + 1)))
            ) if k > 0 and k < alpha else 0.0
            
            # (1-q)^(alpha-k) * q^k
            if 1 - q > 0 and q > 0:
                log_q_term = (alpha - k) * np.log(1 - q) + k * np.log(q)
            elif q == 1:
                log_q_term = 0.0 if k == alpha else -float('inf')
            else:  # q == 0
                log_q_term = 0.0 if k == 0 else -float('inf')
            
            # exp(k(k-1)/(2*sigma^2))
            log_exp_term = k * (k - 1) / (2 * sigma**2)
            
            log_terms.append(log_binom + log_q_term + log_exp_term)
        
        # Log-sum-exp for numerical stability
        max_log = max(log_terms)
        if max_log == -float('inf'):
            return 0.0
        
        log_sum = max_log + np.log(sum(np.exp(t - max_log) for t in log_terms))
        rdp = log_sum / (alpha - 1)
        
        return max(0.0, rdp)
    
    # For non-integer orders, use a simpler upper bound
    # This is the Gaussian mechanism bound without tight subsampling
    # (conservative but correct)
    return alpha * q**2 / (2 * sigma**2)


def compute_epsilon(
    steps: int,
    batch_size: int,
    dataset_size: int,
    noise_multiplier: float,
    delta: float = 1e-5
) -> float:
    """Compute privacy epsilon using RDP accountant.
    
    Uses the analytical RDP bounds for subsampled Gaussian mechanism
    from Mironov 2017, then converts to (epsilon, delta)-DP.
    
    Args:
        steps: Number of training steps
        batch_size: Batch size
        dataset_size: Total dataset size
        noise_multiplier: Noise multiplier (sigma)
        delta: Target delta
    
    Returns:
        Epsilon value
    """
    # Subsampling rate
    q = batch_size / dataset_size
    
    # RDP orders to check (focus on integers for tight bounds)
    orders = list(range(2, 64)) + [64, 128, 256, 512]
    
    # Accumulate RDP over steps (composition)
    rdp = [steps * compute_rdp_subsampled_gaussian(q, noise_multiplier, alpha) 
           for alpha in orders]
    
    # Convert RDP to (epsilon, delta)-DP using optimal conversion
    def rdp_to_dp(rdp_value: float, alpha: float, delta: float) -> float:
        """Convert RDP to (epsilon, delta)-DP."""
        if rdp_value == float('inf'):
            return float('inf')
        if rdp_value == 0:
            return 0.0
        # Standard conversion: ε = ρ - log(δ)/(α-1) + log((α-1)/α)
        # From Mironov 2017, Proposition 3
        return rdp_value + np.log1p(-1/alpha) - (np.log(delta) + np.log(alpha)) / (alpha - 1)
    
    eps_candidates = [rdp_to_dp(r, a, delta) for r, a in zip(rdp, orders)]
    return min(eps_candidates)


def noise_multiplier_from_epsilon(
    target_epsilon: float,
    steps: int,
    batch_size: int,
    dataset_size: int,
    delta: float = 1e-5,
    tol: float = 0.01
) -> float:
    """Binary search for noise multiplier that achieves target epsilon.
    
    Args:
        target_epsilon: Target privacy budget
        steps: Number of training steps
        batch_size: Batch size
        dataset_size: Total dataset size
        delta: Target delta
        tol: Tolerance for binary search
    
    Returns:
        Noise multiplier (sigma)
    """
    lo, hi = 0.1, 100.0
    
    while hi - lo > tol:
        mid = (lo + hi) / 2
        eps = compute_epsilon(steps, batch_size, dataset_size, mid, delta)
        
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    
    return hi


def create_dp_train_state(
    rng: jax.Array,
    model: nn.Module,
    learning_rate: float = 0.1,
    momentum: float = 0.9
) -> train_state.TrainState:
    """Create a Flax TrainState for DP-SGD training.
    
    Args:
        rng: JAX random key
        model: Flax model
        learning_rate: Learning rate
        momentum: SGD momentum
    
    Returns:
        Initialized TrainState
    """
    dummy_input = jnp.ones((1, 32, 32, 3))
    variables = model.init(rng, dummy_input, train=True)
    
    # Use SGD with momentum
    tx = optax.sgd(learning_rate, momentum=momentum)
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )
    
    return state


def augment_image(rng, image):
    """Apply random augmentation to a single image (NHWC format).
    
    Applies random horizontal flip and random crop with 4px padding,
    matching the standard CIFAR-10 augmentation used in DP-SGD literature
    (De et al. 2022, "Unlocking High-Accuracy DP Image Classification").
    
    Args:
        rng: JAX random key
        image: Single image array of shape (H, W, C)
    
    Returns:
        Augmented image of same shape
    """
    rng_flip, rng_crop = random.split(rng)
    
    # Random horizontal flip
    flip = random.bernoulli(rng_flip)
    image = jnp.where(flip, jnp.flip(image, axis=1), image)
    
    # Random crop: pad 4 pixels on each side, then crop back to original size
    h, w, c = image.shape[0], image.shape[1], image.shape[2]
    padded = jnp.pad(image, ((4, 4), (4, 4), (0, 0)), mode='reflect')
    
    crop_y = random.randint(rng_crop, (), 0, 9)  # 0..8 inclusive for 40->32
    rng_x = random.fold_in(rng_crop, 1)
    crop_x = random.randint(rng_x, (), 0, 9)
    
    image = jax.lax.dynamic_slice(padded, (crop_y, crop_x, 0), (h, w, c))
    
    return image


def per_sample_loss(params, apply_fn, x, y):
    """Compute loss for a single sample (for vmap)."""
    # Add batch dimension
    x = jnp.expand_dims(x, 0)
    
    logits = apply_fn(
        {'params': params},
        x,
        train=True
    )
    
    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits)
    return -log_probs[0, y]


def per_sample_loss_augmented(params, apply_fn, x, y, rng, aug_mult):
    """Compute average loss over aug_mult augmented versions of a single sample.
    
    Implements augmentation multiplicity from De et al. 2022:
    for each sample, compute loss on K augmented copies and average.
    Taking grad() of this gives the averaged gradient, which is then
    clipped — improving signal-to-noise ratio at zero privacy cost.
    
    Args:
        params: Model parameters
        apply_fn: Model apply function
        x: Single image (H, W, C)
        y: Label (scalar)
        rng: JAX random key
        aug_mult: Number of augmented copies (K)
    
    Returns:
        Average loss over K augmented copies
    """
    # Generate K different random keys for augmentations
    rngs = random.split(rng, aug_mult)
    # Create K copies and augment each differently
    x_copies = jnp.broadcast_to(x, (aug_mult,) + x.shape)
    augmented = vmap(augment_image)(rngs, x_copies)
    # Compute loss for each augmented copy
    losses = vmap(
        lambda xi: per_sample_loss(params, apply_fn, xi, y)
    )(augmented)
    return jnp.mean(losses)


def _clip_single_grad(g, max_grad_norm):
    """Clip a single sample's gradients to max_grad_norm."""
    flat_grads = jax.tree_util.tree_leaves(g)
    total_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in flat_grads))
    clip_factor = jnp.minimum(1.0, max_grad_norm / (total_norm + 1e-10))
    return jax.tree_util.tree_map(lambda x: x * clip_factor, g)


def clip_and_sum_gradients(
    grads: Dict,
    max_grad_norm: float,
) -> Dict:
    """Clip per-sample gradients and sum them (no noise).
    
    Used for gradient accumulation: process micro-batches independently,
    sum the clipped gradients, then add noise once.
    
    Args:
        grads: Per-sample gradients (leading batch dimension)
        max_grad_norm: Maximum L2 norm for clipping
    
    Returns:
        Sum of clipped per-sample gradients
    """
    clipped_grads = jax.vmap(
        lambda g: _clip_single_grad(g, max_grad_norm)
    )(grads)
    return jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), clipped_grads)


def noise_and_average_gradients(
    summed_grads: Dict,
    max_grad_norm: float,
    noise_multiplier: float,
    rng: jax.Array,
    batch_size: int
) -> Dict:
    """Add Gaussian noise to summed gradients and average.
    
    Args:
        summed_grads: Sum of clipped per-sample gradients
        max_grad_norm: Maximum gradient norm (for noise calibration)
        noise_multiplier: Noise multiplier (sigma)
        rng: JAX random key
        batch_size: Total batch size (for averaging)
    
    Returns:
        Noised and averaged gradients
    """
    def add_noise(g, key):
        noise_std = max_grad_norm * noise_multiplier
        return g + random.normal(key, g.shape) * noise_std
    
    num_leaves = len(jax.tree_util.tree_leaves(summed_grads))
    noise_keys = random.split(rng, num_leaves)
    
    noised_grads = jax.tree_util.tree_map(
        add_noise,
        summed_grads,
        jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(summed_grads),
            list(noise_keys)
        )
    )
    return jax.tree_util.tree_map(lambda x: x / batch_size, noised_grads)


def clip_and_noise_gradients(
    grads: Dict,
    max_grad_norm: float,
    noise_multiplier: float,
    rng: jax.Array,
    batch_size: int
) -> Dict:
    """Clip per-sample gradients, add Gaussian noise, and average.
    
    Convenience wrapper combining clip_and_sum_gradients + noise_and_average_gradients.
    """
    summed = clip_and_sum_gradients(grads, max_grad_norm)
    return noise_and_average_gradients(summed, max_grad_norm, noise_multiplier, rng, batch_size)


@partial(jit, static_argnums=(1, 4, 5))
def _compute_micro_batch_clipped_sum(
    params: Dict,
    apply_fn: Callable,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.Array,
    max_grad_norm: float,
    aug_multiplicity: int = 0
) -> Dict:
    """Process a micro-batch: compute per-sample gradients, clip, and sum.
    
    Used for gradient accumulation. Each micro-batch produces a sum of
    clipped per-sample gradients that can be accumulated across micro-batches.
    
    Args:
        params: Model parameters
        apply_fn: Model apply function
        batch: Tuple of (images, labels)
        rng: JAX random key
        max_grad_norm: Maximum gradient norm for clipping
        aug_multiplicity: Number of augmented copies (0 = no augmentation)
    
    Returns:
        Sum of clipped per-sample gradients for this micro-batch
    """
    images, labels = batch
    
    if aug_multiplicity and aug_multiplicity > 0:
        rng, aug_rng = random.split(rng)
        aug_rngs = random.split(aug_rng, images.shape[0])
        per_sample_grad_fn = grad(per_sample_loss_augmented)
        per_sample_grads = vmap(
            per_sample_grad_fn,
            in_axes=(None, None, 0, 0, 0, None)
        )(params, apply_fn, images, labels, aug_rngs, aug_multiplicity)
    else:
        per_sample_grad_fn = grad(per_sample_loss)
        per_sample_grads = vmap(
            per_sample_grad_fn,
            in_axes=(None, None, 0, 0)
        )(params, apply_fn, images, labels)
    
    return clip_and_sum_gradients(per_sample_grads, max_grad_norm)


@partial(jit, static_argnums=(1, 4, 5, 6))
def dp_train_step(
    state: train_state.TrainState,
    apply_fn: Callable,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.Array,
    max_grad_norm: float,
    noise_multiplier: float,
    aug_multiplicity: int = 0
) -> Tuple[train_state.TrainState, float]:
    """Single DP-SGD training step.
    
    Args:
        state: Current training state
        apply_fn: Model apply function
        batch: Tuple of (images, labels)
        rng: JAX random key
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Noise multiplier for DP
        aug_multiplicity: Number of augmented copies per sample (0 = no augmentation)
    
    Returns:
        Tuple of (new_state, loss)
    """
    images, labels = batch
    batch_size = images.shape[0]
    
    if aug_multiplicity and aug_multiplicity > 0:
        # Augmentation multiplicity: compute averaged gradient over K augmented
        # copies per sample before clipping (De et al. 2022)
        rng, aug_rng = random.split(rng)
        aug_rngs = random.split(aug_rng, batch_size)  # One key per sample
        
        per_sample_grad_fn = grad(per_sample_loss_augmented)
        per_sample_grads = vmap(
            per_sample_grad_fn,
            in_axes=(None, None, 0, 0, 0, None)
        )(state.params, apply_fn, images, labels, aug_rngs, aug_multiplicity)
    else:
        # Standard: no augmentation
        per_sample_grad_fn = grad(per_sample_loss)
        per_sample_grads = vmap(
            per_sample_grad_fn,
            in_axes=(None, None, 0, 0)
        )(state.params, apply_fn, images, labels)
    
    # Clip and add noise
    rng, noise_rng = random.split(rng)
    dp_grads = clip_and_noise_gradients(
        per_sample_grads,
        max_grad_norm,
        noise_multiplier,
        noise_rng,
        batch_size
    )
    
    # Update model
    new_state = state.apply_gradients(grads=dp_grads)
    
    # Compute loss for logging (on a subset to save memory)
    sample_logits = apply_fn(
        {'params': state.params},
        images[:32],
        train=False
    )
    sample_loss = optax.softmax_cross_entropy_with_integer_labels(
        sample_logits, labels[:32]
    ).mean()
    
    return new_state, sample_loss


def train_dpsgd_jax(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray],
    canary_images: np.ndarray,
    canary_labels: np.ndarray,
    in_mask: np.ndarray,
    target_epsilon: float,
    target_delta: float = 1e-5,
    num_epochs: int = 100,
    batch_size: int = 4096,
    learning_rate: float = 0.1,
    max_grad_norm: float = 1.0,
    noise_multiplier: Optional[float] = None,
    aug_multiplicity: int = 0,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[Dict, float, Dict]:
    """Train model with DP-SGD using JAX.
    
    Args:
        model: Flax model
        train_data: Tuple of (images, labels) for training set
        canary_images: Canary images to add to training
        canary_labels: Canary labels
        in_mask: Boolean mask for IN canaries
        target_epsilon: Target privacy budget
        target_delta: Target delta
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: If provided, use this directly instead of computing
            from target_epsilon (e.g., paper uses 1.75 with its own accounting)
        aug_multiplicity: Number of augmented copies per sample (0 = no augmentation,
            16 = paper default). Averages gradients over K augmented copies before
            clipping, improving signal-to-noise ratio at zero privacy cost.
        gradient_accumulation_steps: Number of micro-batches to split each batch into.
            Reduces peak GPU memory by processing smaller chunks. Must evenly divide
            batch_size. Privacy guarantee is unchanged.
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (trained_params, achieved_epsilon, training_stats)
    """
    rng = random.PRNGKey(seed)
    
    # Prepare training data with IN canaries
    train_images, train_labels = train_data
    in_canary_images = canary_images[in_mask]
    in_canary_labels = canary_labels[in_mask]
    
    all_images = np.concatenate([train_images, in_canary_images], axis=0)
    all_labels = np.concatenate([train_labels, in_canary_labels], axis=0)
    
    dataset_size = len(all_images)
    steps_per_epoch = dataset_size // batch_size
    total_steps = num_epochs * steps_per_epoch
    
    # Use provided noise_multiplier or compute from target epsilon
    if noise_multiplier is None:
        noise_multiplier = noise_multiplier_from_epsilon(
            target_epsilon, total_steps, batch_size, dataset_size, target_delta
        )
    
    if verbose:
        print(f"DP-SGD Training (JAX)")
        print(f"  Dataset size: {dataset_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Steps: {total_steps}")
        print(f"  Target epsilon: {target_epsilon}")
        print(f"  Noise multiplier: {noise_multiplier:.4f}")
        if aug_multiplicity and aug_multiplicity > 0:
            print(f"  Aug multiplicity: {aug_multiplicity}")
        if gradient_accumulation_steps > 1:
            print(f"  Grad accumulation steps: {gradient_accumulation_steps}")
            print(f"  Micro-batch size: {batch_size // gradient_accumulation_steps}")
    
    # Initialize model
    rng, init_rng = random.split(rng)
    state = create_dp_train_state(init_rng, model, learning_rate)
    
    # Training loop
    losses = []
    pbar = tqdm(range(num_epochs), disable=not verbose, desc="DP-SGD")
    
    for epoch in pbar:
        # Shuffle data
        rng, perm_rng = random.split(rng)
        perm = random.permutation(perm_rng, dataset_size)
        
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch_idx = perm[step * batch_size:(step + 1) * batch_size]
            batch_images = jnp.array(all_images[batch_idx])
            batch_labels = jnp.array(all_labels[batch_idx])
            
            if gradient_accumulation_steps > 1:
                # Gradient accumulation: process micro-batches, accumulate
                # clipped gradient sums, then add noise once
                micro_batch_size = batch_size // gradient_accumulation_steps
                accumulated_grads = None
                
                for mb_idx in range(gradient_accumulation_steps):
                    mb_start = mb_idx * micro_batch_size
                    mb_end = mb_start + micro_batch_size
                    mb_images = batch_images[mb_start:mb_end]
                    mb_labels = batch_labels[mb_start:mb_end]
                    
                    rng, mb_rng = random.split(rng)
                    clipped_sum = _compute_micro_batch_clipped_sum(
                        state.params, model.apply,
                        (mb_images, mb_labels),
                        mb_rng, max_grad_norm, aug_multiplicity
                    )
                    
                    if accumulated_grads is None:
                        accumulated_grads = clipped_sum
                    else:
                        accumulated_grads = jax.tree_util.tree_map(
                            lambda a, b: a + b,
                            accumulated_grads, clipped_sum
                        )
                
                # Add noise once and average over full batch
                rng, noise_rng = random.split(rng)
                dp_grads = noise_and_average_gradients(
                    accumulated_grads, max_grad_norm,
                    noise_multiplier, noise_rng, batch_size
                )
                state = state.apply_gradients(grads=dp_grads)
                
                # Compute loss for logging
                sample_logits = model.apply(
                    {'params': state.params},
                    batch_images[:32],
                    train=False
                )
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    sample_logits, batch_labels[:32]
                ).mean()
            else:
                # Standard single-step path (no accumulation)
                rng, step_rng = random.split(rng)
                state, loss = dp_train_step(
                    state, model.apply,
                    (batch_images, batch_labels),
                    step_rng, max_grad_norm, noise_multiplier,
                    aug_multiplicity
                )
            epoch_loss += loss
        
        avg_loss = epoch_loss / steps_per_epoch
        losses.append(float(avg_loss))
        
        if verbose:
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    # Compute achieved epsilon
    achieved_epsilon = compute_epsilon(
        total_steps, batch_size, dataset_size, noise_multiplier, target_delta
    )
    
    stats = {
        "losses": losses,
        "noise_multiplier": noise_multiplier,
        "achieved_epsilon": achieved_epsilon,
        "total_steps": total_steps,
    }
    
    return state.params, achieved_epsilon, stats


def evaluate_canary_losses(
    model: nn.Module,
    params: Dict,
    canary_images: np.ndarray,
    canary_labels: np.ndarray,
    in_mask: np.ndarray
) -> Dict:
    """Evaluate model losses on canaries.
    
    Args:
        model: Flax model
        params: Model parameters
        canary_images: All canary images
        canary_labels: All canary labels
        in_mask: Boolean mask for IN canaries
    
    Returns:
        Dict with IN/OUT losses and statistics
    """
    # Convert to JAX arrays
    images = jnp.array(canary_images)
    labels = jnp.array(canary_labels)
    
    # Forward pass
    logits = model.apply(
        {'params': params},
        images,
        train=False
    )
    
    # Compute per-sample losses
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    
    # Split by IN/OUT
    in_losses = losses[in_mask]
    out_losses = losses[~in_mask]
    
    return {
        "in_loss_mean": float(jnp.mean(in_losses)),
        "in_loss_std": float(jnp.std(in_losses)),
        "out_loss_mean": float(jnp.mean(out_losses)),
        "out_loss_std": float(jnp.std(out_losses)),
        "loss_gap": float(jnp.mean(in_losses) - jnp.mean(out_losses)),
    }

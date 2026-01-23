# rw_mh.py
"""
Random-Walk Metropolis–Hastings (RW-MH) kernel.

Minimal implementation intended for Section 4 of Lu (2025):
- Gaussian random-walk proposal
- Symmetric proposal (no proposal density correction)
- Stateless single-step update

This is designed to be used for low-dimensional continuous parameters
(e.g. r), with all conditioning handled outside this file.
"""

import numpy as np


def rw_mh_step(
    theta,
    logp_fn,
    proposal_cov,
    rng,
    current_logp=None,
    reject_if_nan=True,
):
    """
    Perform one Random-Walk Metropolis–Hastings step.

    Parameters
    ----------
    theta : np.ndarray, shape (d,)
        Current parameter vector.
    logp_fn : callable
        Function mapping theta -> log posterior (scalar).
        This should be the full conditional for theta.
    proposal_cov : np.ndarray
        Proposal covariance. Either:
        - shape (d,)  : interpreted as diagonal variances
        - shape (d,d): full covariance matrix
    rng : np.random.Generator
        NumPy random number generator.
    current_logp : float, optional
        Cached log posterior at theta. If None, it is recomputed.
    reject_if_nan : bool, default True
        Reject proposals with NaN or infinite log posterior.

    Returns
    -------
    theta_new : np.ndarray, shape (d,)
        Updated parameter vector (accepted or original).
    accepted : bool
        Whether the proposal was accepted.
    logp_new : float
        Log posterior at theta_new.
    """
    theta = np.asarray(theta)

    if current_logp is None:
        logp_curr = logp_fn(theta)
    else:
        logp_curr = current_logp

    # Draw proposal noise
    if proposal_cov.ndim == 1:
        eps = rng.normal(scale=np.sqrt(proposal_cov), size=theta.shape)
    else:
        eps = rng.multivariate_normal(
            mean=np.zeros_like(theta),
            cov=proposal_cov,
        )

    theta_prop = theta + eps
    logp_prop = logp_fn(theta_prop)

    # Handle invalid proposals
    if reject_if_nan:
        if not np.isfinite(logp_prop):
            return theta, False, logp_curr

    # MH acceptance step (symmetric proposal)
    log_alpha = logp_prop - logp_curr
    if np.log(rng.uniform()) < log_alpha:
        return theta_prop, True, logp_prop
    else:
        return theta, False, logp_curr

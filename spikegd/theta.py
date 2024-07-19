r"""
Oscillatory QIF (theta) neuron with delta-pulse coupling.

    $$\tau \dot{V} = V(V-1) + I_0 + \tau \sum_i w_i \sum_{t_i} \delta(t-t_i)$$
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from spikegd.models import AbstractPseudoPhaseOscNeuron


@dataclass(frozen=True)
class ThetaNeuron(AbstractPseudoPhaseOscNeuron):
    r"""
    Oscillatory QIF (theta) neuron with delta-pulse coupling.

        $$\tau \dot{V} = V(V-1) + I_0 + \tau \sum_i w_i \sum_{t_i} \delta(t-t_i)$$

    Attributes:
        tau: Time constant.
        I0: Current.
        eps: Small value for numerical reasons.
    """

    tau: float = 1.0
    I0: float = 5 / 4
    eps: float = 1e-6

    def __post_init__(self):
        assert self.tau > 0, "`tau` must be positive."
        assert self.I0 > 1 / 4, "`I0` must be greater than 1/4."
        assert self.eps > 0, "`eps` must be positive."

    def Phi(self, V: ArrayLike) -> Array:
        root = jnp.sqrt(self.I0 - 1 / 4)
        return self.tau / root * (jnp.arctan((V - 1 / 2) / root) + jnp.pi / 2)

    def iPhi(self, phi: ArrayLike) -> Array:
        root = jnp.sqrt(self.I0 - 1 / 4)
        V = root * jnp.tan(phi * root / self.tau - jnp.pi / 2) + 1 / 2
        return jnp.where(
            (phi > self.eps) & (phi < self.Theta() - self.eps),
            V,
            jnp.sign(phi - self.Theta() / 2) / self.eps,
        )

    def H(self, phi: ArrayLike, w: ArrayLike) -> Array:
        phi = jnp.asarray(phi)
        V = self.iPhi(phi) + w
        phinew = self.Phi(V)
        return jnp.where(
            (phi > self.eps) & (phi < self.Theta() - self.eps), phinew, phi
        )

    def Theta(self) -> Array:
        return self.tau * jnp.pi / jnp.sqrt(self.I0 - 1 / 4)

    def flow(self, x: Array, dt: ArrayLike) -> Array:
        return x + dt

    def dt_spike(self, x: Array) -> Array:
        phi = x[0]
        return self.Theta() - phi

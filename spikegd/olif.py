r"""
Oscillatory LIF neuron with delta-pulse coupling.

    $$\tau \dot{V} = -V + I_0 + \tau \sum_i w_i \sum_{t_i} \delta(t-t_i)$$
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from spikegd.models import AbstractPseudoPhaseOscNeuron


@dataclass(frozen=True)
class OscLIFNeuron(AbstractPseudoPhaseOscNeuron):
    r"""
    Oscillatory LIF neuron with delta-pulse coupling.

        $$\tau \dot{V} = -V + I_0 + \tau \sum_i w_i \sum_{t_i} \delta(t-t_i)$$

    Attributes:
        tau: Time constant.
        I0: Constant input current.
        V_th: Threshold voltage.
    """

    tau: float = 1.0
    I0: float = 5 / 4
    V_th: float = 1.0

    def __post_init__(self):
        assert self.tau > 0, "`tau` must be positive."
        assert self.V_th > 0, "`V_th` must be positive."
        assert self.I0 > self.V_th, "`I0` must be greater than `V_th`."
        object.__setattr__(self, "theta", float(self.Theta()))

    def Phi(self, V: ArrayLike) -> Array:
        return self.tau * jnp.log(1 / (1 - V / self.I0))

    def iPhi(self, phi: ArrayLike) -> Array:
        phi = jnp.asarray(phi)
        return self.I0 * (1 - jnp.exp(-phi / self.tau))

    def H(self, phi: ArrayLike, w: ArrayLike) -> Array:
        V = self.iPhi(phi) + w
        return self.Phi(jnp.where(V < self.V_th, V, self.V_th))

    def Theta(self) -> Array:
        return self.Phi(self.V_th)

    def flow(self, x: Array, dt: ArrayLike) -> Array:
        return x + dt

    def dt_spike(self, x: Array) -> Array:
        phi = x[0]
        return self.Theta() - phi

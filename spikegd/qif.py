r"""
QIF (theta) neuron with temporally extended coupling and $\tau=\tau_m = 2\tau_s$.

    $$\tau_m \dot{V} = V (V-1) + I$$
    $$\tau_s \dot{I} = -I + \tau_s \sum_i w_i \sum_{t_i} \delta(t-t_i)$$
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int

from spikegd.models import AbstractPseudoIFNeuron

# %%
############################
### Ordinary dynamics
############################


@dataclass(frozen=True)
class QIFNeuron(AbstractPseudoIFNeuron):
    r"""
    QIF (theta) neuron with temporally extended coupling and $\tau=\tau_m = 2\tau_s$.

        $$\tau_m \dot{V} = V (V-1) + I$$
        $$\tau_s \dot{I} = -I + \tau_s \sum_i w_i \sum_{t_i} \delta(t-t_i)$$

    Attributes:
        tau: Time constant.
        eps: Small value for numerical reasons.
        alpha: Scaling factor for softplus necessary for pseudospike time computation.
    """

    tau: float = 1.0
    eps: float = 1e-6
    alpha: float = 10.0

    def __post_init__(self):
        assert self.tau > 0, "`tau` must be positive."
        assert self.eps > 0, "`eps` must be positive."
        assert self.alpha > 0, "`alpha` must be positive."

    ### Ordinary dynamics
    def flow_V(self, V: ArrayLike, I: ArrayLike, dt: ArrayLike) -> Array:  # noqa: E741
        """
        Voltage component of flow.
        """
        # To avoid nans during reverse differentiation, sometimes variables X_cond are
        # introduced. See also:
        # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

        ### Intermediate terms
        dt = jnp.asarray(dt)
        exp = jnp.exp(-dt / self.tau)
        I = jnp.asarray(I)  # noqa: E741
        # I_cond = jnp.where(I != 0, I, 1)
        I_cond = jnp.where(I != 0, I, I + 1e-6)
        I_sqrt = jnp.sqrt(abs(I_cond))
        V = jnp.asarray(V)
        Vabs = jnp.abs(V)

        ### I=0
        cond1 = I == 0
        V1 = V / (V - (V - 1) / exp)

        ### I>0
        cond2 = I > 0
        V2 = I_sqrt * exp * jnp.tan(jnp.arctan(V / I_sqrt) + I_sqrt * (1 - exp))
        # To estimate gradient
        V1 = jax.lax.stop_gradient(V1) + V2 - jax.lax.stop_gradient(V2)

        ### I<0 & abs(V)==I_sqrt
        cond3 = (I < 0) & (Vabs == I_sqrt)
        # cond3 = (I < 0) & (abs(Vabs - I_sqrt) < 1e-6)
        V3 = jnp.sign(V) * I_sqrt * exp

        ### I<0 & abs(V)>I_sqrt
        cond4 = (I < 0) & (Vabs > I_sqrt)
        V_cond = jnp.where(cond4, V, 2 * I_sqrt)
        V_cond = jnp.where(cond3, V + 1e-6, V_cond)
        V4 = I_sqrt * exp / jnp.tanh(jnp.arctanh(I_sqrt / V_cond) - I_sqrt * (1 - exp))
        # To estimate gradient
        V3 = jax.lax.stop_gradient(V3) + V4 - jax.lax.stop_gradient(V4)

        ### I<0 & abs(V)<I_sqrt
        cond5 = (I < 0) & (Vabs < I_sqrt)
        V_cond = jnp.where(cond5, V, 0)
        V5 = I_sqrt * exp * jnp.tanh(jnp.arctanh(V_cond / I_sqrt) - I_sqrt * (1 - exp))

        ### Combine
        conds = [cond1, cond2, cond3, cond4, cond5]
        Vs = [V1, V2, V3, V4, V5]
        V_new = jnp.select(conds, Vs)
        V_new = jnp.minimum(V_new, 1 / self.eps)  # avoids numerical issues

        return jnp.where(dt > 0, V_new, V)

    def flow_I(self, I: ArrayLike, dt: ArrayLike) -> Array:  # noqa: E741
        """
        Current component of flow.
        """
        dt = jnp.asarray(dt)
        I = jnp.asarray(I)  # noqa: E741
        return I * jnp.exp(-2 * dt / self.tau)

    def flow(
        self, x: Float[Array, " D *N"], dt: Float[ArrayLike, ""]
    ) -> Float[Array, " D *N"]:
        V, I = x[0], x[1]  # noqa: E741
        V = self.flow_V(V, I, dt)
        I = self.flow_I(I, dt)  # noqa: E741
        return jnp.stack([V, I])

    def dt_spike(self, x: Float[Array, " D *N"]) -> Float[Array, " *N"]:
        # To avoid nans during reverse differentiation, sometimes variables X_cond are
        # introduced. See also:
        # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

        ### Compute intermediate terms
        V, I = x[0], x[1]  # noqa: E741
        # I_cond = jnp.where(I != 0, I, 1)
        I_cond = jnp.where(I != 0, I, I + 1e-6)
        I_sqrt = jnp.sqrt(abs(I_cond))

        ### I=0 & V>Vsep=1
        cond1 = (I == 0) & (V > 1)
        V_cond = jnp.where(cond1, V, 2)
        dt1 = self.tau * jnp.log(V_cond / (V_cond - 1))

        ### I>0 & I_sqrt + arctan(V/I_sqrt) > pi/2
        cond2 = (I > 0) & (I_sqrt + jnp.arctan(V / I_sqrt) > jnp.pi / 2)
        V_cond = jnp.where(cond2, V, 1 / self.eps)
        V_cond = jnp.where(cond1, V, V_cond)
        dt2 = -self.tau * jnp.log(
            1 - jnp.pi / (2 * I_sqrt) + 1 / I_sqrt * jnp.arctan(V_cond / I_sqrt)
        )
        # To estimate gradient
        dt1 = jax.lax.stop_gradient(dt1) + dt2 - jax.lax.stop_gradient(dt2)

        ### I<0 & V>I_sqrt & arctanh(I_sqrt/V) < I_sqrt
        cond3 = (I < 0) & (V > I_sqrt) & (V * jnp.tanh(I_sqrt) > I_sqrt)
        V_cond = jnp.where(cond3, V, -2 * I_sqrt)
        dt3 = -self.tau * jnp.log(1 - jnp.arctanh(I_sqrt / V_cond) / I_sqrt)

        ### no next spike
        cond4 = ~(cond1 | cond2 | cond3)
        dt4 = jnp.inf

        ### Combine
        conds = [cond1, cond2, cond3, cond4]
        dts = [dt1, dt2, dt3, dt4]
        dt = jnp.select(conds, dts)

        return dt

    def spike(
        self,
        state: tuple[
            Float[Array, ""], Float[Array, "2 N"], Float[Array, " N"], Int[Array, ""]
        ],
        weights_net: Float[Array, "N N"],
        weights_in: Float[Array, "N Nin"],
        spikes_in: tuple[Float[Array, " Kin"], Int[Array, " Kin"]],
        config: dict,
    ) -> tuple[tuple, tuple]:
        ### Unpack arguments
        T: float = config["T"]
        t, x, dt_net, counter_in = state
        V, I = x[0], x[1]  # noqa: E741

        ### Find next event
        j_net = jnp.argmin(dt_net)
        dt_net_j = dt_net[j_net]
        dt_in, j_in = spikes_in[0][counter_in] - t, spikes_in[1][counter_in]
        spike_in = dt_in < dt_net_j
        j = jnp.where(spike_in, j_in, j_net)
        dt = jnp.where(spike_in, dt_in, dt_net_j)
        spike_trial = dt < T - t
        dt = jnp.where(spike_trial, dt, T - t)
        j = jnp.where(spike_trial, j, -1)

        ### Evolve network and input
        t = t + dt
        # State of neurons close to spiking not updated to avoid numerical issues
        mask = abs(dt_net - dt) < self.eps
        V = jnp.where(mask, 0.0, V)
        x = self.flow(jnp.stack((V, I)), dt)  # noqa: E741
        V, I = x[0], x[1]  # noqa: E741
        V = jnp.where(mask, 1 / self.eps, V)
        counter_in = counter_in + spike_in

        ### Transmit spike
        weight_j = jnp.where(spike_in, weights_in[:, j_in], weights_net[:, j_net])
        I += weight_j * spike_trial  # noqa: E741

        ### Reset spiking neuron
        # Reset to -1/eps to avoid numerical issues
        V = V.at[j_net].set(
            jnp.where(spike_in | (~spike_trial), V[j_net], -1 / self.eps)
        )
        x = jnp.stack((V, I))

        ### Update next spike time
        dt_net = jnp.where(mask, dt_net - dt, self.dt_spike(x))
        dt_net = dt_net.at[j_net].set(
            jnp.where(spike_in, dt_net[j_net], self.dt_spike(x[:, j_net]))
        )

        ### Pack state and output
        state = (t, x, dt_net, counter_in)
        out = (t, spike_in, j, x)

        return state, out

    def min_I0_to_spike(self) -> float:
        return jnp.pi**2 / 4

    ### Pseudodynamics
    def softplus(self, I_pseudo: ArrayLike) -> Array:
        """
        Computes suprathreshold current for pseudodynamics.
        """
        I_pseudo = jnp.asarray(I_pseudo)
        return self.alpha * jnp.log(1 + jnp.exp(I_pseudo / self.alpha))

    def phi_pseudo(self, V: ArrayLike, I_pseudo: ArrayLike) -> Array:
        """
        Computes pseudo phase from voltage and pseudo input current.
        """
        root = jnp.sqrt(self.softplus(I_pseudo))
        return self.tau * (jnp.arctan((V - 1 / 2) / root) + jnp.pi / 2) / root

    def theta_pseudo(self, I_pseudo: ArrayLike) -> Array:
        """
        Computes pseudo period from pseudo input current.
        """
        root = jnp.sqrt(self.softplus(I_pseudo))
        return self.tau * jnp.pi / root

    def construct_ratefn(self, x: Array) -> Callable:
        V, I = x[0], x[1]  # noqa: E741

        def ratefn(input: ArrayLike):
            I_pseudo = I + input
            return self.phi_pseudo(V, I_pseudo) / self.theta_pseudo(I_pseudo)

        return ratefn

    def linear(self, r: ArrayLike, w: ArrayLike) -> Array:
        return jnp.dot(w, r)

    def t_pseudo(self, x: Array, input: ArrayLike, k: ArrayLike, config: dict) -> Array:
        T = config["T"]
        V, I = x[0], x[1]  # noqa: E741
        I_pseudo = I + input
        return T + k * self.theta_pseudo(I_pseudo) - self.phi_pseudo(V, I_pseudo)

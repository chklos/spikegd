r"""
LIF neuron with temporally extended coupling and $\tau=\tau_m = 2\tau_s$.

    $$\tau_m \dot{V} = -V + I$$
    $$\tau_s \dot{I} = -I + \tau_s \sum_i w_i \sum_{t_i} \delta(t-t_i)$$
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int

from spikegd.models import AbstractIFNeuron


@dataclass(frozen=True)
class LIFNeuron(AbstractIFNeuron):
    r"""
    LIF neuron with temporally extended coupling and $\tau=\tau_m = 2\tau_s$.

        $$\tau_m \dot{V} = -V + I$$
        $$\tau_s \dot{I} = -I + \tau_s \sum_i w_i \sum_{t_i} \delta(t-t_i)$$

    Attributes:
        tau: Time constant.
        V_th: Threshold voltage.
    """

    tau: float = 1.0
    V_th: float = 1.0

    def __post_init__(self):
        assert self.tau > 0, "`tau` must be positive."
        assert self.V_th > 0, "`V_th` must be positive."

    def flow_V(self, V: ArrayLike, I: ArrayLike, dt: ArrayLike) -> Array:  # noqa: E741
        """
        Voltage component of flow.
        """
        V = jnp.asarray(V)
        I = jnp.asarray(I)  # noqa: E741
        dt = jnp.asarray(dt)
        return V * jnp.exp(-dt / self.tau) + I * (
            jnp.exp(-dt / self.tau) - jnp.exp(-2 * dt / self.tau)
        )

    def flow_I(self, I: ArrayLike, dt: ArrayLike) -> Array:  # noqa: E741
        """
        Current component of flow.
        """
        I = jnp.asarray(I)  # noqa: E741
        dt = jnp.asarray(dt)
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
        I_cond = jnp.where(I > 0, I, 1)
        root_arg = (V + I_cond) ** 2 - 4 * I_cond * self.V_th
        root_arg_cond = jnp.where(root_arg > 0, root_arg, 0)
        log_arg = (V + I_cond + jnp.sqrt(root_arg_cond)) / (2 * I_cond)
        log_arg_cond = jnp.where((log_arg > 0) & (log_arg < 1), log_arg, 1)

        ### Combine
        dt = -self.tau * jnp.log(log_arg_cond)
        dt = jnp.where(
            (I > 0) & (root_arg > 0) & (log_arg > 0) & (log_arg < 1), dt, jnp.inf
        )

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
        x = self.flow(x, dt)  # noqa: E741
        V, I = x[0], x[1]  # noqa: E741
        counter_in = counter_in + spike_in

        ### Transmit spike
        weight_j = jnp.where(spike_in, weights_in[:, j_in], weights_net[:, j_net])
        I += weight_j  # noqa: E741

        ### Reset spiking neuron
        V = V.at[j_net].set(jnp.where(spike_in, V[j_net], 0.0))
        x = jnp.stack((V, I))

        ### Update next spike time
        dt_net = self.dt_spike(x)

        ### Pack state and output
        state = (t, x, dt_net, counter_in)
        out = (t, spike_in, j, x)

        return state, out

    def min_I0_to_spike(self) -> float:
        return 4 * self.V_th

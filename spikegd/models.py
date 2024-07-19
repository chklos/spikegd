"""
Abstract neuron models.

Templates for neuron models to be used in event-based simulations.
Currently this includes integrate-and-fire (IF) neurons with temporally extended,
current based inputs, and phase oscillator models with delta-pulse coupling.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, ArrayLike, Float, Int

# %%
############################
### Neuron model template
############################


class AbstractNeuron(ABC):
    """
    Neuron model to be used in event-based simulations.
    """

    @abstractmethod
    def spike(
        self,
        state: tuple[
            Float[Array, ""], Float[Array, "D N"], Float[Array, " N"], Int[Array, ""]
        ],
        weights_net: Float[Array, "N N"],
        weights_in: Float[Array, "N Nin"],
        spikes_in: tuple[Float[Array, " Kin"], Int[Array, " Kin"]],
        config: dict,
    ) -> tuple[tuple, tuple]:
        """
        Evolves neural network until next spike.

        Args:
            state:
                Current state of the simulation. Tuple containing time, neuronal state,
                time until next spike, and input counter.
            weights_net:
                Network weights.
            weights_in:
                Input weights.
            spikes_in:
                Input queue. First array contains spike times, second array contains
                indices.
            config:
                Simulation configuration. Needs to contain the following items:
                    `T`: Trial duration

        Returns:
            Updated simulation state and output. The output is a tuple containing the
            time of the next spike, a boolean indicating whether the spike was an input
            spike, the index of the spiking neuron, and the updated neuronal state.
        """

    @abstractmethod
    def dt_spike(self, x: Float[Array, " D *N"]) -> Float[Array, " *N"]:
        """
        Computes time until next spike.
        """

    @abstractmethod
    def flow(
        self, x: Float[Array, " D *N"], dt: Float[ArrayLike, ""]
    ) -> Float[Array, " D *N"]:
        """
        Evolves state of `N` single neurons by `dt`.
        """

    def event(
        self,
        x0: Float[Array, "D N"],
        weights_net: Float[Array, "N N"],
        weights_in: Float[Array, "N Nin"] | None,
        spikes_in: tuple[Float[Array, " Kin"], Int[Array, " Kin"]] | None,
        config: dict,
    ) -> tuple[Array, Array, Array, Array]:
        """
        Simulates a neural network in an event-based fashion.

        Args:
            x0:
                Initial state of the neurons.
            weights_net:
                Network weights.
            weights_in:
                Input weights.
            spikes_in:
                Input queue. First array contains spike times, second array contains
                indices.
            config:
                Simulation configuration. Needs to contain the following items:
                    `K`: Maximal number of simulated ordinary spikes
                    `T`: Trial duration

        Returns:
            Tuple containing the times of the spikes, a boolean array indicating whether
            the spike was an input spike, the indices of the spiking neurons, and the
            neuronal states.
        """

        ### Unpack arguments
        K = config["K"]
        T = config["T"]

        ### Initialize simulation state and input queue
        t = jnp.array(0.0)
        counter_in = jnp.array(0)
        dt_net = self.dt_spike(x0)
        state = (t, x0, dt_net, counter_in)
        if weights_in is None:
            weights_in = jnp.zeros((weights_net.shape[0], 1))
        if spikes_in is None:
            spikes_in = (jnp.array([jnp.inf]), jnp.array([0]))
        else:
            order = jnp.argsort(spikes_in[0])
            spikes_in = (spikes_in[0][order], spikes_in[1][order])
            spikes_in = (jnp.append(spikes_in[0], jnp.inf), jnp.append(spikes_in[1], 0))

        ### Run simulation
        def f(state: tuple, k: Array) -> tuple:
            return self.spike(state, weights_net, weights_in, spikes_in, config)

        _, out = lax.scan(f, state, jnp.arange(K))

        # Check if enough spikes have been simulated to reach trial end.
        # Can be used to raise a warning, which cannot be done in a jitted function.
        ts = out[0]
        ts = lax.cond(
            ts[-1] < T,
            lambda _: jnp.full_like(ts, jnp.nan),
            lambda _: ts,
            _,
        )
        out = (ts, *out[1:])

        return out

    def traces(
        self,
        x0: Float[Array, "D N"],
        out: tuple[Array, Array, Array, Array],
        config: dict,
    ) -> tuple[Array, Array]:
        """
        Computes state traces.

        Args:
            x0:
                Initial state of the neurons.
            out:
                Output of the event-based simulation.
            config:
                Simulation configuration. Needs to contain the following items:
                    `dt`: Integration time step

        Returns:
            Tuple containing the times and states.
        """

        ### Unpack arguments
        dt = config["dt"]
        ts, _, js, xs = out
        spikes_in_trial = jnp.sum(js >= 0)

        ### Times
        ts = jnp.concatenate((jnp.array([0.0]), ts[: spikes_in_trial + 1]))
        dts = jnp.diff(ts)
        steps_per_spike = jnp.ceil(dts / dt).astype(int)
        max_steps_per_spike = int(jnp.max(steps_per_spike))
        dts_per_spike = jnp.linspace(
            0, max_steps_per_spike * dt, max_steps_per_spike + 1
        )
        trace_ts = jnp.concatenate(
            [ts[i] + dts_per_spike[:steps] for i, steps in enumerate(steps_per_spike)]
        )

        ### States
        xs = jnp.concatenate((x0[jnp.newaxis], xs[: spikes_in_trial + 1]))
        trace_xs = vmap(lambda dt: self.flow(jnp.swapaxes(xs, 0, 1), dt))(dts_per_spike)
        trace_xs = jnp.concatenate(
            [trace_xs[:steps, :, i] for i, steps in enumerate(steps_per_spike)]
        )

        return trace_ts, trace_xs


# %%
############################
### IF neuron template
############################


class AbstractIFNeuron(AbstractNeuron):
    """
    Integrate-and-fire neuron model with temporally extended, current-based inputs.
    """

    @abstractmethod
    def min_I0_to_spike(self) -> float:
        """
        Minimum of `I0` such that a neuron spikes assuming `V0=V_rest`.
        """


class AbstractPseudoIFNeuron(AbstractIFNeuron):
    """
    Integrate-and-fire neuron model including pseudodynamics.
    """

    @abstractmethod
    def construct_ratefn(self, x: Float[Array, " D *N"]) -> Callable:
        """
        Constructs 'output rate' functions, as used in rate ANNs.

        Necessary to compute pseudospike times. See article for details.
        """

    def linear(self, r: ArrayLike, w: ArrayLike) -> Array:
        """
        Computes preactivation/summed input like in rate ANN.
        """
        return jnp.dot(w, r)

    @abstractmethod
    def t_pseudo(
        self,
        x: Float[Array, " D *N"],
        input: Float[Array, " *N"],
        k: ArrayLike,
        config: dict,
    ) -> Array:
        """
        Computes pseudospike times.

        Args:
            x:
                Neuronal state at the tial end.
            input:
                Input, like in a rate ANN, to each neuron.
            k:
                Pseudospike index. Can be a single value or an array.
            config:
                Simulation configuration. Needs to contain the following items:
                    `T`: Trial duration

        Returns:
            Pseudospike times.
        """


# %%
############################
### Phase oscillator template
############################


class AbstractPhaseOscNeuron(AbstractNeuron):
    """
    Phase oscillator model with delta-pulse coupling.
    """

    @abstractmethod
    def Phi(self, V: ArrayLike) -> Array:
        """
        Transforms `V` to `phi`.
        """

    @abstractmethod
    def iPhi(self, phi: ArrayLike) -> Array:
        """
        Transforms `phi` to `V`.
        """

    @abstractmethod
    def H(self, phi: ArrayLike, w: ArrayLike) -> Array:
        """
        Phase transfer function.
        """

    @abstractmethod
    def Theta(self) -> Array:
        """
        Threshold in phase representation. Equal to free oscillation period.
        """

    def spike(
        self,
        state: tuple[
            Float[Array, ""], Float[Array, "1 N"], Float[Array, " N"], Int[Array, ""]
        ],
        weights_net: Float[Array, "N N"],
        weights_in: Float[Array, "N Nin"],
        spikes_in: tuple[Float[Array, " Kin"], Int[Array, " Kin"]],
        config: dict,
    ) -> tuple[tuple, tuple]:
        ### Unpack arguments
        T: float = config["T"]
        t, x, _, counter_in = state
        phi = x[0]

        ### Find next event
        j_net = jnp.argmax(phi)
        dt_net = self.dt_spike(phi[jnp.newaxis, j_net])
        dt_in, j_in = spikes_in[0][counter_in] - t, spikes_in[1][counter_in]
        spike_in = dt_in < dt_net
        j = jnp.where(spike_in, j_in, j_net)
        dt = jnp.where(spike_in, dt_in, dt_net)
        spike_trial = dt < T - t
        dt = jnp.where(spike_trial, dt, T - t)
        j = jnp.where(spike_trial, j, -1)

        ### Evolve network and input
        t = t + dt
        phi = self.flow(phi, dt)
        counter_in = counter_in + spike_in

        ### Transmit spike
        weight_j = jnp.where(spike_in, weights_in[:, j_in], weights_net[:, j_net])
        phi = self.H(phi, weight_j * spike_trial)

        ### Reset spiking neuron
        phi = phi.at[j_net].set(jnp.where(spike_in | (~spike_trial), phi[j_net], 0.0))

        ### Pack state and output
        x = phi[jnp.newaxis]
        state = (t, x, _, counter_in)
        out = (t, spike_in, j, x)

        return state, out


class AbstractPseudoPhaseOscNeuron(AbstractPhaseOscNeuron):
    """
    Phase oscillator model with delta-pulse coupling including pseudodynamics.
    """

    def construct_ratefn(self, x: Float[Array, " D *N"]) -> Callable:
        """
        Constructs 'output rate' functions, as used in rate ANNs.

        Necessary to compute pseudospike times. See article for details.
        """
        phi = x[0]

        def ratefn(input: ArrayLike):
            return self.H(phi, input) / self.Theta()

        return ratefn

    def linear(self, r: ArrayLike, w: ArrayLike) -> Array:
        """
        Computes preactivation/summed input like in rate ANN.
        """
        return jnp.dot(w, r)

    def t_pseudo(
        self,
        x: Float[Array, " D *N"],
        input: Float[Array, " *N"],
        k: ArrayLike,
        config: dict,
    ) -> Array:
        """
        Computes pseudospike times.

        Args:
            x:
                Neuronal state at the tial end.
            input:
                Input, like in a rate ANN, to each neuron.
            k:
                Pseudospike index. Can be a single value or an array.
            config:
                Simulation configuration. Needs to contain the following items:
                    `T`: Trial duration

        Returns:
            Pseudospike times.
        """
        T = config["T"]
        phi = x[0]
        return T + k * self.Theta() - self.H(phi, input)

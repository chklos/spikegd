from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from jaxtyping import Array, ArrayLike, Float, Int
from matplotlib.axes import Axes

from spikegd.models import AbstractIFNeuron
from spikegd.utils.plotting import cmap_blues, cmap_grays

# %%
############################
### Simulation
############################


def run(
    neuron: AbstractIFNeuron,
    spikes_in: tuple[Float[Array, " Kin"], Int[Array, " Kin"]],
    weights_in: Float[Array, "1 Nin"],
    p_grid: Float[Array, " Nvar"],
    p_ex: Float[Array, " Nex"],
    vary: str,
    config: dict,
) -> tuple[dict, dict]:
    """
    Simulates a single neuron receiving input spikes with specified parameters.

    On the parameter grid, spike times and derivatives are determined.
    For the example parameters, additionally state traces are determined.

    Args:
        neuron:
            Integrate-and-fire neuron model.
        spikes_in:
            Input queue. First array contains spike times, second array contains
            indices.
        weights_in:
            Input weights.
        p_grid:
            Parameter grid.
        p_ex:
            Example parameter values for which state traces are determined.
        vary:
            Parameter to vary. Either 'weight' from the first input neuron
            or 'time' of the input spike first in the input queue.
        config:
            Simulation configuration. Needs to contain the following items:
                `K`: Maximal number of simulated ordinary spikes
                `T`: Trial duration
                `dt`: Integration time step (for state traces)

    Returns:
        A tuple containing the results for the grid and examples.
        Each stored as a dictionary.
    """

    ### Set up the simulation

    # Simulation

    if vary == "weight":

        @jit
        def simulate(p: ArrayLike) -> tuple:
            weights_in_p = weights_in.at[0, 0].set(p)
            out = neuron.event(x0, weights_net, weights_in_p, spikes_in, config)
            return out
    elif vary == "time":

        @jit
        def simulate(p: ArrayLike) -> tuple:
            times_in = spikes_in[0].at[0].set(p)
            order = jnp.argsort(times_in)
            spikes_in_p = (times_in[order], spikes_in[1][order])
            out = neuron.event(x0, weights_net, weights_in, spikes_in_p, config)
            return out
    else:
        raise ValueError(f"`vary` must be either 'weight' or 'time', not {vary}")

    # Output spike times
    @jit
    def outfn(out: tuple) -> Array:
        times: Array = out[0]
        spike_in: Array = out[1]
        neurons: Array = out[2]
        Kout = jnp.sum(
            (neurons == 0) & (spike_in == False)  # noqa: E712
        )  # Number of output spikes
        t_out = times[jnp.argsort(spike_in)[0]]  # Potential output spike time
        return jnp.where(Kout > 0, t_out, jnp.inf)

    # Gradient
    @jit
    @partial(value_and_grad, has_aux=True)
    def gradfn(p: ArrayLike) -> tuple[Array, tuple]:
        out = simulate(p)
        t_out = outfn(out)
        return t_out, out

    # Traces
    def tracefn(p: ArrayLike) -> tuple[Array, Array, tuple]:
        (t_out, out), grad = gradfn(p)
        traces = neuron.traces(x0, out, config)
        return t_out, grad, traces

    ### Simulation

    # Initialize
    V0, I0 = 0.0, 0.0
    x0 = jnp.array([[V0], [I0]])
    weights_net = jnp.zeros((1, 1))  # No self-connection

    # Traverse parameter grid
    (spiketimes, _), grads = vmap(gradfn)(p_grid)
    grid = {"p": p_grid, "spiketimes": spiketimes, "grads": grads}
    if jnp.any(jnp.isnan(spiketimes)):
        print(
            "Warning: A NaN appeared. "
            "Likely not enough spikes have been simulated. "
            "Try increasing `K`."
        )

    # Run examples
    examples = [tracefn(p) for p in p_ex]
    examples = {
        "p": p_ex,
        "spiketimes": jnp.array([ex[0] for ex in examples]),
        "grads": jnp.array([ex[1] for ex in examples]),
        "traces": [ex[2] for ex in examples],
    }

    return grid, examples


def vary_w(
    neuron: AbstractIFNeuron,
    w_ex_relative: Float[Array, " Nex"],
    config: dict,
    Nvar: int = 100_000,
) -> tuple:
    """
    Simulates a single neuron receiving an excitatory input with varying weights.

    On the parameter grid, spike times and derivatives are determined.
    For example parameters, additionally state traces are determined.

    Args:
        neuron:
            Integrate-and-fire neuron model.
        w_ex_relative:
            Scaling factor for the excitatory input weight used for examples.
        config:
            Simulation configuration. Needs to contain the following items:
                `K`: Maximal number of simulated ordinary spikes
                `T`: Trial duration
                `dt`: Integration time step (for state traces)
        Nvar:
            Number of points on the weight grid.

    Returns:
        A tuple containing the results for the grid and examples.
        Each stored as a dictionary.
    """

    ### Input
    spikes_in = (jnp.array([0.5]), jnp.array([0]))
    weights_in = jnp.array([[0.0]])

    ### Input weight grid and example weights
    w_min = neuron.min_I0_to_spike()  # Min input weight for a spike
    w_grid = jnp.linspace(0, 2 * w_min, Nvar + 1)
    w_ex = w_ex_relative * w_min

    ### Run
    grid, examples = run(neuron, spikes_in, weights_in, w_grid, w_ex, "weight", config)

    return grid, examples


def vary_t(
    neuron: AbstractIFNeuron,
    t_ex_relative: Float[Array, " Nex"],
    config: dict,
    w_exc_relative: float = 1.5,
    Nvar: int = 100_000,
) -> tuple:
    """
    Simulates a single neuron receiving a fixed excitatory input and an
    inhibitory input with varying spike time.

    On the parameter grid, spike times and derivatives are determined.
    For example parameters, additionally state traces are determined.

    Args:
        neuron:
            Integrate-and-fire neuron model.
        t_ex_relative:
            Scaling factor for the input time used for examples.
        config:
            Simulation configuration. Needs to contain the following items:
                `K`: Maximal number of simulated ordinary spikes
                `T`: Trial duration
                `dt`: Integration time step (for state traces)
        w_exc_relative:
            Scaling factor for the excitatory input weight.
        Nvar:
            Number of points on the time grid.

    Returns:
        A tuple containing the results for the grid and examples.
        Each stored as a dictionary.
    """

    ### Unpack argumetns
    T = config["T"]

    ### Input
    w_min = neuron.min_I0_to_spike()
    w_exc = w_exc_relative * w_min
    spikes_in = (jnp.array([0.0, 0.5]), jnp.array([0, 1]))
    weights_in = jnp.array([[-w_min, w_exc]])

    ### Input time grid and example times
    # Spike time for excitatory input only
    t_exc_only = neuron.dt_spike(jnp.array([0, w_exc])) + 0.5
    t_grid = jnp.linspace(0, T, Nvar + 1)
    t_ex = t_ex_relative * t_exc_only

    ### Run
    grid, examples = run(neuron, spikes_in, weights_in, t_grid, t_ex, "time", config)

    return grid, examples


# %%
############################
### Plotting
############################


def idx_to_color(cmap: Callable, i: int, n: int, denom: float = 0.0) -> tuple:
    return cmap((i + 1) / (n + denom))


def plot_current(ax: Axes, examples: dict, cmap: Callable = cmap_grays) -> None:
    shift_x = -1
    shift_y = 0.7
    traces = examples["traces"]
    n_ex = len(traces)
    for i, ex in enumerate(traces):
        ax.plot(
            ex[0] - i * shift_x,
            ex[1][:, 1] + i * shift_y,
            c=idx_to_color(cmap, i, n_ex),
            zorder=9 - i,
        )
    ax.margins(0.4, 0.05)
    ax.axis("off")


def plot_spikes(
    ax: Axes, examples: dict, config: dict, cmap: Callable = cmap_blues
) -> None:
    times = examples["spiketimes"]
    T = config["T"]
    n_ex = len(times)
    for i, t in enumerate(times):
        ax.eventplot([t], colors=idx_to_color(cmap, i, n_ex))
    ax.set_xlim(0, T)
    ax.set_axis_off()


def plot_voltage(
    ax: Axes,
    examples: dict,
    config: dict,
    cmap: Callable = cmap_blues,
    neuron_type: str = "QIF",
) -> None:
    T = config["T"]
    traces = examples["traces"]
    n_ex = len(traces)
    for i, (times, xs) in enumerate(traces):
        ax.plot(times, xs[:, 0], color=idx_to_color(cmap, i, n_ex), zorder=9 - i)
    ax.axhline(0, c="gray", alpha=0.3, zorder=0)
    ax.axhline(1, c="gray", alpha=0.3, zorder=0)
    ax.axvline(T, color="k", alpha=0.5, clip_on=False)
    ax.set_xticks([0, T], ["0", "T"])
    if neuron_type == "QIF":
        ax.text(0.01 * T, -1.6, "QIF", color="C0")
        ax.set_ylim(-2.0, 3.0)
        ax.set_yticks([0, 1], [r"$V_\mathrm{rest}$", r"$V_\mathrm{sep}$"])
    elif neuron_type == "LIF":
        ax.text(0.01 * T, -0.4, "LIF", color="C3")
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1], [r"$V_\mathrm{rest}$", r"$V_\Theta$"])


def plot_spiketimes(
    ax: Axes,
    grid: dict,
    examples: dict,
    config: dict,
    c: str,
    cmap: Callable = cmap_blues,
) -> None:
    p_ex = examples["p"]
    spike_times_ex = examples["spiketimes"]
    n_ex = len(p_ex)
    p_max = jnp.max(grid["p"])
    T = config["T"]

    # Plotting only every 10th point to speed up
    ax.plot(grid["p"][::10] / p_max, grid["spiketimes"][::10] / T, c=c, zorder=0)
    for i, (p, t) in enumerate(zip(p_ex, spike_times_ex)):
        ax.scatter(p / p_max, t / T, color=idx_to_color(cmap, i, n_ex), s=5, zorder=10)

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])


def plot_gradients(
    ax: Axes,
    grid: dict,
    examples: dict,
    config: dict,
    c: str,
) -> None:
    grads_ex = examples["grads"]
    grads_ex = jnp.where(examples["spiketimes"] > config["T"], jnp.inf, grads_ex)
    p_max = jnp.max(grid["p"])
    grads = grid["grads"]
    grads = jnp.where(grid["spiketimes"] > config["T"], jnp.inf, grads)

    # Plotting only every 10th point to speed up
    ax.plot(grid["p"][::10] / p_max, grads[::10], c=c, zorder=0)

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(-40, 1)
    ax.set_yticks([-40, -20, 0])

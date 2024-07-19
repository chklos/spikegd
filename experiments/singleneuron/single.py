from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import optax
from jax import jit, random, value_and_grad
from jaxtyping import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from spikegd.models import AbstractPseudoIFNeuron
from spikegd.utils.plotting import cmap_blues, cmap_oranges, cmap_purples, panel_label

# %%
############################
### Simulation
############################


def run(
    neuron: AbstractPseudoIFNeuron,
    config: dict,
) -> tuple[dict, dict]:
    """
    Trains a single neuron to spike at precise time points.

    The trained parameters `p` are the weights and times of `Ntrain` of the `Nin` input
    spikes.

    Args:
        neuron:
            Integrate-and-fire neuron model including pseudodynamics.
        config:
            Simulation configuration. Needs to contain the following items:
                `seed`: Random seed
                `Nin`: Number of input spikes
                `Ntrain`: Number of inputs with adjustable weights
                `T`: Trial duration
                `K`: Maximal number of simulated ordinary spikes
                `dt`: Integration time step (for state traces)
                `t_targets`: Target spike times
                `Ntrial`: Number of trials
                `lr`: Learning rate

    Returns:
        A tuple containing the learning dynamics
        and more detailed results from before and after learning.
        Both stored as a dictionary.
    """

    ### Unpack arguments
    seed: int = config["seed"]
    Nin: int = config["Nin"]
    Ntrain: int = config["Ntrain"]
    T: float = config["T"]
    t_targets: Array = config["t_targets"]
    Ntargets = len(t_targets)
    Ntrial: int = config["Ntrial"]
    lr: float = config["lr"]

    ### Set up the simulation

    # Simulation
    @jit
    def simulate(p: Array) -> tuple:
        ws, ts = p
        weights_in_p = weights_in.at[0, :Ntrain].set(ws)
        times_in = spikes_in[0].at[:Ntrain].set(ts)
        order = jnp.argsort(times_in)
        spikes_in_p = (times_in[order], spikes_in[1][order])
        out = neuron.event(x0, weights_net, weights_in_p, spikes_in_p, config)
        return out

    # Output spike times
    @jit
    def outfn(out: tuple) -> Array:
        ### Unpack output
        times: Array = out[0]
        spike_in: Array = out[1]
        neurons: Array = out[2]
        xs: Array = out[3]

        ### Spike numbers
        Kord = jnp.sum(neurons >= 0)  # Number of ordinary spikes
        Kout = jnp.sum(
            (neurons == 0) & (spike_in == False)  # noqa: E712
        )  # Number of ordinary output spikes

        ### Potential ordinary output spike times
        t_out_ord = times[jnp.argsort(spike_in)[:Ntargets]]

        ### Pseudospike times
        xend = xs[Kord]
        input = jnp.array(0.0)
        t_out_pseudo = neuron.t_pseudo(
            xend, input, jnp.arange(Ntargets) + 1 - Kout, config
        )

        ### Output spike times
        t_out = jnp.where(jnp.arange(Ntargets) < Kout, t_out_ord, t_out_pseudo)

        return t_out

    # Loss function
    @jit
    def lossfn(t_out: Array) -> Array:
        return jnp.mean((t_out - t_targets) ** 2)

    # Gradient
    @jit
    @partial(value_and_grad, has_aux=True)
    def gradfn(p: Array) -> tuple[Array, Array]:
        out = simulate(p)
        t_out = outfn(out)
        loss = lossfn(t_out)
        return loss, t_out

    # Update step
    @jit
    def trial(p: Array, opt_state: optax.OptState) -> tuple:
        (loss, t_out), grad = gradfn(p)
        updates, opt_state = optim.update(grad, opt_state)
        p = optax.apply_updates(p, updates)  # type: ignore
        p = p.at[1].set(jnp.clip(p[1], 0, T))
        metric = {"p": p, "t_out": t_out, "loss": loss, "grad": grad}
        return metric, opt_state

    ### Simulation

    # Create input and target
    key = random.PRNGKey(seed)
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    times_in = jnp.zeros(Nin)
    times_in = times_in.at[Ntrain:].set(
        random.uniform(subkey1, (Nin - Ntrain,), minval=0, maxval=T)
    )
    neurons_in = jnp.arange(Nin, dtype=int)
    spikes_in = (times_in, neurons_in)
    weights_in = jnp.zeros((1, Nin))
    weights_in = weights_in.at[0, Ntrain:].set(random.normal(subkey2, (Nin - Ntrain,)))

    # Initialize
    V0, I0 = 0.0, 0.0
    x0 = jnp.array([[V0], [I0]])
    weights_net = jnp.zeros((1, 1))
    ws = jnp.zeros(Ntrain)
    ts = jnp.array([1.0, 9.0])
    p = jnp.stack((ws, ts))
    optim = optax.chain(optax.clip(2e-2), optax.sgd(lr))
    opt_state = optim.init(p)
    metrics: dict[str, list | Array] = {"p": [p], "t_out": [], "loss": [], "grad": []}

    # Training
    for _ in range(Ntrial):
        metric, opt_state = trial(metrics["p"][-1], opt_state)
        metrics = {k: v + [metric[k]] for k, v in metrics.items()}
    metrics["p"] = metrics["p"][:-1]
    metrics = {k: jnp.array(v) for k, v in metrics.items()}
    if jnp.any(jnp.isnan(metrics["loss"])):  # type: ignore
        print(
            "Warning: A NaN appeared. "
            "Likely not enough spikes have been simulated. "
            "Try increasing `K`."
        )

    # Examples
    examples: dict[str, list | Array] = {
        "p": [],
        "times_in": [],
        "t_out": [],
        "traces": [],
    }
    for i in [0, Ntrial - 1]:
        p = metrics["p"][i]
        times_in = spikes_in[0].at[:Ntrain].set(p[0])
        out = simulate(p)
        t_out = outfn(out)
        traces = neuron.traces(x0, out, config)
        metric = {
            "p": p,
            "times_in": times_in,
            "t_out": t_out,
            "traces": traces,
        }
        examples = {k: v + [metric[k]] for k, v in examples.items()}
    examples["p"] = jnp.array(examples["p"])
    examples["times_in"] = jnp.array(examples["times_in"])
    examples["t_out"] = jnp.array(examples["t_out"])

    return metrics, examples


# %%
############################
### Plotting
############################
def idx_to_color(cmap: Callable, i: int, n: int, denom: float = 0.5) -> tuple:
    return cmap((i + 1) / (n + denom))


def plot_dynamics(
    fig: Figure,
    gs: gridspec.GridSpecFromSubplotSpec,
    examples: dict[str, list | Array],
    i_ex: int,
    config: dict,
    labels: bool = True,
) -> None:
    ### Unpack arguments
    T: float = 10 * config["T"]  # in ms
    Ntrial: int = config["Ntrial"]
    Ntargets = len(config["t_targets"])
    Ntrain: int = config["Ntrain"]
    t_targets: Array = 10 * config["t_targets"]  # in ms
    t_out = 10 * examples["t_out"][i_ex]  # in ms
    times_in = 10 * examples["times_in"][i_ex]  # in ms
    ps = examples["p"][i_ex]
    ps = ps.at[1].multiply(10)  # in ms
    ts = 10 * examples["traces"][i_ex][0]  # in ms
    Vs = examples["traces"][i_ex][1][:, 0]
    Is = examples["traces"][i_ex][1][:, 1]
    scalebar_offset = -2

    ### Output spikes
    ax = fig.add_subplot(gs[:2])
    for i in range(Ntargets):
        ax.eventplot(
            [t_out[i]],
            lineoffsets=0.0,
            colors=idx_to_color(cmap_blues, i, Ntargets),
            zorder=1,
        )
    ax.eventplot(t_targets, lineoffsets=0.0, colors="C4", zorder=0)
    ax.set_title("Trial {}".format([0, Ntrial][i_ex]))
    ax.set_xlim(0, T)
    ax.set_axis_off()
    if labels:
        panel_label(fig, ax, "(b)", x=-0.3, y=0.1)

    ### Voltage dynamics
    ax = fig.add_subplot(gs[2:7])
    ax.plot(ts, Vs, color="C0")
    ax.plot(2 * [scalebar_offset], [0, 1], c="k", clip_on=False)
    ax.axhline(0, c="gray", alpha=0.3)
    ax.axhline(1, c="gray", alpha=0.3)
    ax.set_xlim(0, T)
    ax.set_ylim(-7, 8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if labels:
        ax.set_ylabel(r"Potential $V$")

    ### Current dynamics
    ax = fig.add_subplot(gs[7:10])
    ax.plot(ts, Is, color="C0")
    ax.plot(2 * [scalebar_offset], [0, 1], c="k", clip_on=False)
    ax.axhline(0, c="gray", alpha=0.3)
    ax.set_xlim(0, T)
    ax.set_ylim(-3, 6)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if labels:
        ax.set_ylabel(r"Current $I$")

    ### Input spikes
    ax = fig.add_subplot(gs[10:])
    for i in range(Ntrain):
        ax.eventplot(
            [ps[1, i]],
            lineoffsets=0.0,
            colors=idx_to_color(cmap_oranges, i, Ntargets),
            zorder=10,
        )
    ax.eventplot(times_in[2:], lineoffsets=0.0, colors="k")
    ax.set_xticks([0, T])
    ax.set_xlabel(r"Time $t$ (ms)", labelpad=-5)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_spiketimes(ax: Axes, metrics: dict[str, Array], config: dict) -> None:
    ### Unpack arguments
    T: float = 10 * config["T"]  # in ms
    Ntrial: int = config["Ntrial"]
    Ntrain: int = config["Ntrain"]
    t_targets: Array = 10 * config["t_targets"]  # in ms
    Ntargets = len(t_targets)
    t_outs = 10 * metrics["t_out"]  # in ms
    t_ins = 10 * metrics["p"][:, 1]  # in ms

    ### Output spike times
    for i in range(Ntargets):
        ax.plot(t_outs[:, i], color=idx_to_color(cmap_blues, i, Ntrain))

    ### Input spike times
    for i in range(Ntrain):
        ax.plot(t_ins[:, i], color=idx_to_color(cmap_oranges, i, Ntrain))

    ### Target spike times
    for t in t_targets:
        ax.scatter(Ntrial, t, color="C4", s=10, zorder=10, clip_on=False)

    ### Pseudospike area
    patch = Rectangle((0, T), Ntrial, 100, color="k", alpha=0.2, zorder=10)
    ax.add_patch(patch)
    ax.text(0.4 * Ntrial, 115, "Pseudospike", color="white", zorder=11)

    ### Formatting
    ax.set_xticks([0, Ntrial])
    ax.set_xlabel("Trials", labelpad=-5)
    ax.set_yticks([0, T])
    max_t_out = float(1.1 * t_outs[0, 1])
    ax.set_ylim(0, max_t_out)
    ax.set_ylabel("Spike time (ms)", labelpad=0)


def plot_gradients(ax: Axes, metrics: dict[str, Array], config: dict) -> None:
    ### Unpack arguments
    Ntrain: int = config["Ntrain"]
    t_outs = 10 * metrics["t_out"]  # in ms
    grads = metrics["grad"]

    ### Gradient normalization
    def normalize(x: Array) -> Array:
        return x / jnp.max(jnp.abs(x))

    ### Spike time arc length
    arc_length = jnp.cumsum(
        jnp.sum(abs(jnp.diff(t_outs, axis=0, prepend=t_outs[0][jnp.newaxis])), axis=1)
    )

    ### dL/dt
    for i in range(Ntrain):
        ax.plot(
            arc_length,
            normalize(grads[:, 1, i]),
            color=idx_to_color(cmap_oranges, i, Ntrain),
            zorder=3,
        )

    ### dL/dw
    for i in range(Ntrain):
        ax.plot(
            arc_length,
            normalize(grads[:, 0, i]),
            color=idx_to_color(cmap_purples, i, Ntrain),
        )
    ax.legend(
        [
            r"$\partial L/\partial t_\mathrm{in,1}$",
            r"$\partial L/\partial t_\mathrm{in,2}$",
            r"$\partial L/\partial w_\mathrm{1}$",
            r"$\partial L/\partial w_\mathrm{2}$",
        ],
        labelcolor="linecolor",
        handlelength=0.0,
        # frameon=False,
        labelspacing=0.0,
    )

    ### Formatting
    ax.set_xticks([0, 120])
    ax.set_xlabel("Spike time\narc length (ms)", labelpad=-9)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Normalized\nGradient", labelpad=-2)

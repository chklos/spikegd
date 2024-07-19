from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import optax
from jax import jit, random, value_and_grad, vmap
from jaxtyping import Array
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter
from tqdm.notebook import trange

from spikegd.models import AbstractPseudoIFNeuron
from spikegd.utils.plotting import formatter, loc_min

# %%
############################
### Initialization
############################


def init_input(key: Array, config: dict) -> tuple:
    """
    Creates input spikes.
    """
    ### Unpack arguments
    T: float = config["T"]
    N: int = config["N"]
    Nin = 2 * N
    r_in: float = config["r_in"]
    Tinit = 10.0  # To get same inputs as in paper

    ### Input spikes
    key, subkey = random.split(key)
    ISIs_exc = r_in * random.exponential(subkey, (N, int(5 * (T + Tinit) * r_in)))
    times_exc = jnp.cumsum(ISIs_exc, axis=1)
    key, subkey = random.split(key)
    ISIs_inh = 1 / r_in * random.exponential(subkey, (N, int(5 * (T + Tinit) * r_in)))
    times_inh = jnp.cumsum(ISIs_inh, axis=1)
    times = jnp.concatenate((times_exc, times_inh), axis=0)
    times_in = jnp.inf * jnp.ones_like(times)
    for i in range(Nin):
        times_after_init = times[i, times[i] > Tinit] - Tinit
        times_in = times_in.at[i, : len(times_after_init)].set(times_after_init)
    times_in = jnp.ravel(times_in)
    neurons_in = jnp.repeat(jnp.arange(Nin), times.shape[1])
    order = jnp.argsort(times_in)
    spikes_in = (times_in[order], neurons_in[order])

    return key, spikes_in


def init_weights(config: dict) -> Array:
    """
    Creates input weights.
    """
    ### Unpack arguments
    N: int = config["N"]
    Nin = 2 * N
    w_in_exc: float = config["w_in_exc"]
    w_in_inh: float = config["w_in_inh"]

    ### Weights
    weights_in = jnp.zeros((N, Nin))
    weights_in = weights_in.at[jnp.arange(N), jnp.arange(N)].set(w_in_exc)
    weights_in = weights_in.at[jnp.arange(N), jnp.arange(N, 2 * N)].set(w_in_inh)

    return weights_in


def init_target(key: Array, config: dict) -> tuple:
    """
    Creates target spike times.
    """
    ### Unpack arguments
    T: float = config["T"]
    Ntar: int = config["Ntar"]
    r_targets: list[float] = config["r_targets"]

    ### Create target spikes
    key, subkey = random.split(key)
    ISIs = random.exponential(subkey, (Ntar, int(2.5 * T)))
    for i, rtarget in enumerate(r_targets):
        ISIs = ISIs.at[i, :].divide(rtarget)
    ISIs = ISIs.at[:, 1:].add(1)
    times = jnp.cumsum(ISIs, axis=1)
    t_targets: list[Array] = []
    for i in range(Ntar):
        t_targets.append(jnp.append(times[i, times[i] < T], 1.1 * T))

    return key, t_targets


# %%
############################
### Model
############################


def w_to_weights_net(w: Array) -> Array:
    """
    Transforms learnable weights to network weight matrix.
    """
    weights_net = jnp.array([jnp.insert(w_i, i, 0) for i, w_i in enumerate(w)])
    return weights_net


def weights_net_to_w(weights_net: Array) -> Array:
    """
    Extract learnable weights from network weight matrix.
    """
    w = jnp.array([jnp.delete(w_i, i) for i, w_i in enumerate(weights_net)])
    return w


def outfn(
    neuron: AbstractPseudoIFNeuron,
    out: tuple,
    p: list[Array],
    t_targets: list[Array],
    config: dict,
) -> list[Array]:
    """
    Computes output spike times given simulation results.
    """
    ### Unpack arguments
    Ntar: int = config["Ntar"]

    ### Unpack output
    times: Array = out[0]
    spikes_in: Array = out[1]
    neurons: Array = out[2]
    xs: Array = out[3]

    ### Set weights
    weights_net_p = w_to_weights_net(p[0])

    ### State at trial end
    Kord = jnp.sum(neurons >= 0)  # Number of ordinary spikes
    x_end = xs[Kord]
    pseudo_rates = neuron.construct_ratefn(x_end)(input=-x_end[1])

    ### Spike times for each learned neuron
    t_outs = []
    for i in range(Ntar):
        Ntargets_i = len(t_targets[i])

        ### Potential ordinary output spike times
        mask = (neurons == i) & (spikes_in == False)  # noqa: E712
        Kout = jnp.sum(mask)  # Number of ordinary output spikes
        t_out_ord = times[jnp.argsort(~mask)[:Ntargets_i]]

        ### Pseudospike times
        input = neuron.linear(pseudo_rates, weights_net_p[i])
        t_out_pseudo = neuron.t_pseudo(
            x_end[:, i], input, jnp.arange(Ntargets_i) + 1 - Kout, config
        )

        ### Output spike times
        t_out = jnp.where(jnp.arange(Ntargets_i) < Kout, t_out_ord, t_out_pseudo)
        t_outs.append(t_out)

    return t_outs


def lossfn(t_outs: list[Array], t_targets: list[Array]) -> Array:
    """
    Computes loss given output and target spike times.
    """
    Ntargets = [len(t_target) for t_target in t_targets]

    loss = [
        jnp.sum(((t_out[:-1] - t_target[:-1]) / (t_target[:-1] + 2)) ** 2)
        for t_out, t_target in zip(t_outs, t_targets)
    ]
    loss_extra = [
        ((t_out[-1] - t_target[-1]) / (t_target[-1] + 2)) ** 2
        * (t_out[-1] < t_target[-1])
        for t_out, t_target in zip(t_outs, t_targets)
    ]
    loss = jnp.array([(loss + loss_e) for loss, loss_e in zip(loss, loss_extra)])
    loss = jnp.sum(loss) / jnp.sum(jnp.array(Ntargets))

    return loss


def simulatefn(
    neuron: AbstractPseudoIFNeuron,
    p: list[Array],
    weights_in: Array,
    spikes_in: tuple,
    t_targets: list[Array],
    config: dict,
) -> tuple[Array, list[Array], tuple]:
    """
    Simulates a single trial.
    """
    w, x0 = p
    weights_net = w_to_weights_net(w)
    out = neuron.event(x0, weights_net, weights_in, spikes_in, config)
    t_outs = outfn(neuron, out, p, t_targets, config)
    loss = lossfn(t_outs, t_targets)
    return loss, t_outs, out


# %%
############################
### Training
############################


def run(
    neuron: AbstractPseudoIFNeuron,
    optim: optax.GradientTransformation,
    config: dict,
    mode: str = "normal",
) -> tuple[dict, dict]:
    """
    Trains some neurons of a RNN to reproduce target spike times.

    The RNN is driven by excitatory and inhibitory Poissonian input spike trains.
    The target spike times are generated by a Poissonian process. The trained parameters
    `p` are the recurrent weights and the initial states of the network neurons.

    Args:
        neuron:
            Integrate-and-fire neuron model including pseudodynamics.
        optim:
            Optimizer.
        config:
            Simulation configuration. Needs to contain the following items:
                `seed`: Random seed
                `N`: Number of neurons
                `w_in_exc`: Weight of excitatory inputs
                `w_in_inh`: Weight of inhibitory inputs
                `r_in`: Rate of inputs
                `T`: Trial duration
                `K`: Maximal number of simulated ordinary spikes
                `dt`: Integration time step (for state traces)
                `Ntar`: Number of network neurons whose spike times are learned
                `r_targets`: Rates with which target spikes are generated
                `Ntrial`: Number of training trials
                `dt_max`: Maximal allowed spike time change when using grid search
                `grid`: Grid used to determine best step size when using grid search
        mode:
            Optimization mode. Either 'normal' or 'grid'.
    Returns:
        A tuple containing the learning dynamics and more detailed results from before
        and after learning as well as from after learning with weights of connections
        projecting to non-target neurons set to 0. Both stored as a dictionary.
    """

    ### Unpack arguments
    seed: int = config["seed"]
    N: int = config["N"]
    Ntar: int = config["Ntar"]
    Ntrial: int = config["Ntrial"]

    ### Set up the simulation

    # Simulation
    @jit
    def simulate(p: list[Array]) -> tuple[Array, list[Array], tuple]:
        loss, t_outs, out = simulatefn(
            neuron, p, weights_in, spikes_in, t_targets, config
        )
        return loss, t_outs, out

    # Gradient
    @jit
    @partial(value_and_grad, has_aux=True)
    def gradfn(p: list[Array]) -> tuple[Array, list[Array]]:
        loss, t_outs, _ = simulate(p)
        return loss, t_outs

    # Update step
    if mode == "normal":

        @jit
        def trial(p: list[Array], opt_state: optax.OptState) -> tuple:
            (loss, t_out), grad = gradfn(p)
            updates, opt_state = optim.update(grad, opt_state)
            p = optax.apply_updates(p, updates)  # type: ignore
            metric = {"p": p, "t_out": t_out, "loss": loss}
            return metric, opt_state
    elif mode == "grid":

        @jit
        def trial(p: list[Array], opt_state: optax.OptState) -> tuple:
            ### Unpack arguments
            dt_max: float = config["dt_max"]
            grid: Array = config["grid"]

            ### Run simulation
            (loss, t_out), grad = gradfn(p)
            updates, opt_state = optim.update(grad, opt_state)

            ### Grid search for best step size
            p_grid = [
                p_i[..., jnp.newaxis] + updatei[..., jnp.newaxis] * grid
                for p_i, updatei in zip(p, updates)  # type: ignore
            ]
            loss_grid, t_out_grid, _ = vmap(simulate, in_axes=([-1, -1],))(p_grid)
            mask = [
                jnp.all(abs(t - t_g) < dt_max, 1) for t, t_g in zip(t_out, t_out_grid)
            ]
            mask = jnp.all(jnp.array(mask), 0)
            grid_i = jnp.argmin(jnp.where(mask, loss_grid, jnp.inf))  # type: ignore

            ### Update parameters
            p = [p_i[..., grid_i] for p_i in p_grid]

            ### Store metric
            metric = {"p": p, "t_out": t_out, "loss": loss}

            return metric, opt_state
    else:
        raise ValueError(f"`mode` must be either 'normal' or 'grid', not {mode}.")

    ### Simulation

    # Set up simulation
    key = random.PRNGKey(seed)
    key, spikes_in = init_input(key, config)
    weights_in = init_weights(config)
    key, t_targets = init_target(key, config)

    # Initialize
    V0, I0 = jnp.zeros(N), jnp.zeros(N)
    x0 = jnp.array([V0, I0])
    w0 = jnp.zeros((N, N - 1))
    p = [w0, x0]
    opt_state = optim.init(p)
    metrics: dict[str, list | Array] = {"p": [p], "t_out": [], "loss": []}

    # Training
    for _ in trange(Ntrial + 1):
        metric, opt_state = trial(metrics["p"][-1], opt_state)
        metrics = {k: v + [metric[k]] for k, v in metrics.items()}
    metrics["p"] = metrics["p"][:-1]
    metrics["loss"] = jnp.array(metrics["loss"])
    metrics["t_targets"] = t_targets
    if jnp.any(jnp.isnan(metrics["loss"])):
        print(
            "Warning: A NaN appeared. "
            "Likely not enough spikes have been simulated. "
            "Try increasing `K`."
        )

    # Check max spike time change
    if mode == "grid":
        dt_max: float = config["dt_max"]
        t_out = metrics["t_out"]
        t_out = [jnp.array([ts[i] for ts in t_out]) for i in range(Ntar)]
        t_out_diff = [jnp.max(abs(jnp.diff(ts, 1, axis=0))) for ts in t_out]
        t_out_diff = jnp.max(jnp.array(t_out_diff))
        if t_out_diff > dt_max:
            print(f"Warning: maximum spike time change is {t_out_diff} > {dt_max}.")

    # Examples
    examples: dict[str, list] = {
        "outs": [],
        "t_outs": [],
        "t_targets": t_targets,
    }
    for i in [0, Ntrial]:
        p = metrics["p"][i]
        _, t_out, out = simulate(p)
        examples["outs"].append(out)
        examples["t_outs"].append(t_out)
    # Recurrent weights excluding trained neurons set to 0
    w, x0 = metrics["p"][-1]
    weights_net = w_to_weights_net(w)
    weights_net = weights_net.at[Ntar:, :].set(0)
    w_norec = weights_net_to_w(weights_net)
    _, t_out, out = simulate([w_norec, x0])
    examples["outs"].append(out)
    examples["t_outs"].append(t_out)

    return metrics, examples


# %%
############################
### Plotting
############################


def idx_to_color(cmap: Callable, i: int, n: int, denom: float = 0.5) -> tuple:
    return cmap((i + 1) / (n + denom))


def plot_spikes(ax: Axes, examples: dict, i_ex: int, config: dict) -> None:
    ### Unpack arguments
    T: float = 10 * config["T"]  # in ms
    N: int = config["N"]
    Ntar: int = config["Ntar"]
    Ntrial: int = config["Ntrial"]
    out = examples["outs"][i_ex]
    t_targets = examples["t_targets"]
    t_targets = [10 * ts for ts in t_targets]  # in ms

    ### Get spikes
    times: Array = 10 * out[0]  # in ms
    spike_in: Array = out[1]
    neurons: Array = out[2]
    spiketimes = []
    for i in range(N):
        times_i = times[~spike_in & (neurons == i)]
        times_i = times_i[times_i < T]
        spiketimes.append(times_i)

    ### Plot spikes
    colors = ["C0", "C1"] + (N - Ntar) * ["k"]
    offsets = jnp.arange(Ntar) + 0.5
    offsets = list(jnp.concatenate((offsets, 0.5 * jnp.arange(N - Ntar) + 2.5)))
    lengths = Ntar * [1] + [0.5] * (N - Ntar)
    # Network spikes
    ax.eventplot(
        spiketimes, colors=colors, linelengths=lengths, lineoffsets=offsets, zorder=1
    )
    # Target spikes
    ax.eventplot(
        t_targets,
        colors="C4",
        linelengths=lengths[:Ntar],
        lineoffsets=offsets[:Ntar],
        zorder=0,
    )

    ### Formatting
    ax.set_title("Trial {}".format([0, Ntrial][i_ex]), pad=-1)
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$ (ms)", labelpad=-6)
    ax.set_xticks([0, T])
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])


def plot_times(
    ax: Axes,
    metrics: dict,
    i_ex: int,
    config: dict,
    x: str = "trials",
    smoothing: int = 1,
    color="C0",
) -> None:
    ### Unpack arguments
    T: float = 10 * config["T"]  # in ms
    Ntrial: int = config["Ntrial"]
    t_targets = 10 * metrics["t_targets"][i_ex]  # in ms
    t_outs = 10 * jnp.array([ts[i_ex] for ts in metrics["t_out"]])  # in ms

    ### Plot
    if x == "trials":
        xs = jnp.arange(1, Ntrial + 2)
        # Formatting
        kwargs = {}
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(loc_min)
        ax.set_xlabel("Trials+1", labelpad=2)
        ax.set_ylim(0, 1.3 * T)
    elif x == "arclength":
        smoothed_t_outs = jnp.apply_along_axis(
            lambda x: jnp.convolve(x, jnp.ones(smoothing) / (smoothing), mode="valid"),
            0,
            t_outs,
        )
        arc_length = jnp.cumsum(
            jnp.sum(
                abs(
                    jnp.diff(smoothed_t_outs, axis=0, prepend=smoothed_t_outs[0][None])
                ),
                axis=1,
            )
        )
        xs = arc_length
        if smoothing != 1:
            t_outs = t_outs[(smoothing - 1) // 2 : -(smoothing - 1) // 2, :]
        # Formatting
        kwargs = {"ls": "", "marker": ".", "ms": 1.0, "rasterized": True}
        ax.set_xlabel("Spike time arc length (ms)", labelpad=2)
        ax.set_ylim(0, 1.3 * T)
    else:
        raise ValueError(f"`x` must be either 'trials' or 'arclength', not {x}.")

    ax.plot(xs, t_outs[:, :-1], c=color, **kwargs)
    ax.plot(xs, t_outs[:, -1], c="k", **kwargs)
    patch = Rectangle((0, T), float(xs[-1]), 100, color="k", alpha=0.2, zorder=10)
    ax.add_patch(patch)
    ax.plot(
        float(xs[-1]) * jnp.ones_like(t_targets[:-1]),
        t_targets[:-1],
        c="C4",
        ls="",
        marker=".",
        clip_on=False,
    )


def plot_loss(ax: Axes, metrics: dict, config: dict) -> None:
    ### Unpack arguments
    Ntrial: int = config["Ntrial"]
    loss = metrics["loss"]

    ### Plot
    ax.plot(jnp.arange(Ntrial + 1), loss, c="k")

    ### Formatting
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(loc_min)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Trials+1", labelpad=2)
    ax.set_yscale("log")
    ax.set_ylabel("Loss", labelpad=2)


def plot_spikescomparison(ax: Axes, examples: dict, config: dict) -> None:
    ### Unpack arguments
    T: float = 10 * config["T"]  # in ms
    Ntar: int = config["Ntar"]
    Ntrial: int = config["Ntrial"]
    t_outs = examples["t_outs"]
    t_targets = examples["t_targets"]
    t_targets = [10 * ts for ts in t_targets]  # in ms

    ### Plot spikes
    colors = ["C0", "C1"]
    d = 0.1
    offsets = 3 * (1 + d) * jnp.arange(Ntar) + 0.5
    lengths = jnp.ones(Ntar)
    # Spikes including recurrent weights
    spiketimes = [10 * ts for ts in t_outs[1]]  # in ms
    ax.eventplot(
        spiketimes,
        colors=colors,
        linelengths=lengths,  # type: ignore
        lineoffsets=offsets + 1 + d,  # type: ignore
    )
    # Spikes without recurrent weights
    spiketimes = [10 * ts for ts in t_outs[-1]]  # in ms
    ax.eventplot(
        spiketimes,
        colors=colors,
        alpha=0.5,
        linelengths=lengths,  # type: ignore
        lineoffsets=offsets,  # type: ignore
    )
    # Target spikes
    ax.eventplot(
        t_targets,
        colors="C4",
        linelengths=lengths[:Ntar],  # type: ignore
        lineoffsets=offsets[:Ntar] + 2 * (1 + d),  # type: ignore
    )
    # Separation lines
    for i in range(1, Ntar):
        ax.axhline(3 * (1 + d) * i - d / 2, c="k", lw=1, zorder=10)

    ### Formatting
    ax.set_title(f"Trial {Ntrial}", pad=-1)
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$ (ms)", labelpad=-6)
    ax.set_xticks([0, T])
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

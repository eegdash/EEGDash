from __future__ import annotations

import abc
import math
import re
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle, RegularPolygon

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from eegdash.plotting.dataset_summary_adapter import (
        LegendConfig,
        LegendLabelItem,
        LegendSizeBin,
    )


def _get_hexa_grid(
    n: int,
    diameter: float,
    center: tuple[float, float],
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a hexagonal grid for ``n`` bubbles centred around *center*.

    The historical implementation relied on stochastic jitter which made the
    layout non-deterministic. The helper now disables jitter by default and only
    reintroduces it when ``seed`` is provided, enabling reproducible results.
    """
    base = np.arange(n) - n // 2
    if seed is None:
        offset_x = 0.0
        offset_y = 0.0
    else:
        rng = np.random.default_rng(seed)
        offset_x = rng.uniform(-0.25, 0.25)
        offset_y = rng.uniform(-0.25, 0.25)

    x, y = np.meshgrid(base + offset_x, base + offset_y)
    x = x.flatten()
    y = y.flatten()
    return (
        np.concatenate([x, x + 0.5]) * diameter + center[0],
        np.concatenate([y, y + 0.5]) * diameter * np.sqrt(3) + center[1],
    )


def _get_bubble_coordinates(
    n: int,
    diameter: float,
    center: tuple[float, float],
    *,
    layout_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = _get_hexa_grid(n, diameter, center, seed=layout_seed)
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    sort_idx = dist.argsort()
    x = x[sort_idx]
    y = y[sort_idx]
    return x[:n], y[:n]


def _plot_shape(shape: str, *args, **kwargs):
    if shape == "circle":
        return Circle(*args, **kwargs)
    if shape == "hexagon":
        return RegularPolygon(*args, numVertices=6, **kwargs)
    raise ValueError(f"Unknown shape {shape}")


def _plot_hexa_bubbles(
    *,
    n: int,
    diameter: float,
    center: tuple[float, float] = (0.0, 0.0),
    ax,
    shape: Literal["circle", "hexagon"] = "circle",
    gap: float = 0.0,
    gid: str | None = None,
    layout_seed: int | None = None,
    **kwargs,
):
    x, y = _get_bubble_coordinates(n, diameter + gap, center, layout_seed=layout_seed)
    bubbles = [
        _plot_shape(shape, (xi, yi), radius=diameter / 2, **kwargs)
        for xi, yi in zip(x, y)
    ]
    collection = PatchCollection(bubbles, match_original=True)
    if gid is not None:
        collection.set_gid(gid)
    ax.add_collection(collection)
    return x, y


def _add_bubble_legend(
    scale: float,
    size_mode: Literal["count", "duration"],
    color_map: Mapping[str, Any],
    alphas: Sequence[float],
    fontsize: int,
    shape: Literal["circle", "hexagon"],
    x0: float,
    y0: float,
    ax,
):
    """Legacy legend renderer retained for backwards compatibility."""
    circles = []  # (text, diameter, alpha, color)
    alpha = alphas[0]
    if size_mode == "count":
        sizes = [("100 trials", 100), ("1000 trials", 1000), ("10000 trials", 10000)]
    elif size_mode == "duration":
        sizes = [
            ("6 minutes", 60 * 6),
            ("1 hour", 60 * 60),
            ("10 hours", 60 * 60 * 10),
        ]
    else:
        raise ValueError(f"Unknown size_mode {size_mode}")
    for desc, size in sizes:
        circles.append((desc, np.log(size) * scale, alpha, "black"))
    circles.append(None)
    for paradigm, color in color_map.items():
        circles.append((paradigm, np.log(1000) * scale, alpha, color))
    circles.append(None)
    circles.append(("1 session", np.log(1000) * scale, alphas[0], "black"))
    circles.append(("3 sessions", np.log(1000) * scale, alphas[2], "black"))
    circles.append(("5 sessions", np.log(1000) * scale, alphas[4], "black"))

    for i, item in enumerate(reversed(circles)):
        if item is None:
            continue
        text, diameter, alpha_value, color = item
        y = i * fontsize / 2 + y0
        bubble = _plot_shape(
            shape,
            (x0, y),
            radius=diameter / 2,
            alpha=alpha_value,
            color=color,
            lw=0,
            gid=f"legend/bubble/{text}",
        )
        ax.add_patch(bubble)
        ax.text(
            x0 + 5,
            y,
            text,
            ha="left",
            va="center",
            fontsize=fontsize,
            gid=f"legend/text/{text}",
        )


def _render_structured_legend(
    legend_config: LegendConfig,
    ax,
    *,
    scale: float,
    shape: Literal["circle", "hexagon"],
    fontsize: int,
    x0: float,
    y0: float,
):
    """Render a structured legend driven by :class:`LegendConfig`."""
    line_height = fontsize * 1.25
    y_cursor = y0

    if legend_config.heading:
        ax.text(
            x0,
            y_cursor,
            legend_config.heading,
            ha="left",
            va="center",
            fontsize=fontsize + 1,
            fontweight="bold",
            gid="legend/heading",
        )
        y_cursor += line_height

    if legend_config.size_bins:
        for bin_entry in legend_config.size_bins:
            diameter = max(_compute_diameter(bin_entry.value, scale), fontsize * 0.35)
            patch = _plot_shape(
                shape,
                (x0, y_cursor),
                radius=diameter / 2,
                color="#111827",
                alpha=0.35,
                lw=0,
                gid=bin_entry.gid,
            )
            ax.add_patch(patch)
            ax.text(
                x0 + diameter / 2 + fontsize * 0.6,
                y_cursor,
                bin_entry.label,
                ha="left",
                va="center",
                fontsize=fontsize,
                gid=f"{bin_entry.gid}/label",
            )
            y_cursor += line_height
        y_cursor += line_height * 0.5

    if legend_config.modalities:
        modality_swatch_size = fontsize * 0.8
        modality_margin = modality_swatch_size * 0.1
        for modality in legend_config.modalities:
            patch = Rectangle(
                (x0 - modality_margin, y_cursor - modality_swatch_size * 0.5),
                width=modality_swatch_size,
                height=modality_swatch_size,
                linewidth=0,
                facecolor=getattr(modality, "color", "#111827"),
                gid=modality.gid,
            )
            ax.add_patch(patch)
            ax.text(
                x0 + modality_swatch_size * 0.6,
                y_cursor,
                modality.label,
                ha="left",
                va="center",
                fontsize=fontsize,
                gid=f"{modality.gid}/label",
            )
            y_cursor += line_height
        y_cursor += line_height * 0.5

    if legend_config.type_subjects:
        badge_height = fontsize * 0.9
        badge_width = fontsize * 1.9
        badge_margin = fontsize * 0.1
        for badge in legend_config.type_subjects:
            rect = Rectangle(
                (x0 - badge_margin, y_cursor - badge_height * 0.5),
                width=badge_width,
                height=badge_height,
                linewidth=0,
                facecolor=getattr(badge, "color", "#334155"),
                gid=badge.gid,
            )
            ax.add_patch(rect)
            ax.text(
                x0 + badge_width * 0.05,
                y_cursor,
                badge.label,
                ha="left",
                va="center",
                fontsize=fontsize * 0.9,
                color="#ffffff",
                fontweight="bold",
                gid=f"{badge.gid}/label",
            )
            y_cursor += line_height


def _match_int(value: str) -> int:
    """Match the first integer in a string."""
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise AssertionError(f"Cannot parse number from '{value}'")
    return int(match.group(1))


def _get_dataset_parameters(dataset):
    row = dataset._summary_table
    dataset_name = dataset.__class__.__name__
    paradigm = dataset.paradigm
    n_subjects = len(dataset.subject_list)
    n_sessions = _match_int(row["#Sessions"])
    if paradigm in ["imagery", "ssvep"]:
        n_trials = _match_int(row["#Trials / class"]) * _match_int(row["#Classes"])
    elif paradigm == "rstate":
        n_trials = _match_int(row["#Classes"]) * _match_int(row["#Blocks / class"])
    elif paradigm == "cvep":
        n_trials = _match_int(row["#Trials / class"]) * _match_int(
            row["#Trial classes"]
        )
    else:  # p300
        match = re.search(r"(\d+) NT / (\d+) T", row["#Trials / class"])
        if match is not None:
            n_trials = int(match.group(1)) + int(match.group(2))
        else:
            n_trials = _match_int(row["#Trials / class"])
    trial_len = float(row["Trials length (s)"])
    return (
        dataset_name,
        paradigm,
        n_subjects,
        n_sessions,
        n_trials,
        trial_len,
    )


def get_bubble_size(
    size_mode: Literal["duration", "count"],
    n_sessions: int,
    n_trials: int,
    trial_len: float,
) -> float:
    if size_mode == "duration":
        return n_trials * n_sessions * trial_len
    if size_mode == "count":
        return n_trials * n_sessions
    raise ValueError(f"Unknown size_mode {size_mode}")


def _compute_diameter(size_value: float, scale: float) -> float:
    safe_value = max(float(size_value), math.e**1e-3)
    log_value = math.log(safe_value)
    diameter = log_value * scale
    if diameter <= 0:
        diameter = max(scale * 0.05, 1e-3)
    return diameter


def get_dataset_area(
    n_subjects: int,
    n_sessions: int,
    n_trials: int,
    trial_len: float,
    scale: float = 0.5,
    size_mode: Literal["count", "duration"] = "count",
    gap: float = 0.0,
) -> float:
    size = get_bubble_size(
        size_mode=size_mode,
        n_sessions=n_sessions,
        n_trials=n_trials,
        trial_len=trial_len,
    )
    diameter = _compute_diameter(size, scale) + gap
    return n_subjects * 3 * 3**0.5 / 8 * diameter**2  # area of hexagons


def dataset_bubble_plot(
    dataset=None,
    center: tuple[float, float] = (0.0, 0.0),
    scale: float = 0.5,
    size_mode: Literal["count", "duration"] = "count",
    shape: Literal["circle", "hexagon"] = "circle",
    gap: float = 0.0,
    color_map: Mapping[str, Any] | None = None,
    alphas: Sequence[float] | None = None,
    title: bool = True,
    legend: bool = True,
    legend_position: tuple[float, float] | None = None,
    fontsize: int = 8,
    ax=None,
    scale_ax: bool = True,
    dataset_name: str | None = None,
    paradigm: str | None = None,
    n_subjects: int | None = None,
    n_sessions: int | None = None,
    n_trials: int | None = None,
    trial_len: float | None = None,
    size_override: float | None = None,
    alpha_override: float | None = None,
    legend_config: LegendConfig | None = None,
    layout_seed: int | None = None,
):
    """Plot a bubble plot for a dataset.

    Each bubble represents one subject. The size of the bubble is
    proportional to the selected scaling metric on a log scale, the color
    represents the paradigm, and the alpha is proportional to
    the number of sessions.

    Parameters
    ----------
    legend_config:
        Optional structured legend specification. When provided and ``legend``
        is truthy, the structured legend supersedes the legacy MOABB legend.
    layout_seed:
        Optional seed forwarded to the layout helper to ensure reproducible
        per-dataset jitter. When ``None`` the layout becomes deterministic.
    """
    palette = sea.color_palette("tab10", 5)
    color_map = color_map or dict(
        zip(["imagery", "p300", "ssvep", "cvep", "rstate"], palette)
    )

    alphas = alphas or [0.8, 0.65, 0.5, 0.35, 0.2]

    if dataset is not None:
        (
            derived_dataset_name,
            derived_paradigm,
            derived_n_subjects,
            derived_n_sessions,
            derived_n_trials,
            derived_trial_len,
        ) = _get_dataset_parameters(dataset)
        dataset_name = dataset_name or derived_dataset_name
        paradigm = paradigm or derived_paradigm
        n_subjects = n_subjects or derived_n_subjects
        n_sessions = n_sessions or derived_n_sessions
        n_trials = n_trials or derived_n_trials
        trial_len = trial_len or derived_trial_len
    else:
        if any(
            value is None
            for value in [dataset_name, n_subjects, n_sessions, n_trials, trial_len]
        ):
            raise ValueError(
                "If dataset is None, then dataset_name, n_subjects, n_sessions, "
                "n_trials and trial_len must be provided"
            )

    if dataset_name is None:
        dataset_name = "Unknown dataset"
    if paradigm is None:
        paradigm = "Unknown"
    if n_subjects is None:
        raise ValueError("n_subjects must be provided")
    if n_sessions is None:
        raise ValueError("n_sessions must be provided")
    if n_trials is None:
        raise ValueError("n_trials must be provided")
    if trial_len is None:
        raise ValueError("trial_len must be provided")

    n_subjects = max(int(n_subjects), 1)
    n_sessions = max(int(n_sessions), 1)
    n_trials = max(int(n_trials), 1)
    trial_len = float(trial_len) if trial_len and trial_len > 0 else 1.0

    if size_override is not None:
        size = max(float(size_override), math.e**1e-3)
    else:
        size = get_bubble_size(
            size_mode=size_mode,
            n_sessions=n_sessions,
            n_trials=n_trials,
            trial_len=trial_len,
        )
    diameter = _compute_diameter(size, scale)

    ax = ax or plt.gca()
    alpha_value = (
        alpha_override
        if alpha_override is not None
        else alphas[min(n_sessions, len(alphas)) - 1]
    )
    color = color_map.get(paradigm, next(iter(color_map.values()), "black"))

    x, y = _plot_hexa_bubbles(
        n=n_subjects,
        color=color,
        ax=ax,
        diameter=diameter,
        alpha=alpha_value,
        lw=0,
        center=center,
        shape=shape,
        gap=gap,
        gid=f"bubbles/{dataset_name}",
        layout_seed=layout_seed,
    )
    if title:
        ax.text(
            center[0],
            center[1],
            dataset_name,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="black",
            bbox=dict(
                facecolor="white",
                alpha=0.6,
                linewidth=0,
                boxstyle="round,pad=0.5",
            ),
            gid=f"title/{dataset_name}",
        )
    if legend:
        legend_position = legend_position or (x.max() + fontsize, y.min())
        if legend_config is not None:
            _render_structured_legend(
                legend_config,
                ax,
                scale=scale,
                shape=shape,
                fontsize=fontsize,
                x0=legend_position[0],
                y0=legend_position[1],
            )
        else:
            _add_bubble_legend(
                scale=scale,
                size_mode=size_mode,
                color_map=color_map,
                alphas=alphas,
                fontsize=fontsize,
                x0=legend_position[0],
                y0=legend_position[1],
                ax=ax,
                shape=shape,
            )
    ax.axis("off")
    if scale_ax:
        ax.axis("equal")
        ax.autoscale()
    return ax


class _BubbleChart:
    def __init__(self, area, bubble_spacing=0.0):
        """
        Setup for bubble collapse.

        From https://matplotlib.org/stable/gallery/misc/packed_bubbles.html

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[: len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[: len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3])

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0], bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations: int = 50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform. Increasing the value improves the
            determinism of the resulting layout at the cost of runtime.
        """
        for _ in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bubbles = np.delete(self.bubbles, i, 0)
                direction = self.com - self.bubbles[i, :2]
                direction = direction / np.sqrt(direction.dot(direction))

                new_point = self.bubbles[i, :2] + direction * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                if not self.check_collisions(new_bubble, rest_bubbles):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    for colliding in self.collides_with(new_bubble, rest_bubbles):
                        direction = rest_bubbles[colliding, :2] - self.bubbles[i, :2]
                        direction = direction / np.sqrt(direction.dot(direction))
                        orth = np.array([direction[1], -direction[0]])
                        new_point1 = self.bubbles[i, :2] + orth * self.step_dist
                        new_point2 = self.bubbles[i, :2] - orth * self.step_dist
                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bubbles):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def get_centers(self):
        return self.bubbles[:, :2]


class _BaseDatasetPlotter:
    def __init__(self, datasets, meta_gap, kwargs, n_col=None, collapse_iterations=120):
        self.datasets = (
            datasets
            if datasets is not None
            else sorted(
                [
                    dataset()
                    for dataset in dataset_list
                    if "Fake" not in dataset.__name__
                ],
                key=lambda value: value.__class__.__name__,
            )
        )
        areas_list = []
        for dataset in self.datasets:
            if isinstance(dataset, dict):
                n_subjects = dataset["n_subjects"]
                n_sessions = dataset["n_sessions"]
                n_trials = dataset["n_trials"]
                trial_len = dataset["trial_len"]
            else:
                _, _, n_subjects, n_sessions, n_trials, trial_len = (
                    _get_dataset_parameters(dataset)
                )
            areas_list.append(
                get_dataset_area(
                    n_subjects=n_subjects,
                    n_sessions=n_sessions,
                    n_trials=n_trials,
                    trial_len=trial_len,
                )
            )
        self.areas = np.array(areas_list)
        self.radii = np.sqrt(self.areas / np.pi)
        self.meta_gap = meta_gap
        self.kwargs = kwargs
        self.n_col = n_col
        self.collapse_iterations = collapse_iterations

    @abc.abstractmethod
    def _get_centers(self) -> np.ndarray:
        raise NotImplementedError

    def plot(self):
        centers = self._get_centers()

        radius_margin = self.radii + self.meta_gap
        xlim = (
            (centers[:, 0] - radius_margin).min(),
            (centers[:, 0] + radius_margin).max() + self.meta_gap,
        )
        ylim = (
            (centers[:, 1] - radius_margin).min(),
            (centers[:, 1] + radius_margin).max(),
        )
        lx, ly = xlim[1] - self.meta_gap, ylim[0] + self.meta_gap

        factor = 0.05
        fig, ax = plt.subplots(
            subplot_kw={"aspect": "equal"},
            figsize=(factor * (xlim[1] - xlim[0]), factor * (ylim[1] - ylim[0])),
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        for index, dataset in enumerate(self.datasets):
            dataset_kwargs = (
                dataset if isinstance(dataset, dict) else {"dataset": dataset}
            )
            dataset_bubble_plot(
                **dataset_kwargs,
                ax=ax,
                center=centers[index],
                legend=index == len(self.datasets) - 1,
                legend_position=(lx, ly),
                scale_ax=False,
                **self.kwargs,
            )
        return fig


class _ClusterDatasetPlotter(_BaseDatasetPlotter):
    def _get_centers(self):
        bubble_chart = _BubbleChart(self.areas, bubble_spacing=self.meta_gap)
        bubble_chart.collapse(n_iterations=self.collapse_iterations)
        return bubble_chart.get_centers()


class _GridDatasetPlotter(_BaseDatasetPlotter):
    def _get_centers(self):
        assert isinstance(self.n_col, int)
        height = self.radii.max() * 2
        indices = np.arange(len(self.datasets))
        x = indices % self.n_col
        y = -(indices // self.n_col)
        return np.stack([x, y], axis=1) * height


def plot_datasets_grid(
    datasets: list["BaseDataset" | dict] | None = None,
    n_col: int = 10,
    margin: float = 10.0,
    **kwargs,
):
    """Plots datasets on a deterministic grid layout."""
    plotter = _GridDatasetPlotter(
        datasets=datasets,
        meta_gap=margin,
        n_col=n_col,
        kwargs=kwargs,
    )
    return plotter.plot()


def plot_datasets_cluster(
    datasets: list["BaseDataset" | dict] | None = None,
    meta_gap: float = 10.0,
    collapse_iterations: int = 120,
    **kwargs,
):
    """Plots datasets clustered via iterative bubble collapse."""
    plotter = _ClusterDatasetPlotter(
        datasets=datasets,
        meta_gap=meta_gap,
        kwargs=kwargs,
        collapse_iterations=collapse_iterations,
    )
    return plotter.plot()

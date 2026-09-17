"""Diagnostic figures for five-scene SIFT mosaic."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def draw_footprints(
    scenes,
    output_path: str | Path,
    title: str = "Scene Footprints",
    corrected: bool = False,
    G: list[np.ndarray] | None = None,
) -> None:
    """Draw scene footprints as rectangles on a geographic plot.

    Args:
        scenes: List of Scene objects.
        output_path: Output PNG path.
        title: Plot title.
        corrected: If True, apply G corrections to footprints.
        G: Global world-correction matrices (required if corrected=True).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenes)))

    for i, s in enumerate(scenes):
        b = s.bounds["B14"]
        if corrected and G is not None:
            # Apply G to footprint corners
            g = G[i]
            corners = np.array([
                [b.left, b.bottom, 1],
                [b.right, b.top, 1],
            ]).T
            corrected_corners = g @ corners
            left = corrected_corners[0, 0]
            bottom = corrected_corners[1, 0]
            right = corrected_corners[0, 1]
            top = corrected_corners[1, 1]
        else:
            left, bottom = b.left, b.bottom
            right, top = b.right, b.top

        w = right - left
        h = top - bottom
        rect = Rectangle((left, bottom), w, h,
                         fill=False, edgecolor=colors[i],
                         linewidth=2, label=f"[{i}] {s.name[:30]}")
        ax.add_patch(rect)
        ax.text((left + right) / 2, (bottom + top) / 2, str(i),
                ha="center", va="center", fontsize=9, color=colors[i])

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Footprints plot saved: %s", output_path)


def draw_overlap_graph(
    edges,
    scenes,
    output_path: str | Path,
    title: str = "Geographic Overlap Graph",
) -> None:
    """Draw the overlap graph as a network diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    G_nx = nx.Graph()
    for i, s in enumerate(scenes):
        G_nx.add_node(i, label=s.name[:20])
    for e in edges:
        G_nx.add_edge(e.idx_i, e.idx_j,
                      weight=e.intersection_area / 1e6,
                      ratio=f"{e.overlap_ratio_i:.2f}/{e.overlap_ratio_j:.2f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G_nx, seed=42, k=2)
    nx.draw(G_nx, pos, ax=ax, with_labels=True,
            node_color="lightblue", node_size=800,
            font_size=10, font_weight="bold")
    edge_labels = {(u, v): d["ratio"] for u, v, d in G_nx.edges(data=True)}
    nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels, font_size=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Overlap graph plot saved: %s", output_path)


def draw_accepted_sift_graph(
    accepted,
    pairwise_results,
    scenes,
    output_path: str | Path,
    title: str = "Accepted SIFT Graph",
) -> None:
    """Draw accepted (OK) and failed SIFT edges."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    G_nx = nx.Graph()
    for i, s in enumerate(scenes):
        G_nx.add_node(i)

    accepted_pairs = {(r.idx_i, r.idx_j) for r in pairwise_results
                      if r.status == "OK"}
    failed_pairs = {(r.idx_i, r.idx_j) for r in pairwise_results
                    if r.status != "OK"}

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G_nx, seed=42, k=2)

    # Accepted edges in green
    for u, v in accepted_pairs:
        G_nx.add_edge(u, v, color="green")
    # Failed edges in red (dashed)
    for u, v in failed_pairs:
        G_nx.add_edge(u, v, color="red")

    edge_colors = [G_nx[u][v].get("color", "gray") for u, v in G_nx.edges()]
    edge_styles = ["dashed" if c == "red" else "solid" for c in edge_colors]

    nx.draw(G_nx, pos, ax=ax, with_labels=True,
            node_color="lightblue", node_size=800,
            font_size=10, font_weight="bold",
            edge_color=edge_colors, style=edge_styles,
            width=2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="Accepted (SIFT OK)"),
        Line2D([0], [0], color="red", lw=2, linestyle="dashed",
               label="Failed"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Accepted SIFT graph plot saved: %s", output_path)


def draw_spanning_tree(
    tree_edges,
    scenes,
    output_path: str | Path,
    title: str = "Maximum Spanning Tree",
) -> None:
    """Draw the spanning tree."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    G_nx = nx.DiGraph()
    for i in range(len(scenes)):
        G_nx.add_node(i)
    for e in tree_edges:
        G_nx.add_edge(e["parent"], e["child"],
                      weight=e["weight"])

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G_nx, seed=42, k=2)
    nx.draw(G_nx, pos, ax=ax, with_labels=True,
            node_color="lightgreen", node_size=800,
            font_size=10, font_weight="bold",
            arrows=True, arrowstyle="->", arrowsize=20)
    edge_labels = {(u, v): f"{d['weight']:.1f}"
                   for u, v, d in G_nx.edges(data=True)}
    nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels, font_size=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Spanning tree plot saved: %s", output_path)


def draw_global_consistency(
    consistency_results,
    output_path: str | Path,
    title: str = "Global Edge Consistency",
) -> None:
    """Bar chart of global residuals per edge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not consistency_results:
        logger.warning("No consistency results to plot")
        return

    labels = [f"{r['idx_i']}→{r['idx_j']}" for r in consistency_results]
    residuals = [r["global_residual_px"] for r in consistency_results]
    colors = ["green" if r["in_tree"] else "orange" for r in consistency_results]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, residuals, color=colors)
    ax.axhline(y=2.0, color="red", linestyle="--", label="P95 > 2 px")
    ax.set_ylabel("Global Residual (px)")
    ax.set_title(title)
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, residuals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Consistency plot saved: %s", output_path)
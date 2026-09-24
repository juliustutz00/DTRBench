#!pip install scikit-posthocs

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, linregress, friedmanchisquare, spearmanr
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp


sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2,
)

ANNOTATION_SIZE = 6

COLORBLIND_PALETTE = sns.color_palette("colorblind")
SELECTED_PALETTE = COLORBLIND_PALETTE

REP_NAMES = [
    "Feature Graph",
    "Topological Forest",
    "INDTree",
    "Leaf Profile",
    "Prediction Profile",
    "Tree Descriptor",
]

REP_ACRONYMS = {
    "Feature Graph": "FG",
    "Topological Forest": "TF",
    "INDTree": "ID",
    "Leaf Profile": "LP",
    "Prediction Profile": "PP",
    "Tree Descriptor": "TD",
}

#FG, TF, ID, LP, PP, TD

BASELINES = [
    "Top OOB MCC",
    "Top OOB ACC",
    "Random",
]
ORDER = BASELINES + REP_NAMES

SEL_STRATEGIES = [
    "k-medoid",
    "k-medoid-performance",
    "agglomerative",
    "agglomerative-performance",
    "density",
    "combination-greedy",
    "combination-simulated_annealing",
    "combination-genetic",
]

VALIDATION_DATASETS = [
    "cervical_cancer",
    "isolet",
    "musk_1",
    "waveform",
    "heart_disease",
]

# names and labels
STRATEGY_RENAME = {
    "k-medoid": "KM-S",
    "k-medoid-performance": "KMP-S",
    "agglomerative": "AG-S",
    "agglomerative-performance": "AGP-S",
    "density": "DE-S",
    "combination-greedy": "GR-S",
    #"combination-simulated_annealing": "SA-S",
    "combination-simulated_annealing": r"$\bf{SA}$-$\bf{S}$",
    "combination-genetic": "GE-S",
}

ALL_RENAME = {
    "Tree Descriptor": "TD",
    "Leaf Profile": "LP",
    "Feature Graph": "FG",
    "Topological Forest": "TF",
    "INDTree": "ID",
    "Prediction Profile": "PP",
    "Full Forest": "FF",
    "Top OOB MCC": "TOM",
    "Top OOB ACC": "TOA",
    "Random": "RDM",
    "Single DT": "SDT",
}

PERTURBATION_DISPLAY_NAMES = {
    "change_threshold": "change threshold",
    "change_feature": "change feature",
    "swap_nodes": "swap nodes",
    "remove_nodes": "remove nodes",
    "add_nodes": "add nodes",
}

PERTURBATIONS = [
    "change_threshold",
    "change_feature",
    "swap_nodes",
    "remove_nodes",
    "add_nodes",
]

BASELINES_OVER_K = ["Random", "Top OOB ACC", "Top OOB MCC"]

# colors
REPRESENTATION_COLORS = {
    "Feature Graph":SELECTED_PALETTE[2],
    "Topological Forest":SELECTED_PALETTE[3],
    "INDTree":SELECTED_PALETTE[4],
    "Leaf Profile":SELECTED_PALETTE[1],
    "Prediction Profile":SELECTED_PALETTE[5],
    "Tree Descriptor":SELECTED_PALETTE[0],
}

BASELINE_COLORS = {
    "Random": SELECTED_PALETTE[7],
    "Top OOB ACC": SELECTED_PALETTE[8],
    "Top OOB MCC": SELECTED_PALETTE[9],
}

STRATEGY_COLORS = {
    strategy: SELECTED_PALETTE[i % len(SELECTED_PALETTE)]
    for i, strategy in enumerate(SEL_STRATEGIES)
}

SERIES_COLORS = {
    **REPRESENTATION_COLORS,
    **BASELINE_COLORS,
    **STRATEGY_COLORS,
    "Full Forest": "black",
    "Single DT": "black",
}



def rename_strategy(strategy):
    return STRATEGY_RENAME.get(strategy, strategy)


def rename_representation(rep_name):
    return REP_ACRONYMS.get(rep_name, rep_name)


def rename_perturbation(pert_name):
    return PERTURBATION_DISPLAY_NAMES.get(pert_name, pert_name.replace("_", " ").title())



def _pearson_statistics(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)

    if n < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": n, "r": np.nan, "p": np.nan, "se": np.nan}

    r, p = pearsonr(x, y)
    se = np.sqrt((1 - r**2) / (n - 2)) if n > 2 else np.nan

    return {"n": n, "r": r, "p": p, "se": se}


def _fisher_z_combine_from_z(z, var_z):
    z = np.asarray(z, dtype=float)
    var_z = np.asarray(var_z, dtype=float)

    valid = np.isfinite(z) & np.isfinite(var_z) & (var_z > 0)
    z, var_z = z[valid], var_z[valid]

    if len(z) == 0:
        return {
            "r": np.nan, "se": np.nan,
            "ci_low": np.nan, "ci_high": np.nan,
            "q": np.nan, "df": np.nan, "i2": np.nan,
            "k": 0,
        }

    w = 1.0 / var_z

    z_bar = np.sum(w * z) / np.sum(w)
    se_z = 1.0 / np.sqrt(np.sum(w))
    z_lo, z_hi = z_bar - 1.96 * se_z, z_bar + 1.96 * se_z

    q = np.sum(w * (z - z_bar) ** 2)
    df = len(z) - 1
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0

    return {
        "r": np.tanh(z_bar),
        "se": se_z,
        "ci_low": np.tanh(z_lo),
        "ci_high": np.tanh(z_hi),
        "q": q,
        "df": df,
        "i2": i2,
        "k": len(z),
    }


def _fisher_z_combine(rs, ns):
    rs = np.asarray(rs, dtype=float)
    ns = np.asarray(ns, dtype=float)

    valid = np.isfinite(rs) & np.isfinite(ns) & (ns > 3)
    rs, ns = rs[valid], ns[valid]

    if len(rs) == 0:
        return _fisher_z_combine_from_z([], [])

    z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    var_z = 1.0 / (ns - 3)

    return _fisher_z_combine_from_z(z, var_z)


def calculate_similarity_correlation_data(
    data,
    rep_names,
    cluster_cols=("dataset", "fold"),
    per_perturbation=True,
    aggregation="one_stage",
    dataset_col="dataset",
):

    if aggregation not in ("one_stage", "two_stage"):
        raise ValueError("aggregation must be 'one_stage' or 'two_stage'")
    if aggregation == "two_stage" and dataset_col not in cluster_cols:
        raise ValueError(
            f"aggregation='two_stage' requires '{dataset_col}' to be one of "
            f"cluster_cols so per-dataset grouping is well-defined; got "
            f"cluster_cols={cluster_cols}"
        )

    grouping_cols = list(cluster_cols) + (["perturbation"] if per_perturbation else [])

    plot_data = []
    cluster_rows = []

    performance_difference = (
    pd.to_numeric(data["performance_base"], errors="coerce")
    - pd.to_numeric(data["performance_perturbed"], errors="coerce")
    )
    feature_importance_difference = pd.to_numeric(
        data["feature_importance_difference"], errors="coerce"
    )

    # --- Stage 0: per-cluster correlations (each cluster is [dataset, fold, perturbation]) ---
    for rep_name in rep_names:
        similarity_column = f"sim_{rep_name}"
        delta_similarity = 1.0 - pd.to_numeric(data[similarity_column], errors="coerce")

        for value, label in (
            (performance_difference, "Performance"),
            (feature_importance_difference, "Feature Importance"),
        ):
            mask = np.isfinite(delta_similarity) & np.isfinite(value)

            df = pd.DataFrame({
                "similarity": delta_similarity[mask],
                "value": value[mask],
                "representation": rep_name,
                "type": label,
                "perturbation": data.loc[mask, "perturbation"].values,
                "intensity": data.loc[mask, "intensity"].values,
                **{c: data.loc[mask, c].values for c in cluster_cols},
            })
            plot_data.append(df)

            cell_stats = (
                df.groupby(grouping_cols)
                .apply(lambda g: pd.Series(_pearson_statistics(g["similarity"], g["value"])))
                .reset_index()
            )
            cell_stats = cell_stats[np.isfinite(cell_stats["r"])].assign(
                representation=rep_name, type=label
            )
            cluster_rows.append(cell_stats)

    plot_data = pd.concat(plot_data, ignore_index=True)
    cluster_statistics = pd.concat(cluster_rows, ignore_index=True)

    top_group_cols = ["representation", "type"] + ["perturbation"]

    if aggregation == "one_stage":
        # --- Single Fisher-z combination straight from stage-0 clusters ---
        combined_rows = []
        for keys, group in cluster_statistics.groupby(top_group_cols):
            combined = _fisher_z_combine(group["r"], group["n"])
            key_dict = dict(zip(top_group_cols, keys if isinstance(keys, tuple) else (keys,)))
            combined_rows.append({**key_dict, **combined})

        statistics = pd.DataFrame(combined_rows)

        return {
            "plot_data": plot_data,
            "cluster_statistics": cluster_statistics,
            "statistics": statistics,
        }

    else:  # two_stage
        # --- Stage 1: combine folds -> one estimate per dataset ---
        dataset_group_cols = top_group_cols + [dataset_col]
        dataset_rows = []
        for keys, group in cluster_statistics.groupby(dataset_group_cols):
            combined = _fisher_z_combine(group["r"], group["n"])
            key_dict = dict(zip(dataset_group_cols, keys if isinstance(keys, tuple) else (keys,)))
            # rename i2 -> i2_fold to make clear this is within-dataset
            # (fold-to-fold) heterogeneity, distinct from the stage-2 i2
            combined["i2_fold"] = combined.pop("i2")
            dataset_rows.append({**key_dict, **combined})

        dataset_statistics = pd.DataFrame(dataset_rows)

        # --- Stage 2: combine datasets -> one overall estimate ---
        combined_rows = []
        for keys, group in dataset_statistics.groupby(top_group_cols):
            z = np.arctanh(np.clip(group["r"], -0.999999, 0.999999))
            var_z = group["se"] ** 2
            combined = _fisher_z_combine_from_z(z, var_z)
            combined["i2_dataset"] = combined.pop("i2")
            key_dict = dict(zip(top_group_cols, keys if isinstance(keys, tuple) else (keys,)))
            combined_rows.append({**key_dict, **combined})

        statistics = pd.DataFrame(combined_rows)

        return {
            "plot_data": plot_data,
            "cluster_statistics": cluster_statistics,
            "dataset_statistics": dataset_statistics,
            "statistics": statistics,
        }


def pivot_correlation_table(statistics, value_col="r"):
    rep_order = REP_NAMES
    pert_order = ["change_threshold", "change_feature", "swap_nodes",
                  "remove_nodes", "add_nodes"]
    type_map = {"Performance": "r_MCC", "Feature Importance": "r_FI"}

    df = statistics.copy()
    df["type"] = df["type"].map(type_map)
    df["representation"] = pd.Categorical(df["representation"], rep_order, ordered=True)
    df["perturbation"] = pd.Categorical(df["perturbation"], pert_order, ordered=True)

    wide = (
        df.pivot_table(
            index="perturbation",
            columns=["type", "representation"], 
            values=value_col,
        )
        .reindex(columns=pd.MultiIndex.from_product([["r_MCC", "r_FI"], rep_order]))
    )
    wide.columns = pd.MultiIndex.from_product([["r_MCC", "r_FI"], [REP_ACRONYMS[n] for n in rep_order]])
    wide.loc["add_nodes", "r_MCC"] = np.nan

    return wide.round(3)


def _apply_acronyms(result):
    if result.get("mean_ranks") is None:
        return result

    renamed = dict(result)  # shallow copy; only touch the two renamed fields
    renamed["mean_ranks"] = result["mean_ranks"].rename(index=REP_ACRONYMS)

    if result.get("nemenyi_pvalues") is not None:
        nemenyi = result["nemenyi_pvalues"].rename(index=REP_ACRONYMS, columns=REP_ACRONYMS)
        renamed["nemenyi_pvalues"] = nemenyi

    if result.get("best") is not None:
        renamed["best"] = REP_ACRONYMS.get(result["best"], result["best"])

    return renamed


def compare_representations_one_cell(cluster_statistics, rep_names, alpha=0.05):
    present_reps = [r for r in rep_names if r in cluster_statistics["representation"].unique()]
    excluded = [r for r in rep_names if r not in present_reps]

    z = cluster_statistics.assign(
        z=np.arctanh(cluster_statistics["r"].clip(-0.999999, 0.999999))
    )
    wide = z.pivot_table(
        index=["dataset", "fold"], columns="representation", values="z"
    )[present_reps].dropna()

    if len(present_reps) < 2 or len(wide) < 2:
        return {
            "best": None, "n_blocks": len(wide),
            "excluded_representations": excluded,
            "note": "insufficient data for comparison",
        }

    stat, p_friedman = friedmanchisquare(*[wide[r] for r in present_reps])

    # rank descending (1 = highest correlation) within each block
    ranks = wide.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean().sort_values()
    best_rep = mean_ranks.index[0]

    if p_friedman >= alpha or len(present_reps) < 3:
        significantly_worse_than_best = []
        nemenyi = None
    else:
        nemenyi = sp.posthoc_nemenyi_friedman(wide[present_reps].values)
        nemenyi.columns = nemenyi.index = present_reps
        best_idx = present_reps.index(best_rep)
        significantly_worse_than_best = [
            r for r in present_reps
            if r != best_rep and nemenyi.loc[best_rep, r] < alpha
        ]

    not_significantly_different_from_best = [
        r for r in present_reps
        if r != best_rep and r not in significantly_worse_than_best
    ]

    return {
        "best": best_rep,
        "best_mean_rank": mean_ranks[best_rep],
        "n_blocks": len(wide),
        "friedman_p": p_friedman,
        "significantly_worse_than_best": significantly_worse_than_best,
        "not_significantly_different_from_best": not_significantly_different_from_best,
        "excluded_representations": excluded,
        "mean_ranks": mean_ranks,
        "nemenyi_pvalues": nemenyi,
    }


def _bold_acronym_labels(ax):
    acronym_set = set(REP_ACRONYMS.values())
    for text in ax.texts:
        content = text.get_text().strip()
        if any(
            content == a or content.startswith(a + " ") or content.endswith(" " + a)
            for a in acronym_set
        ):
            text.set_fontweight("bold")


def plot_critical_difference_grid(cluster_statistics, rep_names, alpha=0.05,
                                    save_path=None, legend_cell=("Performance", "add_nodes")):

    legend_cell = tuple(legend_cell) if legend_cell is not None else None

    all_cells = list(
        cluster_statistics[["type", "perturbation"]]
        .drop_duplicates()
        .sort_values(["type", "perturbation"])
        .itertuples(index=False, name=None)
    )

    plot_palette = {REP_ACRONYMS.get(k, k): v for k, v in REPRESENTATION_COLORS.items()}

    n = len(all_cells)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 2.2 * n_rows))
    axes = np.atleast_1d(axes).flatten(order="F")

    skipped = []

    for ax, (type_, pert) in zip(axes, all_cells):
        if (type_, pert) == legend_cell:
            ax.axis("off")
            handles = [
                plt.Line2D(
                    [0], [0], marker="o", linestyle="", markersize=8,
                    color=plot_palette.get(acr, "black"),
                    label=f"{acr} – {full}",
                )
                for full, acr in REP_ACRONYMS.items()
            ]
            ax.legend(
                handles=handles, loc="center", frameon=False, fontsize=12,
                title="Representations", title_fontsize=12,
            )
            continue

        subset = cluster_statistics[
            (cluster_statistics["type"] == type_)
            & (cluster_statistics["perturbation"] == pert)
        ]
        result = compare_representations_one_cell(subset, rep_names, alpha=alpha)
        if result.get("nemenyi_pvalues") is None:
            skipped.append((type_, pert, result.get("note", "non-significant or <3 reps")))
            ax.axis("off")
            continue

        result = _apply_acronyms(result)
        sp.critical_difference_diagram(
            result["mean_ranks"], result["nemenyi_pvalues"], ax=ax,
            color_palette=plot_palette,
        )
        _bold_acronym_labels(ax)

        excl_note = ""
        if result["excluded_representations"]:
            excl = [REP_ACRONYMS.get(r, r) for r in result["excluded_representations"]]
            excl_note = f" (excl.: {', '.join(excl)})"
        title = f"{type_} | {rename_perturbation(pert)}{excl_note}"
        #title = f"{rename_perturbation(pert)}{excl_note}"
        ax.set_title(title, fontsize=13)

    # hide any trailing unused axes beyond the number of actual cells
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()

    if skipped:
        print("Skipped (no valid CD diagram):")
        for type_, pert, note in skipped:
            print(f"  {type_} / {pert}: {note}")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_representative_scatter(
    plot_data,
    combos,
    type_="Feature Importance",
    intensity_col="intensity",
    palette=None,
    point_size=14,
    point_alpha=0.55,
    fit_trend=True,
    n_cols=4,
    figsize_per_panel=(2.2, 3.3),
    fontsize_base=11,
    save_path=None,
    share_y=True,
    ylim=(-0.7, 1.1),
):

    n = len(combos)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharey=share_y,
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    axes_flat = axes.flatten()

    intensities = sorted(plot_data[intensity_col].dropna().unique())
    if palette is None:
        cmap = sns.color_palette("flare", as_cmap=True)
        # sample n evenly-spaced points along the continuous colormap,
        # one per intensity level, so the discrete legend still reflects
        # the sequential palette's light-to-dark ordering
        n_levels = len(intensities)
        sample_points = (
            np.linspace(0.15, 0.95, n_levels) if n_levels > 1 else [0.6]
        )
        palette = {
            level: cmap(pos) for level, pos in zip(intensities, sample_points)
        }

    for ax, (rep, pert) in zip(axes_flat, combos):
        subset = plot_data[
            (plot_data["representation"] == rep)
            & (plot_data["perturbation"] == pert)
            & (plot_data["type"] == type_)
        ]

        if subset.empty:
            ax.set_title(f"{rep} | {pert}\n(no data)", fontsize=fontsize_base)
            ax.axis("off")
            continue

        for level in intensities:
            level_data = subset[subset[intensity_col] == level]
            if level_data.empty:
                continue
            ax.scatter(
                [], [], color=palette[level], s=point_size * 2.2,
                label=f"{level:.1f}",
            )  # legend-only proxy artist; real points drawn below

        shuffled = subset.sample(frac=1, random_state=0)
        colors = shuffled[intensity_col].map(palette)
        ax.scatter(
            shuffled["similarity"], shuffled["value"],
            color=colors, s=point_size, alpha=point_alpha, linewidths=0,
        )

        if fit_trend:
            valid = subset.dropna(subset=["similarity", "value"])
            if len(valid) > 2 and valid["similarity"].std() > 0:
                fit = linregress(valid["similarity"], valid["value"])
                x_line = np.array([valid["similarity"].min(), valid["similarity"].max()])
                y_line = fit.intercept + fit.slope * x_line
                ax.plot(x_line, y_line, color="black", linewidth=1.6, zorder=4)


        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", zorder=0)
        #ax.set_title(f"{rep} | {pert.replace('_', ' ')}", fontsize=fontsize_base + 1, pad=8)
        ax.set_title(f"{rename_perturbation(pert)}", fontsize=fontsize_base + 1, pad=8)
        ax.set_xlabel("Δ Representation", fontsize=fontsize_base)
        ax.tick_params(labelsize=fontsize_base - 2)
        ax.locator_params(axis="x", nbins=5)
        if not share_y:
            ax.locator_params(axis="y", nbins=5)

        # visible box around each panel
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("0.3")

    # y-axis: shared range/ticks, label and tick labels only on the
    # leftmost column of each row to avoid repeating them n_cols times
    y_label = "FI difference" if type_ == "Feature Importance" else "Δ MCC"
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx >= n:
                continue
            ax = axes[row, col]
            if share_y:
                ax.set_ylim(*ylim)
                ax.locator_params(axis="y", nbins=5)
                if col == 0:
                    ax.set_ylabel(y_label, fontsize=fontsize_base)
                else:
                    ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(y_label, fontsize=fontsize_base)

    for ax in axes_flat[n:]:
        ax.axis("off")

    # single shared legend for intensity levels (deduplicated across panels)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(),
        loc="lower center",
        ncol=len(intensities),
        fontsize=fontsize_base,
        bbox_to_anchor=(0.57, -0.07),
        frameon=False,
        markerscale=1.5,
    )

    fig.text(
        0.28, -0.005, "Intensity",
        ha="right", va="center",
        fontsize=fontsize_base,
        #fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def calculate_similarity_intensity_data(data):

    data["mcc_diff_abs"] = (data["performance_base"] - data["performance_perturbed"]).abs()

    dissim_cols = [f"dissim_{rep}" for rep in REP_NAMES]

    perturbations_present = [
        perturbation
        for perturbation in PERTURBATIONS
        if perturbation in data["perturbation"].dropna().unique()
    ]

    data = data.copy()
    for rep, dissim_col in zip(REP_NAMES, dissim_cols):
        data[dissim_col] = 1.0 - pd.to_numeric(data[f"sim_{rep}"], errors="coerce")

    aggregation_columns = dissim_cols + ["mcc_diff_abs", "feature_importance_difference"]

    aggregated = (
        data[data["perturbation"].isin(perturbations_present)]
        .groupby(["perturbation", "intensity"])[aggregation_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    # flatten MultiIndex columns
    aggregated.columns = ["perturbation", "intensity"] + [
        f"{column}_{statistic}" for column, statistic in aggregated.columns[2:]
    ]

    return {
        "data": aggregated,
        "perturbations": perturbations_present,
        "representation_columns": dissim_cols,
    }


def plot_similarity_vs_intensity_per_perturbation(similarity_intensity_data, save_path=None):
    aggregated = similarity_intensity_data["data"]
    perturbations_present = similarity_intensity_data["perturbations"]
    representation_columns = similarity_intensity_data["representation_columns"]

    if not perturbations_present:
        print("No perturbations available for plotting.")
        return

    n_perturbations = len(perturbations_present)

    fig, axes = plt.subplots(
        1, n_perturbations,
        figsize=(max(2.0 * n_perturbations, 5), 2.5),
        sharey=True,
    )

    if n_perturbations == 1:
        axes = [axes]

    for ax, perturbation in zip(axes, perturbations_present):
        subset = aggregated[aggregated["perturbation"] == perturbation].sort_values("intensity")

        for dissim_column, rep_name in zip(representation_columns, REP_NAMES):
            mean_column = f"{dissim_column}_mean"

            if mean_column not in subset:
                continue

            color = REPRESENTATION_COLORS.get(rep_name, "gray")

            ax.plot(
                subset["intensity"], subset[mean_column],
                marker="o", markersize=3.5, color=color,
                label=rep_name, linewidth=1.4,
            )

        ax.set_title(rename_perturbation(perturbation), fontsize=13)
        ax.set_xlabel("Intensity", fontsize=11)

        if ax == axes[0]:
            ax.set_ylabel("Δ Representation", fontsize=13)

        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)

        # visible box around each panel
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("0.3")
        ax.set_box_aspect(1)

    fig.subplots_adjust(wspace=0.08)
    fig.tight_layout()

    handles, labels = axes[0].get_legend_handles_labels()
    labels = [REP_ACRONYMS.get(l, l) for l in labels]  # remap to acronyms for legend
    legend = fig.legend(
        handles, labels, loc="lower center", ncol=6, fontsize=13,
        bbox_to_anchor=(0.6, -0.2), frameon=True,
    )

    fig.text(
        0.26, -0.1, "Representation",
        ha="right", va="center",
        fontsize=12,
        #fontweight="bold",
    )
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def compute_recovery(df):
    full_forest_mcc = (
        df[df["representation"] == "Full Forest"]
        .set_index(["dataset", "seed", "fold"])["mcc"]
        .rename("full_forest_mcc")
    )
    out = df.join(full_forest_mcc, on=["dataset", "seed", "fold"])

    valid = (
        out["full_forest_mcc"].notna()
        & np.isfinite(out["full_forest_mcc"])
        & (out["full_forest_mcc"] != 0)
    )

    out["mcc_recovery"] = np.nan
    out.loc[valid, "mcc_recovery"] = out.loc[valid, "mcc"] / out.loc[valid, "full_forest_mcc"]

    out["mcc_recovery"] = out["mcc_recovery"].where(np.isfinite(out["mcc_recovery"]), np.nan)

    out.loc[out["representation"] == "Full Forest", "mcc_recovery"] = 1.0

    return out


def collapse_replicates(df):
    group_cols = [
        "dataset", "seed", "fold", "representation",
        "selection_strategy", "subforest_size",
    ]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in group_cols]
    return df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()


def calculate_representation_strategy_table(df, value_col, representations=REP_NAMES,
                                             strategies=SEL_STRATEGIES):
    subset = df[
        df["representation"].isin(representations)
        & df["selection_strategy"].isin(strategies)
    ]

    table = subset.pivot_table(
        index="representation", columns="selection_strategy",
        values=value_col, aggfunc="mean",
    )

    return table.reindex(index=representations, columns=strategies)


def build_recovery_heatmap_tables(splits, metrics,
                                   representations=REP_NAMES, strategies=SEL_STRATEGIES):
    tables = {}
    for split_name, split_df in splits:
        for value_col, metric_label in metrics:
            tables[(split_name, metric_label)] = calculate_representation_strategy_table(
                split_df, value_col, representations, strategies
            )
    return tables

def plot_recovery_heatmaps(
    tables,
    splits,
    metrics,
    save_path=None,
    cmap="cividis",
    percentile_range=(2, 98)
):

    n_rows, n_cols = len(metrics), len(splits)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        sharey=True,
        sharex=True,
    )

    axes = np.atleast_2d(axes)

    for row, (_, metric_label) in enumerate(metrics):
        for col, (split_name, _) in enumerate(splits):

            ax = axes[row, col]

            table = tables[
                (split_name, metric_label)
            ].copy()

            table.index = [
                rename_representation(r)
                for r in table.index
            ]

            table.columns = [
                rename_strategy(s)
                for s in table.columns
            ]

            # Average across representations
            avg_row = table.mean(axis=0).to_frame().T
            avg_row.index = ["mean"]

            table_plot = pd.concat(
                [table, avg_row]
            )

            # Color scale based on actual metric values
            values = table_plot.to_numpy(dtype=float)

            lower_pct, upper_pct = percentile_range

            vmin = np.nanpercentile(
                values,
                lower_pct,
            )

            vmax = np.nanpercentile(
                values,
                upper_pct,
            )

            # Annotations: actual values only
            annot = table_plot.map(
                lambda x: f"{x:.3f}"
            )

            sns.heatmap(
                table_plot,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                annot=annot,
                fmt="",
                square=True,
                cbar=False,
                linewidths=0.5,
                linecolor="white",
                annot_kws={
                    "size": ANNOTATION_SIZE + 2,
                },
            )

            # Box around Average row
            nrows, ncols = table_plot.shape

            ax.add_patch(
                Rectangle(
                    (0, nrows - 1),
                    ncols,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5,
                    clip_on=False,
                )
            )

            ax.set_title(
                f"{metric_label} — {split_name} datasets",
                fontsize=12,
            )

            ax.set_xlabel("")

            ax.set_ylabel(
                "Representation" if col == 0 else "",
                fontsize=12,
                labelpad=0,
            )

            ax.set_xlabel(
                "Selection strategy" if row == n_rows - 1 else "",
                fontsize=12,
                labelpad=10,
            )

            ax.tick_params(
                axis="x",
                rotation=90,
                labelsize=12,
                bottom=False,
            )

            ax.tick_params(
                axis="y",
                rotation=0,
                labelsize=12,
            )

    fig.subplots_adjust(
        wspace=0.12,
        hspace=0.12,
    )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def build_method_table(df, strategy_map, baselines=BASELINES_OVER_K):
    rep_rows = []
    for rep, strategy in strategy_map.items():
        subset = df[
            (df["representation"] == rep) & (df["selection_strategy"] == strategy)
        ].copy()
        subset["method"] = rep
        rep_rows.append(subset)

    baseline_rows = df[df["representation"].isin(baselines)].copy()
    baseline_rows["method"] = baseline_rows["representation"]

    return pd.concat(rep_rows + [baseline_rows], ignore_index=True)


def compute_mean_min_k(df, thresholds, full_forest_size=1000, unit_cols=("dataset",)):

    unit_cols = list(unit_cols)
    all_units = df[unit_cols].drop_duplicates()
    n_units_total = len(all_units)

    rows = []
    for threshold in thresholds:
        for method in df["representation"].unique():
            min_k_per_unit = (
                df[
                    (df["representation"] == method)
                    & (df["mcc_recovery"] >= threshold)
                ]
                .groupby(unit_cols)["subforest_size"]
                .min()
            )
            # reindex against the full set of units so missing ones -> NaN
            min_k_per_unit = (
                all_units.merge(
                    min_k_per_unit.reset_index(), on=unit_cols, how="left"
                )["subforest_size"]
            )

            n_censored = min_k_per_unit.isna().sum()
            censored_values = min_k_per_unit.fillna(full_forest_size)

            rows.append({
                "threshold": threshold,
                "representation": method,
                "mean_min_k": censored_values.mean(),
                "n_censored": n_censored,
                "n_units": n_units_total,
            })

    return pd.DataFrame(rows)

def get_min_k_at_threshold(data, threshold, full_forest_size=1000, ratio=False):
    min_k_table = (
        data[data["mcc_recovery"] >= threshold]
        .groupby(["representation", "dataset"])["subforest_size"]
        .min()
        .unstack("dataset")
        .reindex(columns=data["dataset"].unique())
        .fillna(full_forest_size)
    )
    if ratio:
        min_k_table = min_k_table / full_forest_size
    return min_k_table


def compute_min_k_auc(summary_censored, log_scale=True):
    rows = []
    for method, group in summary_censored.groupby("representation"):
        group = group.sort_values("threshold")
        y = np.log10(group["mean_min_k"]) if log_scale else group["mean_min_k"]
        auc = np.trapz(y, group["threshold"])
        rows.append({"representation": method, "auc": auc, "log_scale": log_scale})

    return pd.DataFrame(rows).sort_values("auc")


def plot_min_k_auc_bar(
    auc_df,
    ylim=(None, None),
    palette=None,
    ax=None,
    save_path=None
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.2, 4.5))

    ymin, ymax = ylim

    # Use custom palette if provided, otherwise SERIES_COLORS
    if palette is None:
        colors = [SERIES_COLORS[m] for m in auc_df["representation"]]
    else:
        colors = [palette.get(m, "gray") for m in auc_df["representation"]]

    ax.bar(
        auc_df["representation"],
        auc_df["auc"],
        color=colors,
        alpha=0.85,
    )

    ax.set_xticks(range(len(auc_df)))
    ax.set_xticklabels(
        auc_df["representation"].map(ALL_RENAME),
        fontsize=11,
        rotation=90,
        ha="center",
    )

    label = r"Area under $\log_{10}(\mathrm{min}\ k)$-vs-threshold curve"
    ax.set_ylabel(label, fontsize=11)
    ax.set_xlabel("Selection approach", labelpad=10)

    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("0.3")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax


def get_inference_time_at_min_k(method_means, threshold, inf_time_path, ratio=False):

    inf_time = pd.read_csv(inf_time_path)
    min_k_table = get_min_k_at_threshold(method_means, threshold, ratio=False)

    baseline = inf_time[inf_time["k"] == 1000].set_index("dataset")["inference_time"]
    inf_time["normalized_time"] = inf_time["inference_time"] / inf_time["dataset"].map(baseline)

    long_min_k = (
            min_k_table.drop(index="Full Forest", errors="ignore")
            .reset_index()
            .rename(columns={"index": "representation"} if "representation" not in min_k_table.reset_index().columns else {})
            .melt(id_vars=min_k_table.index.name or "representation", var_name="dataset", value_name="k")
            .rename(columns={min_k_table.index.name or "representation": "method"})
        )
    merged = long_min_k.merge(inf_time, on=["dataset", "k"], how="left") 
    if ratio:
        inf_time_scaled = merged.pivot(index="method", columns="dataset", values="normalized_time")
    else:
        inf_time_scaled = merged.pivot(index="method", columns="dataset", values="inference_time")
    return inf_time_scaled


def plot_min_k_heatmap(
    summary,
    stat="median",
    ax=None,
    cmap="RdYlGn_r",
    save_path=None
):

    col_name = f"{stat}_min_k"

    pivot_val = summary.pivot(
        index="representation",
        columns="threshold",
        values=col_name,
    )

    # Order rows by their value at the strictest threshold, best first
    row_order = pivot_val.iloc[:, -1].sort_values().index
    pivot_val = pivot_val.loc[row_order]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(
                0.4 * pivot_val.shape[1] + 2,
                0.4 * pivot_val.shape[0] + 2,
            )
        )

    # Log-transform for the color scale
    log_values = np.log10(pivot_val.values)

    im = ax.imshow(
        log_values,
        cmap=cmap,
        aspect="auto",
    )

    # Annotate cells with original k values
    for i in range(pivot_val.shape[0]):
        for j in range(pivot_val.shape[1]):
            val = pivot_val.values[i, j]

            if np.isnan(val):
                ax.text(
                    j,
                    i,
                    "--",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="gray",
                )
                continue

            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=14,
                color="black",
                #fontweight="bold",
            )

    # Axes
    ax.set_xticks(range(pivot_val.shape[1]))
    ax.set_xticklabels(
        (pivot_val.columns * 100).astype(int),
        fontsize=13,
    )

    ax.set_yticks(range(pivot_val.shape[0]))
    ax.set_yticklabels(
        pivot_val.index.map(ALL_RENAME),
        fontsize=13,
    )

    ax.set_xlabel(
        "MCC threshold (%)",
        fontsize=15,
        labelpad=10,
    )
    ax.set_ylabel(
        "Selection approach",
        fontsize=15,
        labelpad=10,
    )
    # ax.set_title(
    #     f"{stat.capitalize()} minimum subforest size to reach each normalized MCC threshold",
    #     fontsize=14,
    #     pad=15,
    # )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("0.3")
    #ax.set_box_aspect(1)
    # Remove grid
    ax.grid(False)
    ax.minorticks_off()

    # No colorbar
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax, pivot_val


def plot_subforest_selection_at_threshold(
    df,
    plot_type="bar",
    ylabel="Score",
    figsize=(4.5, 3.9),
    sort=True,
    ascending=True,
    save_path=None,
):

    df = df.copy()
    df.columns.name = "dataset"
    df.index.name = "representation"

    df_long = (
        df.reset_index()
          .melt(
              id_vars="representation",
              var_name="dataset",
              value_name="score"
          )
    )

    # Sort representations by mean score
    if sort:
        order = (
            df_long.groupby("representation")["score"]
            .mean()
            .sort_values(ascending=ascending)
            .index
        )
    else:
        order = df_long["representation"].unique()

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "bar":

        sns.barplot(
            data=df_long,
            x="representation",
            y="score",
            order=order,
            estimator="mean",
            errorbar="se",
            capsize=0.15,
            ax=ax,
            palette=SERIES_COLORS
        )

    elif plot_type == "swarm":

        sns.swarmplot(
            data=df_long,
            x="representation",
            y="score",
            order=order,
            size=4,
            ax=ax,
            palette=SERIES_COLORS,
        )

    elif plot_type == "violin":

        sns.violinplot(
            data=df_long,
            x="representation",
            y="score",
            order=order,
            inner="box",
            cut=0,
            ax=ax,
            palette=SERIES_COLORS
        )

    elif plot_type == "box":

        sns.boxplot(
            data=df_long,
            x="representation",
            y="score",
            order=order,
            ax=ax,
            palette=SERIES_COLORS,
        )

    else:
        raise ValueError(
            "plot_type must be one of: "
            "'bar', 'swarm', 'violin', 'box'"
        )

    # Formatting
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("0.3")

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Selection approach")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [ALL_RENAME.get(name, name) for name in order],
        rotation=0
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def compute_fi_spearman_corr(df):
    df = df.copy()
    df["fi_arr"] = _parse_fi(df["feature_importances"])

    full_forest_fi = (
        df[df["representation"] == "Full Forest"]
        .set_index(["dataset", "seed", "fold"])["fi_arr"]
        .rename("full_forest_fi_arr")
    )
    # guard against duplicate (dataset, seed, fold) rows for Full Forest
    full_forest_fi = full_forest_fi[~full_forest_fi.index.duplicated(keep="first")]

    out = df.join(full_forest_fi, on=["dataset", "seed", "fold"])

    def _row_corr(row):
        a, b = row["fi_arr"], row["full_forest_fi_arr"]
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return np.nan
        min_len = min(len(a), len(b))
        if min_len < 2:
            return np.nan
        a, b = a[:min_len], b[:min_len]
        if np.all(a == a[0]) or np.all(b == b[0]):
            return np.nan
        corr, _ = spearmanr(a, b)
        return corr

    out["fi_spearman_corr"] = out.apply(_row_corr, axis=1)
    out.loc[out["representation"] == "Full Forest", "fi_spearman_corr"] = 1.0

    return out


def _parse_fi(series):
    def parse_one(x):
        if pd.isnull(x):
            return None
        try:
            return np.array(ast.literal_eval(str(x)), dtype=float)
        except Exception:
            return None
    return series.apply(parse_one)


def _kendall_w(fi_list):
    valid = [f for f in fi_list if f is not None]
    if len(valid) < 2:
        return np.nan
    min_len = min(len(v) for v in valid)
    valid = [v[:min_len] for v in valid]
    rankings = np.array([pd.Series(v).rank(ascending=False).values for v in valid])
    m, n = rankings.shape
    if m < 2 or n < 2:
        return np.nan
    column_sums = rankings.sum(axis=0)
    S = np.sum((column_sums - column_sums.mean()) ** 2)
    return (12 * S) / (m**2 * (n**3 - n))


def compute_stability_recovery(df):
    df = df.copy()
    df["fi_arr"] = _parse_fi(df["feature_importances"])

    config_w = (
        df[df["representation"] != "Full Forest"]
        .groupby(["dataset", "representation", "subforest_size"])["fi_arr"]
        .apply(lambda g: _kendall_w(g.tolist()))
        .reset_index(name="kendall_w")
    )

    ff_w = (
        df[df["representation"] == "Full Forest"]
        .groupby("dataset")["fi_arr"]
        .apply(lambda g: _kendall_w(g.tolist()))
        .reset_index(name="full_forest_kendall_w")
    )

    merged = config_w.merge(ff_w, on="dataset", how="left")
    valid = (
        merged["full_forest_kendall_w"].notna()
        & np.isfinite(merged["full_forest_kendall_w"])
        & (merged["full_forest_kendall_w"] != 0)
    )
    merged["stability_recovery"] = np.nan
    merged.loc[valid, "stability_recovery"] = (
        merged.loc[valid, "kendall_w"] / merged.loc[valid, "full_forest_kendall_w"]
    )
    return merged


def build_unified_table(df, best_overall):

    subforest_bs = df[
    ~df["dataset"].isin(VALIDATION_DATASETS) # compute recovery on test datasets only
    & ~df["selection_strategy"].isin( # compute recovery on best selection strategy or baseline only
        [s for s in SEL_STRATEGIES if s != best_overall]
    )
    ]

    subforest_bs = compute_fi_spearman_corr(subforest_bs)

    row_level = (
        subforest_bs[subforest_bs["representation"] != "Full Forest"]
        .groupby(["dataset", "representation", "subforest_size"])
        .agg(
            mcc_recovery=("mcc_recovery", "mean"),
            agreement_with_full_forest=("agreement_with_full_forest", "mean"),
            fi_spearman_corr=("fi_spearman_corr", "mean"),
        )
        .reset_index()
    )

    stability = compute_stability_recovery(subforest_bs)[
        ["dataset", "representation", "subforest_size", "stability_recovery"]
    ]

    return row_level.merge(stability, on=["dataset", "representation", "subforest_size"], how="left")


def stratified_mean(unified, metric, size_group):
    subset = unified[unified["subforest_size"].isin(size_group)]
    per_dataset = (
        subset.groupby(["representation", "dataset"])[metric].mean().reset_index()
    )
    return per_dataset.groupby("representation")[metric].mean()


def build_results_matrix(unified, size_list):
    columns = {}

    for metric in ["mcc_recovery", "agreement_with_full_forest", "stability_recovery", "fi_spearman_corr"]:
        for label, group in size_list.items():
            col_name = f"{metric}\n{label}"
            columns[col_name] = stratified_mean(unified, metric, group)

    results = pd.DataFrame(columns)
    results = results.reindex(ORDER).dropna()
    results.index = results.index.map(ALL_RENAME)
    return results


def declutter_and_label(ax, entries, min_gap_frac=0.055):
    if not entries:
        return

    to_axes = ax.transAxes.inverted()
    positioned = []
    for x, y, text, color in entries:
        disp = ax.transData.transform((x, y))
        _, fy = to_axes.transform(disp)
        fy = min(max(fy, 0.03), 0.97)
        positioned.append([fy, text, color])

    positioned.sort(key=lambda e: e[0])
    for i in range(1, len(positioned)):
        if positioned[i][0] - positioned[i - 1][0] < min_gap_frac:
            positioned[i][0] = positioned[i - 1][0] + min_gap_frac

    for fy, text, color in positioned:
        ax.annotate(
            text, xy=(1.0, fy), xycoords="axes fraction",
            xytext=(1.04, fy), textcoords="axes fraction",
            fontsize=8.5, color=color, va="center", ha="left",
            annotation_clip=False,
        )


def plot_resource_benchmark_panels(panels, reference_forest_size=50, rep_names=None, dpi=200, save_path=None):

    rep_names = rep_names if rep_names is not None else REP_NAMES

    n_rows = len(panels)
    n_cols = len(panels[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 9.5))
    pending_labels = {}

    for row, row_panels in enumerate(panels):
        for col, panel in enumerate(row_panels):
            ax = axes[row, col]

            if panel is None:
                ax.axis("off")
                continue

            value_col, ylabel, phase_label, data, reps_override = panel
            row_reps = reps_override or rep_names

            if value_col not in data.columns:
                ax.axis("off")
                ax.text(0.5, 0.5, "N/A\n(no stored artifact\nfor this phase)",
                        ha="center", va="center", fontsize=10, color="gray",
                        transform=ax.transAxes)
                continue

            plot_data = calculate_resource_plot_data(data, row_reps, value_col)
            scaling = calculate_scaling_exponents(
                data, value_col, row_reps, reference_forest_size
            ).set_index("representation")

            positive_vals = plot_data.loc[plot_data[value_col] > 0, value_col]
            floor = positive_vals.min() / 10 if len(positive_vals) else 1e-6
            plot_data = plot_data.copy()
            plot_data[value_col] = plot_data[value_col].clip(lower=floor)

            label_entries = []
            for rep in row_reps:
                rep_plot = plot_data[plot_data["representation"] == rep].sort_values("random_forest_size")
                if rep_plot.empty:
                    continue

                color = REPRESENTATION_COLORS[rep]
                ax.plot(
                    rep_plot["random_forest_size"], rep_plot[value_col],
                    marker="o", markersize=4, linewidth=1.8, color=color,
                    label=rep,
                )

                alpha = scaling.loc[rep, "alpha_mean"] if rep in scaling.index else np.nan
                if np.isfinite(alpha):
                    x_end = rep_plot["random_forest_size"].iloc[-1]
                    y_end = rep_plot[value_col].iloc[-1]
                    label_entries.append((x_end, y_end, f"{alpha:.2f}", color))

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.margins(x=0.08)
            ax.tick_params(labelsize=10)

            # visible box around each panel
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color("0.3")

            ax.set_title(phase_label, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11.5)
            if row == n_rows - 1:
                ax.set_xlabel("Random Forest size (# trees)", fontsize=11.5)

            pending_labels[(row, col)] = label_entries

    # finalize the layout before computing any axes-fraction label positions
    fig.tight_layout(rect=[0, 0, 0.87, 0.93])

    handles, labels = axes[0, 0].get_legend_handles_labels()

    labels = [
        REP_ACRONYMS.get(label, label)
        for label in labels
    ]
    fig.legend(handles, labels, loc="upper center", ncol=len(rep_names),
               bbox_to_anchor=(0.46, 0.98), fontsize=11, frameon=False)
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    fig.canvas.draw()

    for (row, col), entries in pending_labels.items():
        declutter_and_label(axes[row, col], entries)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def calculate_resource_plot_data(
    data,
    REP_NAMES,
    value_col,
):

    plot_data = (
        data[
            data["representation"].isin(REP_NAMES)
        ]
        .groupby(
            [
                "representation",
                "random_forest_size",
            ],
            as_index=False,
        )[value_col]
        .mean()
    )

    return plot_data


def calculate_scaling_exponents(
    data,
    value_col,
    REP_NAMES,
    reference_forest_size=50,
):

    results = []

    for representation in REP_NAMES:
        fold_results = []

        rep_data = data[
            data["representation"] == representation
        ]

        for fold, fold_data in rep_data.groupby("fold"):
            fold_data = fold_data[
                [
                    "random_forest_size",
                    value_col,
                ]
            ].dropna()

            # Log-log regression requires strictly positive values
            fold_data = fold_data[
                (fold_data["random_forest_size"] > 0)
                & (fold_data[value_col] > 0)
            ]

            if len(fold_data) < 2:
                continue

            result = linregress(
                np.log(fold_data["random_forest_size"]),
                np.log(fold_data[value_col]),
            )

            # resource cost at the reference forest size
            reference_cost = fold_data.loc[
                fold_data["random_forest_size"] == reference_forest_size,
                value_col,
            ]

            if len(reference_cost) == 0:
                reference_cost = np.nan
            else:
                reference_cost = reference_cost.mean()

            fold_results.append(
                {
                    "fold": fold,
                    "alpha": result.slope,
                    "r_squared": result.rvalue ** 2,
                    "reference_cost": reference_cost,
                }
            )

        # no valid folds for this representation
        if not fold_results:
            results.append(
                {
                    "representation": representation,
                    "alpha_mean": np.nan,
                    "alpha_std": np.nan,
                    "r_squared_mean": np.nan,
                    "reference_cost_mean": np.nan,
                    "reference_cost_std": np.nan,
                    "n_folds": 0,
                }
            )
            continue

        fold_results_df = pd.DataFrame(fold_results)

        results.append(
            {
                "representation": representation,
                "alpha_mean": fold_results_df["alpha"].mean(),
                "alpha_std": fold_results_df["alpha"].std(),
                "r_squared_mean": fold_results_df["r_squared"].mean(),
                "reference_cost_mean": (
                    fold_results_df["reference_cost"].mean()
                ),
                "reference_cost_std": (
                    fold_results_df["reference_cost"].std()
                ),
                "n_folds": len(fold_results_df),
            }
        )

    results_df = pd.DataFrame(results)

    return results_df

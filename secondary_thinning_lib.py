# secondary_thinning_lib.py
# ---------------------------------------------------------------------
# Complete forest-thinning utility module.
# strategies, helpers, metrics tables, and visualization.
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


__all__ = [
    # Strategies
    "thin_from_below_adjacent_simple",
    "thin_from_above_neighbors",                            # Thin from above-1
    "thin_from_above_phase2_anchor_immediate5",             # Thin from above-2 (anchor)
    "thin_from_above2_tiered_immediate5",                   # Thin from above-2 (tiered)
    # Reports / metrics
    "table_final_vs_initial",
    "table_final_vs_after_first",
    "top25_release_table",
    "anchor_release_table_immediate5",
    # Viz
    "plot_thinning_map",
    "center_figure_html",
]


# ==============================
# Helpers (distance, windows)
# ==============================
def _euclid2d(a_rows: np.ndarray, a_trees: np.ndarray,
              b_rows: np.ndarray, b_trees: np.ndarray,
              row_scale: float = 1.0, tree_scale: float = 1.0) -> np.ndarray:
    """Pairwise Euclidean distance on (row, tree) grid."""
    dr = (a_rows[:, None] - b_rows[None, :]) * row_scale
    dt = (a_trees[:, None] - b_trees[None, :]) * tree_scale
    return np.sqrt(dr * dr + dt * dt)


def _build_immediate5_windows(baseline_df: pd.DataFrame,
                              anchor_ids: pd.Index,
                              *,
                              row_col: str,
                              tree_col: str,
                              k_neighbors: int = 5,
                              row_scale: float = 1.0,
                              tree_scale: float = 1.0) -> Dict[int, np.ndarray]:
    """
    For each anchor (by index), return up to 5 nearest neighbors from the residual baseline,
    including anchors and excluding self.
    """
    B_ids = baseline_df.index.to_numpy()
    B_rows = baseline_df[row_col].to_numpy(float)
    B_trees = baseline_df[tree_col].to_numpy(float)
    id2pos = {B_ids[i]: i for i in range(len(B_ids))}
    out: Dict[int, np.ndarray] = {}
    for a in anchor_ids:
        ai = id2pos[a]
        D = _euclid2d(np.array([B_rows[ai]]), np.array([B_trees[ai]]),
                      B_rows, B_trees, row_scale=row_scale, tree_scale=tree_scale).ravel()
        D[ai] = np.inf
        k = min(int(k_neighbors), len(B_ids) - 1)
        order = np.argpartition(D, k - 1)[:k]
        order = order[np.argsort(D[order])]
        out[a] = B_ids[order]
    return out


# ==============================
# Strategy: Thin from below
# ==============================
def thin_from_below_adjacent_simple(df_best3: pd.DataFrame,
                                    fraction: float = 1/3,
                                    metric: str = 'pre_DBH',
                                    row_col: str = 'Row',
                                    status_col: str = 'status'):
    """
    From a stand that's already 3-row thinned (df_best3), remove the lowest `fraction`
    of DBHs on the rows immediately adjacent to each corridor (row-1 and row+1),
    considering only Alive & currently Keep trees.
    """
    d = df_best3.copy()
    alive = d[status_col].eq('Alive')

    # Corridor rows
    corridor_mask = alive & d['thin_decision'].eq('Thin')
    corridor_rows = np.sort(d.loc[corridor_mask, row_col].unique())

    if len(corridor_rows) == 0:
        return d, {'corridor_rows': [], 'side_rows': [], 'n_removed_below': 0}

    all_rows = np.sort(d[row_col].unique())
    all_rows_set = set(all_rows)

    # Side rows: rows ±1 of each corridor row
    side_rows = set()
    for rc in corridor_rows:
        if (rc - 1) in all_rows_set:
            side_rows.add(rc - 1)
        if (rc + 1) in all_rows_set:
            side_rows.add(rc + 1)
    side_rows = sorted([r for r in side_rows if r not in set(corridor_rows)])

    to_remove_idx: List[pd.Index] = []
    for r in side_rows:
        # Eligible: Alive, currently Keep, and on this side row
        elig = alive & d['thin_decision'].eq('Keep') & d[row_col].eq(r)
        sub = d.loc[elig, [metric]]
        n = len(sub)
        if n == 0:
            continue
        k_rm = int(np.floor(n * fraction))
        if k_rm <= 0:
            continue

        # Pick the k_rm smallest DBHs on this side row
        idx = sub.nsmallest(k_rm, metric).index
        to_remove_idx.append(idx)

    if len(to_remove_idx) > 0:
        to_remove_idx = pd.Index(np.concatenate([ix.values for ix in to_remove_idx]))
        d.loc[to_remove_idx, 'thin_decision'] = 'Thin'
        n_removed_below = int(len(to_remove_idx))
    else:
        n_removed_below = 0

    info = {
        'corridor_rows': corridor_rows.tolist(),
        'side_rows': side_rows,
        'n_removed_below': n_removed_below
    }
    return d, info


# ==============================
# Strategy: Thin from above-1
# ==============================
def thin_from_above_neighbors(df_best3: pd.DataFrame,
                              removal_fraction: float = 1/3,
                              metric: str = 'pre_DBH',
                              row_col: str = 'Row',
                              x_col: str = 'Tree',
                              status_col: str = 'status',
                              thin_col: str = 'thin_decision',
                              keep_val: str = 'Keep',
                              thin_val: str = 'Thin',
                              anchor_fraction: float = 0.10,
                              min_anchors: int = 1,
                              radius: Optional[int] = None,
                              combine: str = 'max'):
    """
    Per side-row:
      1) Choose anchors = top DBH trees (Alive & Keep) by anchor_fraction (>= min_anchors).
      2) Score non-anchors by influence of nearby anchors:
         score = max_or_sum( anchor_DBH / (|x_i - x_anchor| + 1) ).
      3) Thin top-scoring non-anchors up to quota = floor(removal_fraction * eligible_on_row).
    Anchors are never cut.
    """
    assert 0 < removal_fraction < 1
    assert 0 < anchor_fraction <= 1

    d0 = df_best3.copy()
    alive = d0[status_col].eq('Alive')

    corridor_mask = alive & d0[thin_col].eq(thin_val)
    corridor_rows = np.sort(d0.loc[corridor_mask, row_col].unique())
    if len(corridor_rows) == 0:
        return d0, {'corridor_rows': [], 'side_rows': [], 'per_row_quota': {}, 'per_row_removed': {}, 'anchors_per_row': {}}

    all_rows = np.sort(d0[row_col].unique())
    corridor_set = set(corridor_rows)

    corr_to_side: Dict[int, List[int]] = {}
    for rc in corridor_rows:
        side = []
        if (rc - 1) in all_rows and (rc - 1) not in corridor_set: side.append(rc - 1)
        if (rc + 1) in all_rows and (rc + 1) not in corridor_set: side.append(rc + 1)
        corr_to_side[rc] = side

    side_rows_all = sorted({r for rows in corr_to_side.values() for r in rows})
    per_row_quota: Dict[int, int] = {}
    per_row_eligible_idx: Dict[int, pd.Index] = {}
    per_row_anchors: Dict[int, pd.Index] = {}

    for r in side_rows_all:
        elig = alive & d0[thin_col].eq(keep_val) & d0[row_col].eq(r)
        idx = d0.index[elig]
        n = len(idx)
        if n == 0:
            per_row_quota[r] = 0
            per_row_eligible_idx[r] = idx
            per_row_anchors[r] = pd.Index([])
            continue
        quota = int(np.floor(n * removal_fraction))
        quota = max(0, min(quota, n))
        per_row_quota[r] = quota

        sub = d0.loc[idx, [metric, x_col]]
        k_anchor = max(min_anchors, int(np.ceil(n * anchor_fraction)))
        k_anchor = min(k_anchor, n)
        anchors_idx = sub.nlargest(k_anchor, metric).index
        per_row_anchors[r] = anchors_idx
        per_row_eligible_idx[r] = idx

    # Influence scores
    scores: Dict[int, float] = {}
    for _, side_rows in corr_to_side.items():
        for r in side_rows:
            idx = per_row_eligible_idx[r]
            if len(idx) == 0:
                continue
            sub = d0.loc[idx, [metric, x_col]]
            anchors_idx = per_row_anchors[r]
            if len(anchors_idx) == 0:
                continue
            anchors = d0.loc[anchors_idx, [metric, x_col]]

            for i, row_i in sub.iterrows():
                if i in anchors_idx:
                    continue
                xi = float(row_i[x_col])
                inf_vals = []
                for _, row_a in anchors.iterrows():
                    dx = abs(xi - float(row_a[x_col]))
                    if (radius is not None) and (dx > radius):
                        continue
                    inf_vals.append(float(row_a[metric]) / (dx + 1.0))
                if not inf_vals:
                    continue
                s_i = max(inf_vals) if combine == 'max' else sum(inf_vals)
                if i in scores:
                    scores[i] = max(scores[i], s_i) if combine == 'max' else (scores[i] + s_i)
                else:
                    scores[i] = s_i

    selected_idx: List[int] = []
    per_row_removed: Dict[int, int] = {}
    for r in side_rows_all:
        quota = per_row_quota[r]
        if quota <= 0:
            per_row_removed[r] = 0
            continue

        anchors_idx = set(per_row_anchors[r])
        row_idxs = [i for i in per_row_eligible_idx[r] if (i not in anchors_idx) and (i in scores)]
        if len(row_idxs) == 0:
            per_row_removed[r] = 0
            continue

        row_scores = pd.Series({i: scores[i] for i in row_idxs}).sort_values(ascending=False)
        take = row_scores.index[:quota]
        selected_idx.extend(take.tolist())
        per_row_removed[r] = int(len(take))

    df_out = df_best3.copy()
    if selected_idx:
        df_out.loc[selected_idx, thin_col] = thin_val

    info = {
        'corridor_rows': list(map(int, corridor_rows.tolist())),
        'side_rows': list(map(int, side_rows_all)),
        'per_row_quota': {int(k): int(v) for k, v in per_row_quota.items()},
        'anchors_per_row': {int(r): list(map(int, per_row_anchors[r])) for r in side_rows_all},
        'per_row_removed': {int(k): int(v) for k, v in per_row_removed.items()},
        'n_removed_total': int(len(selected_idx)),
        'params': {
            'removal_fraction': removal_fraction,
            'anchor_fraction': anchor_fraction,
            'min_anchors': min_anchors,
            'radius': radius,
            'combine': combine,
            'metric': metric,
            'x_col': x_col
        }
    }
    return df_out, info


# ==============================
# Strategy: Thin from above-2 (anchor)
# ==============================
def thin_from_above_phase2_anchor_immediate5(
    df_best3: pd.DataFrame,
    *,
    dbh_col: str = 'pre_DBH',
    row_col: str = 'Row',
    tree_col: str = 'Tree',
    status_col: str = 'status',
    thin_col: str = 'thin_decision',
    keep_val: str = 'Keep',
    thin_val: str = 'Thin',
    # anchors & window
    top_pct_anchors: float = 0.10,   # fraction of residual chosen as anchors
    k_neighbors: int = 5,            # immediate window size (<=5 if not enough neighbors exist)
    # budget
    stand_target_removed_frac: float = 1/3,
    target_rounding: str = 'round',  # 'round' | 'floor' | 'ceil'
    reserve_for_phaseB: float = 0.0, # portion of budget held for Phase B
    # distance scaling
    row_scale: float = 1.0,
    tree_scale: float = 1.0,
    # verbosity
    verbose: bool = False
):
    """
     immediate-5 logic:
      • Build, for each anchor, the list of its 5 nearest residual neighbors (Alive&Keep after 3-row),
        INCLUDING other anchors; exclude self. This window is FIXED for all phases.
      • CUTTING may thin only those neighbors in the window that are NON-anchors (anchors are skipped).
      • COUNTING uses the SAME window; sides released = # of those 5 that end up Thin.
    """
    d0 = df_best3.copy()

    baseline_mask = d0[status_col].eq('Alive') & d0[thin_col].eq(keep_val)
    baseline_idx  = d0.index[baseline_mask]
    n_base = len(baseline_idx)
    if n_base == 0:
        return d0, {'error': 'Empty residual baseline after 3-row'}

    baseline = d0.loc[baseline_idx]

    # Stand-wide budget
    if target_rounding == 'floor':
        target_total = int(np.floor(n_base * stand_target_removed_frac))
    elif target_rounding == 'ceil':
        target_total = int(np.ceil(n_base * stand_target_removed_frac))
    else:
        target_total = int(np.round(n_base * stand_target_removed_frac))
    target_total = max(0, min(target_total, n_base))

    reserve = int(np.floor(float(reserve_for_phaseB) * target_total))
    phaseA_cap = max(0, target_total - reserve)

    # Anchors: top X% by DBH within residual baseline
    n_anchors = max(1, int(np.ceil(n_base * float(top_pct_anchors))))
    anchors_idx = baseline.nlargest(n_anchors, dbh_col).index
    anchors_df  = baseline.loc[anchors_idx].sort_values(dbh_col, ascending=False)
    anchors_set = set(anchors_idx)

    # Build the immediate-5 windows
    windows = _build_immediate5_windows(baseline, anchors_df.index,
                                        row_col=row_col, tree_col=tree_col,
                                        k_neighbors=k_neighbors,
                                        row_scale=row_scale, tree_scale=tree_scale)

    # Cuttable set per anchor = window minus anchors
    cuttable5 = {a: [nid for nid in windows[a] if nid not in anchors_set] for a in anchors_df.index}
    all_cuttable = pd.Index(np.unique([nid for lst in cuttable5.values() for nid in lst]))
    max_possible_from_windows = int(len(all_cuttable))

    d1 = d0.copy()

    # Phase A: largest anchors first, cut only within their immediate-5
    picksA: List[int] = []
    budgetA = phaseA_cap
    for a in anchors_df.index:
        if budgetA <= 0:
            break
        for nid in cuttable5[a]:
            if budgetA <= 0:
                break
            if d1.at[nid, thin_col] != keep_val:
                continue
            d1.at[nid, thin_col] = thin_val
            picksA.append(nid)
            budgetA -= 1

    removedA = int(len(picksA))
    remaining = target_total - removedA
    remaining = max(0, remaining)

    # Phase B: top-up strictly within the same immediate-5 windows
    picksB: List[int] = []
    phaseB_triggered = remaining > 0
    if phaseB_triggered:
        for a in anchors_df.index:
            if remaining <= 0:
                break
            for nid in cuttable5[a]:
                if remaining <= 0:
                    break
                if d1.at[nid, thin_col] == keep_val:
                    d1.at[nid, thin_col] = thin_val
                    picksB.append(nid)
                    remaining -= 1

    # Count release using the immediate-5 windows
    def _count_release_buckets(df_view: pd.DataFrame) -> Dict[int,int]:
        buckets = {k: 0 for k in range(0, k_neighbors+1)}
        for a in anchors_df.index:
            window_ids = windows[a]
            if len(window_ids) == 0:
                buckets[0] += 1
                continue
            k_cut = int(df_view.loc[window_ids, thin_col].eq(thin_val).sum())
            k_cut = int(max(0, min(k_cut, k_neighbors)))
            buckets[k_cut] += 1
        return buckets

    buckets_after = _count_release_buckets(d1)

    info = {
        'baseline_count': int(n_base),
        'target_removed_total': int(target_total),
        'removed_phaseA': removedA,
        'removed_phaseB': int(len(pd.Index(picksB).unique())),
        'removed_total': int(removedA + len(pd.Index(picksB).unique())),
        'unused_budget_after_B': int(remaining),
        'phaseB_triggered': bool(phaseB_triggered),
        'anchors_count': int(len(anchors_idx)),
        'capacity_immediate5_nonanchors_unique': int(max_possible_from_windows),
        'release_buckets_after': {int(k): int(v) for k, v in buckets_after.items()},
    }

    if verbose:
        print(f"[TFA2-immediate5] baseline={n_base} target={target_total} "
              f"A={removedA} B={len(picksB)} unused={remaining} "
              f"anchors={len(anchors_idx)} capacity={max_possible_from_windows} "
              f"release_after={buckets_after}")

    return d1, info


# ==============================
# Strategy: Thin from above-2 (Tiered)
# ==============================
def thin_from_above2_tiered_immediate5(
    df_best3: pd.DataFrame,
    *,
    # columns
    dbh_col: str = 'pre_DBH',
    vol_col: str = 'pre_stem_vol',
    row_col: str = 'Row',
    tree_col: str = 'Tree',
    status_col: str = 'status',
    thin_col: str = 'thin_decision',
    keep_val: str = 'Keep',
    thin_val: str = 'Thin',
    # windows
    k_neighbors: int = 5,
    row_scale: float = 1.0, tree_scale: float = 1.0,
    # tiers
    tier1_frac: float = 0.10,              # top 10% = Tier-1 (protected)
    tier2_frac: float = 0.15,              # next 15% = Tier-2 (may be cut by Tier-1; protected in Phase 2)
    # budget
    stand_target_removed_frac: float = 1/3,
    target_rounding: str = 'round',        # 'round' | 'floor' | 'ceil'
    reserve_for_tier2_ratio: float = 0.30, # portion of budget saved for Phase 2
    # verbosity
    verbose: bool = False
):
    """
    Phase 0: build tiers on the post–3-row residual.
    Phase 1: Tier-1 anchors (largest→smallest). In immediate-5, cut any neighbor not Tier-1.
    Phase 2: Surviving Tier-2 anchors (largest→smallest). In immediate-5, cut neighbors that are
             neither Tier-1 nor surviving Tier-2.
    """
    d0 = df_best3.copy()

    baseline_mask = d0[status_col].eq('Alive') & d0[thin_col].eq(keep_val)
    baseline_idx  = d0.index[baseline_mask]
    if len(baseline_idx) == 0:
        return d0, {'error': 'Empty baseline after 3-row'}

    baseline = d0.loc[baseline_idx]
    n_base   = len(baseline_idx)

    # Budget
    if target_rounding == 'floor':
        target_total = int(np.floor(n_base * stand_target_removed_frac))
    elif target_rounding == 'ceil':
        target_total = int(np.ceil(n_base * stand_target_removed_frac))
    else:
        target_total = int(np.round(n_base * stand_target_removed_frac))
    target_total = max(0, min(target_total, n_base))
    phase1_cap   = int(max(0, target_total * (1.0 - float(reserve_for_tier2_ratio))))

    # Tiers by DBH
    baseline_sorted = baseline.sort_values(dbh_col, ascending=False)
    n_t1      = max(1, int(np.ceil(n_base * float(tier1_frac))))
    tier1_ids = baseline_sorted.index[:n_t1]

    n_t2_add   = max(0, int(np.ceil(n_base * float(tier2_frac))))
    tier2_pool = baseline_sorted.index.difference(tier1_ids)
    tier2_ids  = tier2_pool[:n_t2_add]

    tier1_set = set(tier1_ids)
    tier2_set = set(tier2_ids)

    # Windows
    windows_t1 = _build_immediate5_windows(baseline, tier1_ids, row_col=row_col, tree_col=tree_col,
                                           k_neighbors=k_neighbors, row_scale=row_scale, tree_scale=tree_scale)
    windows_t2 = _build_immediate5_windows(baseline, tier2_ids, row_col=row_col, tree_col=tree_col,
                                           k_neighbors=k_neighbors, row_scale=row_scale, tree_scale=tree_scale)

    # Phase 1 (Tier-1 release)
    d1 = d0.copy()
    picks1: List[int] = []
    for a in tier1_ids:
        if len(picks1) >= phase1_cap:
            break
        for nid in windows_t1[a]:
            if len(picks1) >= phase1_cap:
                break
            if nid in tier1_set:
                continue  # never cut Tier-1
            if d1.at[nid, thin_col] == keep_val:
                d1.at[nid, thin_col] = thin_val
                picks1.append(nid)

    removed1 = int(len(picks1))
    remaining_budget = max(0, target_total - removed1)

    # Surviving Tier-2 after Phase 1
    tier2_survivors = [t for t in tier2_ids if d1.at[t, thin_col] == keep_val]
    t2_survivor_set = set(tier2_survivors)

    # Phase 2 (Tier-2 release)
    picks2: List[int] = []
    if remaining_budget > 0 and len(tier2_survivors) > 0:
        t2_sorted = baseline.loc[tier2_survivors].sort_values(dbh_col, ascending=False).index
        for a in t2_sorted:
            if len(picks2) >= remaining_budget:
                break
            for nid in windows_t2[a]:
                if len(picks2) >= remaining_budget:
                    break
                if (nid in tier1_set) or (nid in t2_survivor_set):
                    continue  # protect Tier-1 and surviving Tier-2
                if d1.at[nid, thin_col] == keep_val:
                    d1.at[nid, thin_col] = thin_val
                    picks2.append(nid)

    removed2      = int(len(picks2))
    total_removed = removed1 + removed2
    unused_budget = max(0, target_total - total_removed)

    info = {
        'baseline_count': int(n_base),
        'target_removed_total': int(target_total),
        'removed_phase1': removed1,
        'removed_phase2': removed2,
        'removed_total': total_removed,
        'unused_budget': int(unused_budget),
        'tier1': {'count': int(len(tier1_ids)), 'ids': list(map(int, tier1_ids))},
        'tier2': {'count': int(len(tier2_ids)), 'ids': list(map(int, tier2_ids)),
                  'survivors_after_p1': int(len(tier2_survivors))}
    }
    if verbose:
        print(f"[TFA2-Tiered] base={n_base} target={target_total} "
              f"P1={removed1} P2={removed2} total={total_removed} unused={unused_budget} "
              f"T1={len(tier1_ids)} T2={len(tier2_ids)}")
    return d1, info


# ==============================
# Reporting / Metrics
# ==============================
def _stand_metrics_relative(base_df: pd.DataFrame, final_df: pd.DataFrame, base_mask: pd.Series, strategy: str,
                            metric: str = 'pre_DBH', vol_col: str = 'pre_stem_vol',
                            status_col: str = 'status', thin_col: str = 'thin_decision',
                            keep_val: str = 'Keep', thin_val: str = 'Thin') -> Dict[str, float]:

    base_idx = base_df.index[base_mask]
    base_idx = base_idx.intersection(final_df.index)

    if len(base_idx) == 0:
        return {
            'Strategy': strategy,
            'Trees removed(%)': 0.0, 'Volume removed(%)': 0.0,
            'Trees kept': 0, 'Trees removed': 0,
            'Q1 cut (count)': 0,
            'Q4 remaining (count)': 0, 'Q4 cut (count)': 0,
            'Removed volume-Q4(%)': 0.0, 'Removed volume-Q4': 0.0,
            'Change-Median DBH': np.nan, 'Change-Mean DBH': np.nan,
            'Volume removed': 0.0, 'Post-thinning total volume': 0.0,
            'Q1 removal ratio': np.nan, 'Q4 retention ratio': np.nan,
            'Pre-thinning Median DBH': np.nan, 'Pre-thinning Mean DBH': np.nan,
            'Post-thinning Median DBH': np.nan, 'Post-thinning Mean DBH': np.nan
        }

    pre = base_df.loc[base_idx]
    post = final_df.loc[base_idx]

    keep_mask = post[thin_col].eq(keep_val)
    cut_mask = post[thin_col].eq(thin_val)

    n_base = len(base_idx)
    n_kept = int(keep_mask.sum())
    n_cut = int(cut_mask.sum())

    base_vol = float(pre[vol_col].sum())
    vol_cut = float(pre.loc[cut_mask, vol_col].sum())
    vol_post = float(pre.loc[keep_mask, vol_col].sum())

    trees_removed_pct = 100 * (n_cut / n_base) if n_base else 0.0
    vol_removed_pct = 100 * (vol_cut / base_vol) if base_vol > 0 else 0.0

    x = pre[metric].astype(float)
    q1_thr = float(x.quantile(0.25))
    q3_thr = float(x.quantile(0.75))
    q1_mask = x <= q1_thr
    q4_mask = x >= q3_thr

    q1_cut = int((q1_mask & cut_mask).sum())
    q4_keep = int((q4_mask & keep_mask).sum())
    q4_cut = int((q4_mask & cut_mask).sum())

    vol_cut_q4 = float(pre.loc[q4_mask & cut_mask, vol_col].sum())
    vol_cut_q4_pct = 100 * (vol_cut_q4 / vol_cut) if vol_cut > 0 else 0.0

    pre_med = float(x.median()); pre_mean = float(x.mean())
    x_post = pre.loc[keep_mask, metric].astype(float)
    post_med = float(x_post.median()) if len(x_post) > 0 else np.nan
    post_mean = float(x_post.mean()) if len(x_post) > 0 else np.nan

    change_med = post_med - pre_med if pd.notnull(post_med) else np.nan
    change_mean = post_mean - pre_mean if pd.notnull(post_mean) else np.nan

    base_q1 = int(q1_mask.sum()); base_q4 = int(q4_mask.sum())
    q1_removal_ratio = (q1_cut / base_q1) if base_q1 > 0 else np.nan
    q4_retention_ratio = (q4_keep / base_q4) if base_q4 > 0 else np.nan

    return {
        'Strategy': strategy,
        'Trees removed(%)': trees_removed_pct,
        'Volume removed(%)': vol_removed_pct,
        'Trees kept': n_kept,
        'Trees removed': n_cut,
        'Q1 cut (count)': q1_cut,
        'Q4 remaining (count)': q4_keep,
        'Q4 cut (count)': q4_cut,
        'Removed volume-Q4(%)': vol_cut_q4_pct,
        'Removed volume-Q4': vol_cut_q4,
        'Change-Median DBH': change_med,
        'Change-Mean DBH': change_mean,
        'Volume removed': vol_cut,
        'Post-thinning total volume': vol_post,
        'Q1 removal ratio': q1_removal_ratio,
        'Q4 retention ratio': q4_retention_ratio,
        'Pre-thinning Median DBH': pre_med,
        'Pre-thinning Mean DBH': pre_mean,
        'Post-thinning Median DBH': post_med,
        'Post-thinning Mean DBH': post_mean
    }


def table_final_vs_initial(df_initial: pd.DataFrame, df_final: pd.DataFrame, *,
                           metric: str = 'pre_DBH', vol_col: str = 'pre_stem_vol',
                           status_col: str = 'status', thin_col: str = 'thin_decision',
                           keep_val: str = 'Keep', thin_val: str = 'Thin',
                           strategy: str = 'Final vs Initial (Alive baseline)') -> pd.DataFrame:

    base_mask = df_initial[status_col].eq('Alive')
    rep = _stand_metrics_relative(df_initial, df_final, base_mask, strategy,
                                  metric, vol_col, status_col, thin_col, keep_val, thin_val)
    return pd.DataFrame([rep])


def table_final_vs_after_first(df_after_first: pd.DataFrame, df_final: pd.DataFrame, *,
                               metric: str = 'pre_DBH', vol_col: str = 'pre_stem_vol',
                               status_col: str = 'status', thin_col: str = 'thin_decision',
                               keep_val: str = 'Keep', thin_val: str = 'Thin',
                               strategy: str = 'Final vs After 1st Thinning (Alive&Keep baseline)') -> pd.DataFrame:

    base_mask = df_after_first[status_col].eq('Alive') & df_after_first[thin_col].eq(keep_val)
    rep = _stand_metrics_relative(df_after_first, df_final, base_mask, strategy,
                                  metric, vol_col, status_col, thin_col, keep_val, thin_val)
    return pd.DataFrame([rep])


# ==============================
# Release tables 
# ==============================
def top25_release_table(df_after_first: pd.DataFrame, df_final: pd.DataFrame, *,
                        treatment: str,
                        dbh_col: str = 'pre_DBH',
                        row_col: str = 'Row',
                        tree_col: str = 'Tree',
                        status_col: str = 'status',
                        thin_col: str = 'thin_decision',
                        keep_val: str = 'Keep',
                        thin_val: str = 'Thin',
                        neighbors_k: int = 5,
                        row_scale: float = 1.0,
                        tree_scale: float = 1.0) -> pd.DataFrame:

    base_mask = df_after_first[status_col].eq('Alive') & df_after_first[thin_col].eq(keep_val)
    base_idx = df_after_first.index[base_mask]
    if len(base_idx) == 0:
        cols = ['Treatment','Initial tree quantity','Post-thin tree quantity',
                'pre-thin mean DBH','post-thin mean DBH',
                'Top 25% DBH trees (count)','Top 25% DBH trees cut off (count)'] + \
               [f'No. of Top 25% DBH trees with {k}-sided release' for k in [5,4,3,2,1,0]]
        return pd.DataFrame([dict(zip(cols, [treatment,0,0,np.nan,np.nan,0,0,0,0,0,0,0]))])

    d0 = df_after_first.loc[base_idx]
    d1 = df_final.loc[base_idx]

    initial_qty = len(base_idx)
    post_keep_mask = d1[thin_col].eq(keep_val)
    post_qty = int(post_keep_mask.sum())

    pre_mean_dbh  = float(d0[dbh_col].astype(float).mean())
    post_mean_dbh = float(d0.loc[post_keep_mask, dbh_col].astype(float).mean()) if post_qty > 0 else np.nan

    x = d0[dbh_col].astype(float)
    q3_thr = float(x.quantile(0.75))
    big_mask = x >= q3_thr
    big_idx = d0.index[big_mask]
    big_total = int(len(big_idx))

    if big_total == 0:
        out = {
            'Treatment': treatment,
            'Initial tree quantity': initial_qty,
            'Post-thin tree quantity': post_qty,
            'pre-thin mean DBH': pre_mean_dbh,
            'post-thin mean DBH': post_mean_dbh,
            'Top 25% DBH trees (count)': 0,
            'Top 25% DBH trees cut off (count)': 0,
            'No. of Top 25% DBH trees with 5-sided release': 0,
            'No. of Top 25% DBH trees with 4-sided release': 0,
            'No. of Top 25% DBH trees with 3-sided release': 0,
            'No. of Top 25% DBH trees with 2-sided release': 0,
            'No. of Top 25% DBH trees with 1-sided release': 0,
            'No. of Top 25% DBH trees with 0-sided release': 0
        }
        return pd.DataFrame([out])

    big_cut_bool  = d1.loc[big_idx, thin_col].eq(thin_val)
    big_cut_ids   = big_cut_bool[big_cut_bool].index
    big_cut_count = int(len(big_cut_ids))
    big_alive_idx = big_idx.difference(big_cut_ids)

    base_rows  = d0[row_col].to_numpy(float)
    base_trees = d0[tree_col].to_numpy(float)
    base_ids   = np.array(list(base_idx))
    id_to_pos  = {base_ids[pos]: pos for pos in range(len(base_ids))}

    buckets = {k: 0 for k in range(0, neighbors_k+1)}

    for bid in big_alive_idx:
        pos = id_to_pos.get(bid, None)
        if pos is None:
            continue

        r0 = float(base_rows[pos]); t0 = float(base_trees[pos])
        dr = (base_rows - r0) * row_scale
        dt = (base_trees - t0) * tree_scale
        dist = np.sqrt(dr*dr + dt*dt)
        dist[pos] = np.inf

        k = min(int(neighbors_k), len(base_idx) - 1)
        if k <= 0:
            buckets[0] += 1
            continue

        nn_pos = np.argpartition(dist, k-1)[:k]
        nn_ids = base_ids[nn_pos]

        nn_cut = int(d1.loc[nn_ids, thin_col].eq(thin_val).sum())
        nn_cut = max(0, min(nn_cut, neighbors_k))
        buckets[nn_cut] += 1

    out = {
        'Treatment': treatment,
        'Initial tree quantity': initial_qty,
        'Post-thin tree quantity': post_qty,
        'pre-thin mean DBH': pre_mean_dbh,
        'post-thin mean DBH': post_mean_dbh,
        'Top 25% DBH trees (count)': big_total,
        'Top 25% DBH trees cut off (count)': big_cut_count,
        'No. of Top 25% DBH trees with 5-sided release': buckets[5],
        'No. of Top 25% DBH trees with 4-sided release': buckets[4],
        'No. of Top 25% DBH trees with 3-sided release': buckets[3],
        'No. of Top 25% DBH trees with 2-sided release': buckets[2],
        'No. of Top 25% DBH trees with 1-sided release': buckets[1],
        'No. of Top 25% DBH trees with 0-sided release': buckets[0]
    }
    return pd.DataFrame([out])


def anchor_release_table_immediate5(
    df_after_first: pd.DataFrame, df_final: pd.DataFrame, *,
    treatment: str,
    top_pct_anchors: float = 0.10,
    neighbors_k: int = 5,
    dbh_col: str = 'pre_DBH',
    row_col: str = 'Row',
    tree_col: str = 'Tree',
    status_col: str = 'status',
    thin_col: str = 'thin_decision',
    keep_val: str = 'Keep',
    thin_val: str = 'Thin',
    row_scale: float = 1.0,
    tree_scale: float = 1.0
) -> pd.DataFrame:

    base_mask = df_after_first[status_col].eq('Alive') & df_after_first[thin_col].eq(keep_val)
    base_idx  = df_after_first.index[base_mask]
    if len(base_idx) == 0:
        cols = ['Treatment','Initial tree quantity','Post-thin tree quantity',
                'pre-thin mean DBH','post-thin mean DBH',
                'Anchors (count)','Anchors cut off (count)'] + \
               [f'No. of anchors with {k}-sided release' for k in [5,4,3,2,1,0]]
        return pd.DataFrame([dict(zip(cols,[treatment,0,0,np.nan,np.nan,0,0,0,0,0,0,0]))])

    d0 = df_after_first.loc[base_idx]
    d1 = df_final.loc[base_idx]

    initial_qty = len(base_idx)
    post_keep_mask = d1[thin_col].eq(keep_val)
    post_qty = int(post_keep_mask.sum())
    pre_mean_dbh  = float(d0[dbh_col].astype(float).mean())
    post_mean_dbh = float(d0.loc[post_keep_mask, dbh_col].astype(float).mean()) if post_qty>0 else np.nan

    n_base = len(base_idx)
    n_anchors = max(1, int(np.ceil(n_base * float(top_pct_anchors))))
    anchors_idx = d0.nlargest(n_anchors, dbh_col).index

    anchors_cut_count = int(d1.loc[anchors_idx, thin_col].eq(thin_val).sum())
    anchors_alive_idx = anchors_idx.difference(d1.index[d1[thin_col].eq(thin_val)])

    B_rows  = d0[row_col].to_numpy(float)
    B_trees = d0[tree_col].to_numpy(float)
    B_ids   = np.array(list(base_idx))
    id2pos  = {B_ids[i]: i for i in range(len(B_ids))}

    buckets = {k: 0 for k in range(0, neighbors_k+1)}

    for a in anchors_alive_idx:
        ai = id2pos[a]
        D = _euclid2d(np.array([B_rows[ai]]), np.array([B_trees[ai]]),
                      B_rows, B_trees, row_scale=row_scale, tree_scale=tree_scale).ravel()
        D[ai] = np.inf
        k = min(neighbors_k, len(B_ids)-1)
        order = np.argpartition(D, k-1)[:k]
        order = order[np.argsort(D[order])]
        window_ids = B_ids[order]
        k_cut = int(d1.loc[window_ids, thin_col].eq(thin_val).sum())
        k_cut = max(0, min(k_cut, neighbors_k))
        buckets[k_cut] += 1

    out = {
        'Treatment': treatment,
        'Initial tree quantity': initial_qty,
        'Post-thin tree quantity': post_qty,
        'pre-thin mean DBH': pre_mean_dbh,
        'post-thin mean DBH': post_mean_dbh,
        'Anchors (count)': int(len(anchors_idx)),
        'Anchors cut off (count)': anchors_cut_count,
        'No. of anchors with 5-sided release': buckets[5],
        'No. of anchors with 4-sided release': buckets[4],
        'No. of anchors with 3-sided release': buckets[3],
        'No. of anchors with 2-sided release': buckets[2],
        'No. of anchors with 1-sided release': buckets[1],
        'No. of anchors with 0-sided release': buckets[0]
    }
    return pd.DataFrame([out])


def _stand_metrics_relative(
    base_df, final_df, base_mask, strategy,
    metric='pre_DBH', vol_col='pre_stem_vol',
    status_col='status', thin_col='thin_decision',
    keep_val='Keep', thin_val='Thin',
    include_thinned_stats: bool = False,   # NEW: off by default
):
    base_idx = base_df.index[base_mask]
    base_idx = base_idx.intersection(final_df.index)

    if len(base_idx) == 0:
        return {
            'Strategy': strategy,
            'Trees removed(%)': 0.0, 'Volume removed(%)': 0.0,
            'Trees kept': 0, 'Trees removed': 0,
            'Thinned Median DBH': np.nan, 'Thinned Mean DBH': np.nan,     # New fields
            'Q1 cut (count)': 0,
            'Q4 remaining (count)': 0, 'Q4 cut (count)': 0,
            'Removed volume-Q4(%)': 0.0, 'Removed volume-Q4': 0.0,
            'Change-Median DBH': np.nan, 'Change-Mean DBH': np.nan,
            'Volume removed': 0.0, 'Post-thinning total volume': 0.0,
            'Q1 removal ratio': np.nan, 'Q4 retention ratio': np.nan,
            'Pre-thinning Median DBH': np.nan, 'Pre-thinning Mean DBH': np.nan,
            'Post-thinning Median DBH': np.nan, 'Post-thinning Mean DBH': np.nan
            
        }

    pre  = base_df.loc[base_idx]    # baseline cohort (e.g., after 3-row)
    post = final_df.loc[base_idx]   # same cohort after secondary thinning

    keep_mask = post[thin_col].eq(keep_val)
    cut_mask  = post[thin_col].eq(thin_val)

    n_base = len(base_idx)
    n_kept = int(keep_mask.sum())
    n_cut  = int(cut_mask.sum())

    base_vol  = float(pre[vol_col].sum())
    vol_cut   = float(pre.loc[cut_mask, vol_col].sum())
    vol_post  = float(pre.loc[keep_mask, vol_col].sum())

    trees_removed_pct = 100.0 * (n_cut / n_base) if n_base else 0.0
    vol_removed_pct   = 100.0 * (vol_cut / base_vol) if base_vol > 0 else 0.0

    x = pre[metric].astype(float)
    q1_thr = float(x.quantile(0.25))
    q3_thr = float(x.quantile(0.75))
    q1_mask = x <= q1_thr
    q4_mask = x >= q3_thr

    q1_cut = int((q1_mask & cut_mask).sum())
    q4_keep = int((q4_mask & keep_mask).sum())
    q4_cut  = int((q4_mask & cut_mask).sum())

    vol_cut_q4     = float(pre.loc[q4_mask & cut_mask, vol_col].sum())
    vol_cut_q4_pct = 100.0 * (vol_cut_q4 / vol_cut) if vol_cut > 0 else 0.0

    pre_med  = float(x.median()); pre_mean  = float(x.mean())
    x_post   = pre.loc[keep_mask, metric].astype(float)
    post_med = float(x_post.median()) if len(x_post) > 0 else np.nan
    post_mean= float(x_post.mean())   if len(x_post) > 0 else np.nan

    change_med  = post_med - pre_med  if pd.notnull(post_med)  else np.nan
    change_mean = post_mean - pre_mean if pd.notnull(post_mean) else np.nan

    base_q1 = int(q1_mask.sum()); base_q4 = int(q4_mask.sum())
    q1_removal_ratio = (q1_cut / base_q1) if base_q1 > 0 else np.nan
    q4_retention_ratio = (q4_keep / base_q4) if base_q4 > 0 else np.nan

    out = {
        'Strategy': strategy,
        'Trees removed(%)': trees_removed_pct,
        'Volume removed(%)': vol_removed_pct,
        'Trees kept': n_kept,
        'Trees removed': n_cut,
        'Q1 cut (count)': q1_cut,
        'Q4 remaining (count)': q4_keep,
        'Q4 cut (count)': q4_cut,
        'Removed volume-Q4(%)': vol_cut_q4_pct,
        'Removed volume-Q4': vol_cut_q4,
        'Change-Median DBH': change_med,
        'Change-Mean DBH': change_mean,
        'Volume removed': vol_cut,
        'Post-thinning total volume': vol_post,
        'Q1 removal ratio': q1_removal_ratio,
        'Q4 retention ratio': q4_retention_ratio,
        'Pre-thinning Median DBH': pre_med,
        'Pre-thinning Mean DBH': pre_mean,
        'Post-thinning Median DBH': post_med,
        'Post-thinning Mean DBH': post_mean,
    }

    if include_thinned_stats:
        x_cut = pre.loc[cut_mask, metric].astype(float)
        out['Thinned Median DBH'] = float(x_cut.median()) if len(x_cut) > 0 else np.nan
        out['Thinned Mean DBH']   = float(x_cut.mean())   if len(x_cut) > 0 else np.nan
    else:
        out['Thinned Median DBH'] = np.nan
        out['Thinned Mean DBH']   = np.nan

    return out


def table_final_vs_initial(
    df_initial, df_final, *,
    metric='pre_DBH', vol_col='pre_stem_vol',
    status_col='status', thin_col='thin_decision',
    keep_val='Keep', thin_val='Thin',
    strategy='Final vs Initial (Alive baseline)'
):

    base_mask = df_initial[status_col].eq('Alive')
    rep = _stand_metrics_relative(
        df_initial, df_final, base_mask, strategy,
        metric, vol_col, status_col, thin_col, keep_val, thin_val,
        include_thinned_stats=False     # keep this table unchanged
    )
    return pd.DataFrame([rep])


def table_final_vs_after_first(
    df_after_first, df_final, *,
    metric='pre_DBH', vol_col='pre_stem_vol',
    status_col='status', thin_col='thin_decision',
    keep_val='Keep', thin_val='Thin',
    strategy='Final vs After 1st Thinning (Alive&Keep baseline)'
):

    base_mask = df_after_first[status_col].eq('Alive') & df_after_first[thin_col].eq(keep_val)
    rep = _stand_metrics_relative(
        df_after_first, df_final, base_mask, strategy,
        metric, vol_col, status_col, thin_col, keep_val, thin_val,
        include_thinned_stats=True     
    )
    return pd.DataFrame([rep])


# ==============================
# Visualization
# ==============================
def plot_thinning_map(df_thinned: pd.DataFrame, row_col: str = 'Row',
                      x_col: str = 'Tree', status_col: str = 'status',
                      title: str = 'Selected strategy — spatial map'):
    alive  = df_thinned[df_thinned[status_col] == 'Alive']
    kept   = alive[alive['thin_decision'] == 'Keep']
    thinned= alive[alive['thin_decision'] == 'Thin']
    dead   = df_thinned[df_thinned[status_col] == 'Dead']

    fig, ax = plt.subplots(figsize=(8, 8))
    if not dead.empty:
        ax.scatter(dead[x_col], dead[row_col], s=18, c='gray', alpha=0.5,
                   label=f'Dead (n={len(dead)})', edgecolors='none')
    if not kept.empty:
        ax.scatter(kept[x_col], kept[row_col], s=21, c='green',
                   label=f'Kept (n={len(kept)})', edgecolors='k', linewidths=0.25)
    if not thinned.empty:
        ax.scatter(thinned[x_col], thinned[row_col], s=18, c='red', alpha=0.3,
                   label=f'Thinned (n={len(thinned)})')

    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Tree (column)')
    ax.set_ylabel('Row')
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    plt.show()


def center_figure_html(fig: plt.Figure) -> str:

    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"<div style='text-align:center'><img src='data:image/png;base64,{data}'/></div>"

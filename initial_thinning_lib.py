# initial_thinning_lib.py
# ---------------------------------------------------------------------
# Importable utilities for "initial" thinning:
# - k-row thinning (3/4/5-row helpers)
# - variable-row thinning (gap rules, DP optimizer)
# - stand analysis + comparison builders
# - Lexi-APB ranking helpers
# - plotting
# ---------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

__all__ = [
    # core thinning
    "k_row_thinning", "three_row_thinning", "four_row_thinning", "five_row_thinning",
    "variable_row_thinning",
    # variable-gap optimizers / builders
    "choose_variable_cut_rows_q4volume", "choose_variable_cut_rows_q4volume_gap34",
    "variable_thinning_variants_volume_pure",
    # analysis / ranking
    "stand_analysis", "build_comp_for_k", "build_comp_for_variable_variants",
    "adaptive_practical_band", "rank_by_lexi_apb_strict",
    "score_krow_lexi_apb", "score_variable_variants_lexi_apb",
    # viz
    "plot_thinning_map",
    # UI
    "make_krow_dashboard",
]

# =====================================================================
# K-row thinning
# =====================================================================

def k_row_thinning(df: pd.DataFrame, k: int, start_row: int = 1,
                   row_col: str = 'Row', status_col: str = 'status') -> pd.DataFrame:
    """
    Mark every k-th row (aligned so `start_row` is thinned) as 'Thin' for Alive trees.
    Others (Alive) are 'Keep'. Non-Alive set to 'Dead (ignored)'.
    """
    assert k >= 2 and 1 <= start_row <= k
    d = df.copy()
    d['thin_decision'] = 'Dead (ignored)'
    alive = d[status_col].eq('Alive')
    rows_to_thin = ((d[row_col] - start_row) % k == 0)
    d.loc[alive & rows_to_thin, 'thin_decision'] = 'Thin'
    d.loc[alive & ~rows_to_thin, 'thin_decision'] = 'Keep'
    return d

three_row_thinning = lambda df, start_row=1: k_row_thinning(df, 3, start_row)
four_row_thinning  = lambda df, start_row=1: k_row_thinning(df, 4, start_row)
five_row_thinning  = lambda df, start_row=1: k_row_thinning(df, 5, start_row)

# =====================================================================
# Stand analysis
# =====================================================================

def stand_analysis(df_thinned: pd.DataFrame, metric: str = 'pre_DBH',
                   vol_col: str = 'pre_stem_vol', status_col: str = 'status') -> Dict[str, float]:
    alive   = df_thinned[df_thinned[status_col] == 'Alive'].copy()
    kept    = alive[alive['thin_decision'] == 'Keep']
    removed = alive[alive['thin_decision'] == 'Thin']

    # volumes
    pre_total_vol     = float(alive[vol_col].sum())   if len(alive) else 0.0
    post_total_vol    = float(kept[vol_col].sum())    if len(kept)  else 0.0
    removed_total_vol = float(removed[vol_col].sum()) if len(removed) else 0.0

    # central tendencies
    pre_median  = float(alive[metric].median()) if len(alive) else np.nan
    pre_mean    = float(alive[metric].mean())   if len(alive) else np.nan
    post_median = float(kept[metric].median())  if len(kept)  else np.nan
    post_mean   = float(kept[metric].mean())    if len(kept)  else np.nan
    d_median    = (post_median - pre_median) if (not np.isnan(post_median) and not np.isnan(pre_median)) else np.nan
    d_mean      = (post_mean   - pre_mean)   if (not np.isnan(post_mean)   and not np.isnan(pre_mean))   else np.nan

    # quartiles based on pre distribution (Alive)
    q1 = alive[metric].quantile(0.25) if len(alive) else np.nan
    q3 = alive[metric].quantile(0.75) if len(alive) else np.nan

    # counts by quartile
    L_pre  = int((alive [metric] <= q1).sum()) if len(alive) else 0  # Q1
    H_pre  = int((alive [metric] >= q3).sum()) if len(alive) else 0  # Q4
    L_post = int((kept  [metric] <= q1).sum()) if len(kept)  else 0
    H_post = int((kept  [metric] >= q3).sum()) if len(kept)  else 0

    # removed by quartile
    L_cut = L_pre - L_post
    H_cut = H_pre - H_post

    # volume by Q4 removed
    removed_q4_vol = float(removed.loc[removed[metric] >= q3, vol_col].sum()) if len(removed) else 0.0
    pct_removed_vol_from_q4 = (100 * removed_q4_vol / removed_total_vol) if removed_total_vol > 0 else np.nan

    # key ratios
    r_q4 = (H_post / H_pre) if H_pre > 0 else np.nan
    r_q1 = (L_cut / L_pre) if L_pre > 0 else np.nan

    return {
        # overview
        'Trees removed(%)': round((len(removed) / len(alive)) * 100, 2) if len(alive) else np.nan,
        'Volume removed(%)': round((removed_total_vol / pre_total_vol) * 100, 2) if pre_total_vol > 0 else np.nan,
        'Trees kept': int(len(kept)),
        'Trees removed': int(len(removed)),

        # quartile event columns
        'Q1 cut (count)': int(L_cut),
        'Q4 cut (count)': int(H_cut),
        'Q4 remaining (count)': int(H_post),
        'Removed volume-Q4': round(removed_q4_vol, 2),
        'Removed volume-Q4(%)': round(pct_removed_vol_from_q4, 2) if not np.isnan(pct_removed_vol_from_q4) else np.nan,

        # ratios
        'Q1 removal ratio': round(r_q1, 3) if not np.isnan(r_q1) else np.nan,
        'Q4 retention ratio': round(r_q4, 3) if not np.isnan(r_q4) else np.nan,

        'Q1 pre (count)': int(L_pre),
        'Q4 pre (count)': int(H_pre),

        # DBH/volume stats
        'Post-thinning total volume': round(post_total_vol, 2),
        'Volume removed': round(removed_total_vol, 2),
        'Pre-thinning Median DBH': round(pre_median, 2) if not np.isnan(pre_median) else np.nan,
        'Pre-thinning Mean DBH': round(pre_mean, 2)     if not np.isnan(pre_mean)   else np.nan,
        'Post-thinning Median DBH': round(post_median, 2) if not np.isnan(post_median) else np.nan,
        'Post-thinning Mean DBH': round(post_mean, 2)     if not np.isnan(post_mean)   else np.nan,
        'Change-Median DBH': round(d_median, 2) if not np.isnan(d_median) else np.nan,
        'Change-Mean DBH':   round(d_mean, 2)   if not np.isnan(d_mean)   else np.nan,
    }

def build_comp_for_k(df: pd.DataFrame, k: int, metric: str = 'pre_DBH',
                     vol_col: str = 'pre_stem_vol', row_col: str = 'Row',
                     status_col: str = 'status') -> pd.DataFrame:
    rows = []
    for s in range(1, k+1):
        d = k_row_thinning(df, k, s, row_col=row_col, status_col=status_col)
        a = stand_analysis(d, metric=metric, vol_col=vol_col, status_col=status_col)
        a['Strategy']  = f'{k}-row start={s}'
        a['k']         = k
        a['start_row'] = s
        rows.append(a)
    comp = pd.DataFrame(rows)
    for c in ('Q4 retention ratio', 'Q1 removal ratio'):
        if c in comp.columns:
            comp[c] = pd.to_numeric(comp[c], errors='coerce')
    return comp

# =====================================================================
# Variable row thinning (row selection by Q4-volume minimization)
# =====================================================================

def variable_row_thinning(df: pd.DataFrame, cut_rows: List[int],
                          row_col: str = 'Row', status_col: str = 'status') -> pd.DataFrame:
    d = df.copy()
    d['thin_decision'] = 'Dead (ignored)'
    alive = d[status_col].eq('Alive')
    in_cut = d[row_col].isin(cut_rows)
    d.loc[alive & in_cut,  'thin_decision'] = 'Thin'
    d.loc[alive & ~in_cut, 'thin_decision'] = 'Keep'
    return d

def _row_q4_volume_by_row(df: pd.DataFrame, metric: str = 'pre_DBH',
                          vol_col: str = 'pre_stem_vol', row_col: str = 'Row',
                          status_col: str = 'status') -> Tuple[np.ndarray, float, np.ndarray]:
    alive = df[df[status_col] == 'Alive'].copy()
    if alive.empty:
        raise ValueError("No Alive trees found; cannot compute Q4 volumes.")
    q3 = float(alive[metric].quantile(0.75))
    rows = np.sort(pd.unique(df[row_col]))
    q4_rows = alive.loc[alive[metric] >= q3, [row_col, vol_col]]
    q4_vol_by_row = q4_rows.groupby(row_col)[vol_col].sum()
    q4_vol_by_row = q4_vol_by_row.reindex(rows, fill_value=0.0)
    return rows, q3, q4_vol_by_row.values

def _best_sequence_from_start_q4vol(rows: np.ndarray, q4_vols: np.ndarray, start_idx: int,
                                    target_cuts: int, steps: Tuple[int, ...] = (3,4,5)) -> Optional[Tuple[float, Tuple[int,...]]]:
    N = len(rows)
    if target_cuts <= 0 or start_idx < 0 or start_idx >= N:
        return None
    if start_idx + 3*(target_cuts-1) > N-1:
        return None

    @lru_cache(maxsize=None)
    def dp(last_idx: int, selected: int):
        if selected == target_cuts:
            return (0.0, ())
        remaining = target_cuts - selected
        if last_idx + 3*remaining > N-1:
            return None
        best = None
        cand = []
        for st in steps:
            nxt = last_idx + st
            if nxt <= N-1:
                cand.append((q4_vols[nxt], st, nxt))
        cand.sort(key=lambda x: (x[0], x[1], x[2]))
        for q4v, st, nxt in cand:
            rem_after = target_cuts - (selected + 1)
            if nxt + 3*rem_after > N-1:
                continue
            sub = dp(nxt, selected + 1)
            if sub is None:
                continue
            sub_q4, sub_path = sub
            cand_val = (q4v + sub_q4, (nxt,) + sub_path)
            if best is None or (cand_val[0] < best[0]) or (cand_val[0] == best[0] and cand_val[1] < best[1]):
                best = cand_val
        return best

    start_cost = float(q4_vols[start_idx])
    sub = dp(start_idx, 1)
    if sub is None:
        return None
    sub_q4, sub_path = sub
    total_q4 = start_cost + sub_q4
    path = (start_idx,) + sub_path
    return (total_q4, path)

def choose_variable_cut_rows_q4volume(df: pd.DataFrame, target_cuts: int,
                                      metric: str = 'pre_DBH', vol_col: str = 'pre_stem_vol',
                                      row_col: str = 'Row', status_col: str = 'status',
                                      first_start_rows: int = 5,
                                      min_in_between: int = 2, max_in_between: int = 4) -> List[int]:
    assert min_in_between == 2 and max_in_between == 4, "This version fixes steps to {3,4,5}."
    rows, q3, q4_vols = _row_q4_volume_by_row(df, metric=metric, vol_col=vol_col,
                                              row_col=row_col, status_col=status_col)
    N = len(rows)
    if N == 0:
        return []

    max_start_idx = min(first_start_rows, N) - 1
    feasible_starts = [s for s in range(0, max_start_idx + 1) if s + 3*(target_cuts-1) <= N-1]
    if not feasible_starts:
        raise ValueError(f"Infeasible: cannot place {target_cuts} cuts starting within first {first_start_rows} rows.")

    best_total = None
    best_path  = None
    best_start = None
    for s in feasible_starts:
        res = _best_sequence_from_start_q4vol(rows, q4_vols, s, target_cuts, steps=(3,4,5))
        if res is None:
            continue
        tot_q4, path = res
        if (best_total is None) or (tot_q4 < best_total) or (tot_q4 == best_total and s < best_start):
            best_total = tot_q4
            best_path  = path
            best_start = s

    if best_path is None:
        raise ValueError("No feasible sequence found.")
    return rows[list(best_path)].tolist()

def choose_variable_cut_rows_q4volume_gap34(
    df: pd.DataFrame, target_cuts: int, metric: str = 'pre_DBH', vol_col: str = 'pre_stem_vol',
    row_col: str = 'Row', status_col: str = 'status',
    first_start_rows: int = 5,
    min_in_between: int = 3, max_in_between: int = 4
) -> List[int]:
    assert min_in_between == 3 and max_in_between == 4, "This variant fixes steps to {4,5}."
    rows, q3, q4_vols = _row_q4_volume_by_row(
        df, metric=metric, vol_col=vol_col, row_col=row_col, status_col=status_col
    )
    N = len(rows)
    if N == 0:
        return []

    min_step = 4
    max_start_idx = min(first_start_rows, N) - 1
    feasible_starts = [s for s in range(0, max_start_idx + 1)
                       if s + min_step * (target_cuts - 1) <= N - 1]
    if not feasible_starts:
        raise ValueError(
            f"Infeasible: cannot place {target_cuts} cuts with gaps [3,4] starting within first {first_start_rows} rows."
        )

    best_total = None
    best_path  = None
    best_start = None
    for s in feasible_starts:
        res = _best_sequence_from_start_q4vol(rows, q4_vols, s, target_cuts, steps=(4,5))
        if res is None:
            continue
        tot_q4, path = res
        if (best_total is None) or (tot_q4 < best_total) or (tot_q4 == best_total and s < best_start):
            best_total = tot_q4
            best_path  = path
            best_start = s

    if best_path is None:
        raise ValueError("No feasible sequence found.")
    return rows[list(best_path)].tolist()

def variable_thinning_variants_volume_pure(
    df: pd.DataFrame, metric: str = 'pre_DBH', vol_col: str = 'pre_stem_vol',
    row_col: str = 'Row', status_col: str = 'status'
) -> Dict[str, Tuple[List[int], pd.DataFrame]]:
    rows_sorted = np.sort(pd.unique(df[row_col]))
    R = len(rows_sorted)
    targets = {
        '3_row_eqv': R // 3,
        '4_row_eqv': R // 4,
        '5_row_eqv': R // 5,
    }

    out: Dict[str, Tuple[List[int], pd.DataFrame]] = {}
    for label, m in targets.items():
        if label == '5_row_eqv':
            cuts = choose_variable_cut_rows_q4volume_gap34(
                df, m, metric=metric, vol_col=vol_col, row_col=row_col, status_col=status_col,
                first_start_rows=5, min_in_between=3, max_in_between=4
            )
        else:
            cuts = choose_variable_cut_rows_q4volume(
                df, m, metric=metric, vol_col=vol_col, row_col=row_col, status_col=status_col,
                first_start_rows=5, min_in_between=2, max_in_between=4
            )
        d_thin = variable_row_thinning(df, cuts, row_col=row_col, status_col=status_col)
        out[label] = (cuts, d_thin)
    return out

def build_comp_for_variable_variants(variants_dict: Dict[str, Tuple[List[int], pd.DataFrame]],
                                     metric: str = 'pre_DBH', vol_col: str = 'pre_stem_vol',
                                     status_col: str = 'status') -> pd.DataFrame:
    rows = []
    for label, (cut_rows, d_thin) in variants_dict.items():
        a = stand_analysis(d_thin, metric=metric, vol_col=vol_col, status_col=status_col)
        a['Strategy']  = label
        a['k']         = 'variable'
        a['start_row'] = cut_rows[0] if len(cut_rows) else np.nan
        rows.append(a)
    comp = pd.DataFrame(rows)
    for c in ('Q4 retention ratio', 'Q1 removal ratio'):
        if c in comp.columns:
            comp[c] = pd.to_numeric(comp[c], errors='coerce')
    return comp

def compute_best3(df, *, metric='pre_DBH', vol_col='pre_stem_vol',
                  row_col='Row', status_col='status'):
    _, ranked = score_krow_lexi_apb(df, k=3, metric=metric, vol_col=vol_col,
                                    row_col=row_col, status_col=status_col, return_comp=True)
    start = int(ranked.loc[0, 'start_row'])
    df_best3 = k_row_thinning(df, 3, start_row=start, row_col=row_col, status_col=status_col)
    return df_best3, start, ranked


# =====================================================================
# Ranking helpers (Lexi-APB)
# =====================================================================

def adaptive_practical_band(q4_series: pd.Series, lam: float = 0.5,
                            delta_min: float = 0.002, delta_max: float = 0.012) -> float:
    q4 = pd.to_numeric(q4_series, errors='coerce')
    rng = float(q4.max() - q4.min()) if q4.notna().any() else 0.0
    return float(np.clip(lam * rng, delta_min, delta_max))

def rank_by_lexi_apb_strict(comp: pd.DataFrame,
                            p_col: str = 'Q4 retention ratio',
                            s_col: str = 'Q1 removal ratio',
                            lam: float = 0.5, delta_min: float = 0.002, delta_max: float = 0.012,
                            delta_override: float | None = None) -> pd.DataFrame:
    df = comp.copy()
    p = pd.to_numeric(df[p_col], errors='coerce').fillna(-np.inf)
    s = pd.to_numeric(df[s_col], errors='coerce').fillna(-np.inf)

    delta  = float(delta_override) if delta_override is not None else adaptive_practical_band(p, lam, delta_min, delta_max)
    p_best = float(p.max())
    close  = (p_best - p) <= delta
    df['_apb_close'] = close

    df['_apb_s_key'] = np.where(close, s, -np.inf)
    df['_apb_p_key'] = p

    def _norm(x):
        xmin, xmax = float(x.min()), float(x.max())
        return (x - xmin) / (xmax - xmin) if xmax > xmin else pd.Series(0.5, index=x.index)
    df['Lexi-APB index'] = close.astype(float) + np.where(close, _norm(s), _norm(p)) * 1e-3

    df = (df.sort_values(by=['_apb_close', '_apb_s_key', '_apb_p_key'],
                         ascending=[False,        False,       False])
            .drop(columns=['_apb_close','_apb_s_key','_apb_p_key'])
            .reset_index(drop=True))

    df.attrs['apb_delta'] = float(delta)
    return df

def score_krow_lexi_apb(df: pd.DataFrame, k: int, metric: str = 'pre_DBH',
                        vol_col: str = 'pre_stem_vol',
                        row_col: str = 'Row', status_col: str = 'status',
                        lam: float = 0.5, delta_min: float = 0.002, delta_max: float = 0.012,
                        delta_override: float | None = None, title: Optional[str] = None,
                        return_comp: bool = False):
    comp = build_comp_for_k(df, k, metric=metric, vol_col=vol_col, row_col=row_col, status_col=status_col)
    ranked = rank_by_lexi_apb_strict(comp,
                                     p_col='Q4 retention ratio',
                                     s_col='Q1 removal ratio',
                                     lam=lam, delta_min=delta_min, delta_max=delta_max,
                                     delta_override=delta_override)
    apb = ranked.attrs.get('apb_delta', None)
    title = title or f'{k}-row — Lexi-APB (δ={apb:.4f})'

    sty = style_comp_table(ranked, title)
    if return_comp:
        return sty, ranked
    return sty

def score_variable_variants_lexi_apb(df: pd.DataFrame, metric: str = 'pre_DBH',
                                     vol_col: str = 'pre_stem_vol',
                                     row_col: str = 'Row', status_col: str = 'status',
                                     lam: float = 0.5, delta_min: float = 0.002, delta_max: float = 0.012,
                                     delta_override: float | None = None, title: Optional[str] = None,
                                     return_comp: bool = False):
    variants = variable_thinning_variants_volume_pure(df, metric=metric, vol_col=vol_col,
                                                      row_col=row_col, status_col=status_col)
    comp = build_comp_for_variable_variants(variants, metric=metric, vol_col=vol_col, status_col=status_col)
    ranked = rank_by_lexi_apb_strict(comp,
                                     p_col='Q4 retention ratio',
                                     s_col='Q1 removal ratio',
                                     lam=lam, delta_min=delta_min, delta_max=delta_max,
                                     delta_override=delta_override)
    apb = ranked.attrs.get('apb_delta', None)
    title = title or f'Variable thinning — Lexi-APB (δ={apb:.4f})'
    sty = style_comp_table(ranked, title)
    if return_comp:
        return sty, ranked, variants
    return sty


def style_comp_table(comp_ranked: pd.DataFrame, title: Optional[str] = None):
    show = [
        'Strategy',
        'Lexi-APB index',
        'Trees removed(%)','Volume removed(%)',
        'Trees kept','Trees removed',
        'Q1 cut (count)','Q1 removal ratio',
        'Q4 remaining (count)','Q4 cut (count)',
        'Q4 retention ratio',
        'Removed volume-Q4(%)','Removed volume-Q4',
        'Volume removed','Post-thinning total volume',
        'start_row'
    ]
    cols = [c for c in show if c in comp_ranked.columns]
    tbl  = comp_ranked[cols].copy()

    count_cols = ['Trees kept','Trees removed','Q1 cut (count)','Q4 cut (count)','Q4 remaining (count)']
    pct_cols   = ['Trees removed(%)','Volume removed(%)','Removed volume-Q4(%)']
    ratio_cols = ['Q4 retention ratio','Q1 removal ratio']
    money_cols = ['Removed volume-Q4','Volume removed','Post-thinning total volume']

    sty = (tbl.reset_index(drop=True)
           .style
           .format({c:'{:,.0f}' for c in count_cols if c in tbl})
           .format({c:'{:.2f}%' for c in pct_cols   if c in tbl})
           .format({c:'{:.3f}'  for c in ratio_cols if c in tbl})
           .format({c:'{:,.6f}' for c in ['Lexi-APB index'] if c in tbl})
           .format({c:'{:,.2f}' for c in money_cols if c in tbl})
           .set_caption(title or 'Strategies ranked — Lexi-APB')
           .set_properties(**{'font-variant-numeric':'tabular-nums'})
           .hide(axis='index'))
    return sty

def style_comp_table_detailed(comp_ranked: pd.DataFrame, title: Optional[str] = None):
    show = [
        'Strategy',
        'Trees removed(%)','Volume removed(%)',
        'Trees kept','Trees removed',
        'Q1 cut (count)',
        'Q4 remaining (count)','Q4 cut (count)',
        'Removed volume-Q4(%)','Removed volume-Q4',
        'Change-Median DBH','Change-Mean DBH',
        'Volume removed','Post-thinning total volume',
        'Q1 removal ratio',
        'Q4 retention ratio',
        'Pre-thinning Median DBH','Pre-thinning Mean DBH',
        'Post-thinning Median DBH','Post-thinning Mean DBH',
        'Lexi-APB index'
    ]
    cols = [c for c in show if c in comp_ranked.columns]
    tbl  = comp_ranked[cols].copy()

    count_cols = ['Trees kept','Trees removed','Q1 cut (count)','Q4 cut (count)','Q4 remaining (count)']
    pct_cols   = ['Trees removed(%)','Volume removed(%)','Removed volume-Q4(%)']
    ratio_cols = ['Q4 retention ratio','Q1 removal ratio']
    money_cols = ['Removed volume-Q4','Volume removed','Post-thinning total volume']
    dbh_cols   = ['Pre-thinning Mean DBH','Post-thinning Mean DBH',
                  'Pre-thinning Median DBH','Post-thinning Median DBH',
                  'Change-Median DBH','Change-Mean DBH']  # <-- fixed names

    sty = (tbl.reset_index(drop=True)
           .style
           .format({c:'{:,.0f}' for c in count_cols if c in tbl})
           .format({c:'{:.2f}%' for c in pct_cols   if c in tbl})
           .format({c:'{:.3f}'  for c in ratio_cols if c in tbl})
           .format({c:'{:,.6f}' for c in ['Lexi-APB index'] if c in tbl})
           .format({c:'{:,.2f}' for c in money_cols if c in tbl})
           .format({c:'{:,.2f}' for c in dbh_cols   if c in tbl})
           .set_caption(title or 'Strategies ranked — Lexi-APB (detailed)')
           .set_properties(**{'font-variant-numeric':'tabular-nums'})
           .hide(axis='index'))
    return sty

# =====================================================================
# Visualization
# =====================================================================

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

# =====================================================================
# Dashboard
# =====================================================================

def make_krow_dashboard(df: pd.DataFrame, *,
                        metric: str = 'pre_DBH', vol_col: str = 'pre_stem_vol',
                        row_col: str = 'Row', status_col: str = 'status') -> widgets.VBox:
    """
    Built an interactive dashboard to explore 3/4/5-row best strategies and
    their variable-row equivalents.
    """
    ddl = widgets.Dropdown(
        options=[('3-row thinning','3'), ('4-row thinning','4'), ('5-row thinning','5')],
        value='3',
        description='Thinning:',
        layout=widgets.Layout(width='240px')
    )

    table_k_out   = widgets.Output()
    table_var_out = widgets.Output()
    map_best_out  = widgets.Output()
    map_var_out   = widgets.Output()
    note_out      = widgets.Output()

    def _best_krow_result(df_local, k: int):
        comp = build_comp_for_k(df_local, k, metric=metric, vol_col=vol_col,
                                row_col=row_col, status_col=status_col)
        ranked = rank_by_lexi_apb_strict(comp,
                                         p_col='Q4 retention ratio',
                                         s_col='Q1 removal ratio')
        apb = ranked.attrs.get('apb_delta', np.nan)
        sty = style_comp_table_detailed(ranked, title=f'{k}-row — Lexi-APB (δ={apb:.4f})')
        best_start = int(ranked.iloc[0]['start_row'])
        df_best = k_row_thinning(df_local, k, start_row=best_start,
                                 row_col=row_col, status_col=status_col)
        return sty, ranked, df_best

    def _variable_equivalent_for_k(df_local, k: int):
        variants = variable_thinning_variants_volume_pure(df_local, metric=metric,
                                                          vol_col=vol_col, row_col=row_col,
                                                          status_col=status_col)
        key = {3: '3_row_eqv', 4: '4_row_eqv', 5: '5_row_eqv'}[k]
        cut_rows, d_thin = variants[key]
        a = stand_analysis(d_thin, metric=metric, vol_col=vol_col, status_col=status_col)
        a['Strategy']  = f'variable-{key}'
        a['k']         = 'variable'
        a['start_row'] = cut_rows[0] if len(cut_rows) else np.nan
        comp_var = pd.DataFrame([a])
        comp_var['Lexi-APB index'] = 1.0
        sty_var = style_comp_table_detailed(comp_var, title=f'Variable equivalent — {key}')
        return sty_var, comp_var, d_thin

    def render():
        with table_k_out:   clear_output(wait=True)
        with table_var_out: clear_output(wait=True)
        with map_best_out:  clear_output(wait=True)
        with map_var_out:   clear_output(wait=True)
        with note_out:      clear_output(wait=True)

        k = int(ddl.value)
        sty_k, ranked_k, df_best = _best_krow_result(df, k=k)
        with table_k_out:
            display(sty_k)

        sty_var, comp_var, df_var = _variable_equivalent_for_k(df, k=k)
        with table_var_out:
            display(sty_var)

        with note_out:
            best_start = int(ranked_k.iloc[0]['start_row'])
            print(f'Winner ({k}-row): start_row = {best_start}')

        with map_best_out:
            plot_thinning_map(df_best, row_col=row_col, x_col='Tree', status_col=status_col,
                              title=f'Spatial map — Best {k}-row strategy')
        with map_var_out:
            plot_thinning_map(df_var, row_col=row_col, x_col='Tree', status_col=status_col,
                              title=f'Spatial map — Variable equivalent for {k}-row')

    def _on_change(change):
        if change['name'] == 'value' and change['type'] == 'change':
            render()

    ddl.observe(_on_change, names='value')

    header = widgets.HBox([
        widgets.HTML("<b>Select thinning strategy:</b>"),
        ddl
    ], layout=widgets.Layout(align_items='center', gap='12px'))

    tables = widgets.VBox([
        table_k_out,
        widgets.HTML("<hr style='margin:8px 0;'>"),
        table_var_out
    ])

    maps = widgets.VBox([
        note_out,
        map_best_out,
        map_var_out
    ])

    ui = widgets.VBox([
        header,
        widgets.HTML("<hr>"),
        tables,
        widgets.HTML("<hr>"),
        maps
    ], layout=widgets.Layout(width='100%'))

    # initial render
    render()
    return ui

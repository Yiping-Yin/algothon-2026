"""Standalone source-faithful Simple2 submission for Algothon 2026.

Economic source lineage:
    strategy_final/simple2_pruned_single_engine.py
    source commit 901a01f57cfb8b4bf793f0e61f401904bc3ed48a
    source blob   4fb63fd0c8c6c2e5308d197397afca8ae23cbc55

This file is self-contained.  It reads only the visible ``prcSoFar`` prefix and
exposes the required global ``getMyPosition`` function.  The implementation was
assembled from the source-faithful reproduction used in the V3/V4 audits and
validated against its complete public-data decision path.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

# These defaults are harmless when BLAS has already been initialised by the
# evaluator, and reduce oversubscription when this module is imported first.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

# Small matrices are much faster without BLAS oversubscription.  scikit-learn
# is part of the official grading image and exposes a backend-neutral controller
# for already-loaded NumPy/SciPy thread pools.
try:
    from sklearn.utils.parallel import ThreadpoolController

    _THREADPOOL_CONTROLLER = ThreadpoolController()
    _THREADPOOL_LIMIT = _THREADPOOL_CONTROLLER.limit(limits=1)
except Exception:
    _THREADPOOL_CONTROLLER = None
    _THREADPOOL_LIMIT = None

# =============================== parameters ==================================
EPS = 1e-8
MIN_OBS = 70
DEFAULT_DLR_LIMIT = 10_000.0
ALGO_DLR_LIMIT = 100_000.0

BETA_CLIP = (0.25, 2.50)
BETA_SHORT_WINDOW = 250
RIDGE = 0.75
MARKET_LEADER_WEIGHT = 2.50
LEADLAG_MIN_OBS = 30
GROUP_LASSO_ALPHA = 0.25
GROUP_LASSO_IDENTIFICATION_RIDGE = 0.01
GROUP_LASSO_MAX_ITER = 300
GROUP_LASSO_TOLERANCE = 1e-6
GROUP_LASSO_ACTIVE_THRESHOLD = 1e-6
REV_WINS = (15, 20, 25)
REV_WEIGHT = 0.20
REV_NEUTRAL_PC = 2
REV_NEUTRAL_WINDOW = 250

LOWRANK_RIDGE = 1.60
LOWRANK_RANKS = (3, 4, 5, 6)
PAIRNET_TOP_K = 20
PAIRNET_ADF_MAX = -3.20
PAIRNET_Z_WINDOW = 250
PAIRNET_REFRESH = 50

PAIR_COUNT = 12
PAIR_MIN_FORMATION_OBS = 250
PAIR_MIN_CORRELATION = 0.15
PAIR_MAX_AR1 = 0.98
PAIR_MIN_CROSSINGS = 8
PAIR_GATE_Z = 0.40
PAIR_EXIT_Z = 0.35
PAIR_REFRESH_DAYS = 60
PAIR_RECENT_SELECTION_WINDOW = 500
PAIR_MID_SHORT_WINDOW = 360
PAIR_MID_LONG_WINDOW = 420
PAIR_EXTRA_START = 500
PAIR_ACTIVE_CAP = 10
PAIR_CONVICTION_RATIO = 0.50
PAIR_BETA_LONG_WINDOW = 500
PAIR_BETA_SHORT_WINDOW = 250
PAIR_BETA_DRIFT_MAX = 0.40
DAY6_PROTECT_TOP_K = 2
PAIR_EARLY_OVERRIDE_CAP = 12
PAIR_FULL_OVERRIDE_CAP = 11
PAIR_Z_WINDOWS = (100, 120, 150)

LEVEL_HORIZONS = (250, 350, 500, 650)
LEVEL_TAIL_Z = 2.0
BLEND_PRIOR_WEIGHTS = np.asarray((1.0, 0.0, 1.0, 1.0, 0.25, 0.25, 0.25))
BLEND_WINDOW = 125
BLEND_L2 = 0.001
BLEND_MAX_ITER = 50
ALGO_TILT_DLR = 100_000.0
ALGO_BREADTH_GATE = 0.04
ALGO_IDLE_MIN_NET_VOTES = 6
ALGO_IDLE_SEASONAL_LADDERS = ((22, 44, 88), (23, 46, 92))
ALGO_SEASONAL_MIN_OBS = 92

def cs_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(centered)
    return np.clip(centered / scale, -4.0, 4.0)

def position_limits(nins: int) -> np.ndarray:
    limits = np.full(nins, DEFAULT_DLR_LIMIT)
    limits[0] = ALGO_DLR_LIMIT
    return limits

def ols_beta_and_se(
    stocks: np.ndarray, market: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    market_centered = market - np.mean(market)
    stocks_centered = stocks - np.mean(stocks, axis=1, keepdims=True)
    denominator = float(market_centered @ market_centered) + EPS
    beta = stocks_centered @ market_centered / denominator
    errors = stocks_centered - beta[:, None] * market_centered[None, :]
    degrees = max(len(market) - 2, 1)
    variance = np.sum(errors * errors, axis=1) / degrees
    return beta, np.sqrt(variance / denominator)

def residual_reversal(resid: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.mean(resid * resid, axis=1) + EPS * EPS)
    reversal = np.zeros(resid.shape[0])
    for window in REV_WINS:
        width = min(window, resid.shape[1])
        reversal += -np.sum(resid[:, -width:], axis=1) / (
            vol * np.sqrt(width) + EPS
        )
    reversal /= len(REV_WINS)
    if 0 < REV_NEUTRAL_PC < resid.shape[0]:
        try:
            width = min(REV_NEUTRAL_WINDOW, resid.shape[1])
            correlation = np.nan_to_num(
                np.corrcoef(resid[:, -width:]), nan=0.0
            )
            np.fill_diagonal(correlation, 1.0)
            loadings = np.linalg.eigh(correlation)[1][:, -REV_NEUTRAL_PC:]
            reversal = reversal - loadings @ (loadings.T @ reversal)
        except np.linalg.LinAlgError:
            pass
    return reversal

def group_soft_threshold(
    values: np.ndarray, threshold: float, axis: int
) -> np.ndarray:
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    scale = np.maximum(0.0, 1.0 - threshold / np.maximum(norms, 1e-15))
    return values * scale

def post_selection_ridge(
    xtx: np.ndarray, xty: np.ndarray, support: np.ndarray
) -> np.ndarray:
    coefficients = np.zeros(xty.shape, dtype=float)
    for follower in range(xty.shape[1]):
        keep = np.concatenate((np.asarray((True,)), support[:, follower]))
        retained = int(np.sum(keep))
        coefficients[keep, follower] = np.linalg.solve(
            xtx[np.ix_(keep, keep)] + RIDGE * np.eye(retained),
            xty[keep, follower],
        )
    return coefficients

def lowrank_lag1_signal(
    stocks: np.ndarray, market: np.ndarray, resid: np.ndarray
) -> np.ndarray:
    if stocks.shape[1] < 2:
        return np.zeros(stocks.shape[0])
    predictors = np.vstack((stocks, market[None, :]))
    predictors = (
        predictors - np.mean(predictors, axis=1, keepdims=True)
    ) / (np.std(predictors, axis=1, keepdims=True) + EPS)
    targets = (resid - np.mean(resid, axis=1, keepdims=True)) / (
        np.std(resid, axis=1, keepdims=True) + EPS
    )
    x = predictors[:, :-1].T
    y = targets[:, 1:].T
    gram = x.T @ x / len(x)
    cross = x.T @ y / len(x)
    try:
        coefficients = np.linalg.solve(
            gram + LOWRANK_RIDGE * np.eye(gram.shape[0]), cross
        )
        left, spectrum, right = np.linalg.svd(coefficients, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(stocks.shape[0])
    latest = predictors[:, -1]
    accumulated = np.zeros(stocks.shape[0])
    for rank in LOWRANK_RANKS:
        accumulated += cs_z(
            latest @ ((left[:, :rank] * spectrum[:rank]) @ right[:rank])
        )
    return cs_z(accumulated)

def ar1(spread: np.ndarray) -> float:
    left = spread[:-1] - np.mean(spread[:-1])
    right = spread[1:] - np.mean(spread[1:])
    return float(left @ right / (left @ left + EPS))

def adf_t_lag1(spread: np.ndarray) -> float:
    delta = np.diff(spread)
    dependent = delta[1:]
    design = np.column_stack((np.ones(len(dependent)), spread[1:-1], delta[:-1]))
    if len(dependent) <= design.shape[1]:
        return np.nan
    try:
        coefficient = np.linalg.lstsq(design, dependent, rcond=None)[0]
        error = dependent - design @ coefficient
        variance = float(error @ error) / (len(dependent) - design.shape[1])
        covariance = variance * np.linalg.inv(design.T @ design)
        standard_error = float(np.sqrt(covariance[1, 1]))
    except np.linalg.LinAlgError:
        return np.nan
    if not np.isfinite(standard_error) or standard_error <= 0.0:
        return np.nan
    return float(coefficient[1] / standard_error)

def fit_pair_models(
    price_history: np.ndarray, selection_window: int | None = None
) -> tuple[tuple[float, int, int, float, float, float, float], ...]:
    log_prices = np.log(np.asarray(price_history, dtype=float).T)
    if selection_window is not None:
        log_prices = log_prices[-min(int(selection_window), len(log_prices)) :]
    returns = np.diff(log_prices, axis=0)
    models: list[tuple[float, int, int, float, float, float, float]] = []
    for left_index in range(1, log_prices.shape[1]):
        for right_index in range(left_index + 1, log_prices.shape[1]):
            left = log_prices[:, left_index]
            right = log_prices[:, right_index]
            intercept = float(np.mean(left - right))
            spread = left - intercept - right
            spread_scale = float(np.std(spread))
            if not np.isfinite(spread_scale) or spread_scale <= EPS:
                continue
            phi = ar1(spread)
            correlation = float(
                np.corrcoef(returns[:, left_index], returns[:, right_index])[0, 1]
            )
            crossings = int(
                np.count_nonzero(np.diff(np.sign(spread - np.mean(spread))))
            )
            if (
                not np.isfinite(phi)
                or not 0.0 < phi < PAIR_MAX_AR1
                or not np.isfinite(correlation)
                or correlation <= PAIR_MIN_CORRELATION
                or crossings < PAIR_MIN_CROSSINGS
            ):
                continue
            adf_t = adf_t_lag1(spread)
            if not np.isfinite(adf_t):
                continue
            models.append(
                (
                    -adf_t,
                    left_index,
                    right_index,
                    intercept,
                    1.0,
                    float(np.mean(spread)),
                    spread_scale,
                )
            )
    models.sort(key=lambda model: (-model[0], model[1], model[2]))
    selected = []
    used_instruments: set[int] = set()
    for model in models:
        left_index, right_index = model[1], model[2]
        if left_index in used_instruments or right_index in used_instruments:
            continue
        selected.append(model)
        used_instruments.update((left_index, right_index))
        if len(selected) >= PAIR_COUNT:
            break
    return tuple(selected)

def beta_structure_healthy(
    log_prices: np.ndarray, left: int, right: int
) -> bool:
    long_width = min(PAIR_BETA_LONG_WINDOW, log_prices.shape[1])
    short_width = min(PAIR_BETA_SHORT_WINDOW, log_prices.shape[1])
    x_long = log_prices[right, -long_width:]
    y_long = log_prices[left, -long_width:]
    x_short = log_prices[right, -short_width:]
    y_short = log_prices[left, -short_width:]
    cx_long = x_long - np.mean(x_long)
    cx_short = x_short - np.mean(x_short)
    beta_long = float(
        cx_long @ (y_long - np.mean(y_long)) / (cx_long @ cx_long + EPS)
    )
    beta_short = float(
        cx_short @ (y_short - np.mean(y_short)) / (cx_short @ cx_short + EPS)
    )
    return bool(
        np.isfinite(beta_long)
        and np.isfinite(beta_short)
        and abs(beta_short - beta_long) <= PAIR_BETA_DRIFT_MAX
    )

def book_target_sign(
    log_prices: np.ndarray,
    models: tuple[tuple[float, int, int, float, float, float, float], ...],
    states: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    model_keys = {(model[1], model[2]) for model in models}
    for key in tuple(states):
        if key not in model_keys:
            del states[key]

    active: list[tuple[float, int, int, int]] = []
    for _, left, right, _intercept, _beta, _mean, _scale in models:
        key = (left, right)
        if not beta_structure_healthy(log_prices, left, right):
            states.pop(key, None)
            continue
        z_values = []
        for window in PAIR_Z_WINDOWS:
            width = min(window, log_prices.shape[1])
            spread = log_prices[left, -width:] - log_prices[right, -width:]
            rolling_scale = float(np.std(spread))
            if not np.isfinite(rolling_scale) or rolling_scale <= EPS:
                z_values = []
                break
            z_values.append(
                float((spread[-1] - np.mean(spread)) / (rolling_scale + EPS))
            )
        if not z_values:
            states.pop(key, None)
            continue
        z_array = np.asarray(z_values)
        unanimous = bool(np.all(z_array > 0.0) or np.all(z_array < 0.0))
        z_score = float(np.median(z_array))
        desired_side = -int(np.sign(z_score))
        side = int(states.get(key, 0))
        if side == 0:
            if unanimous and abs(z_score) >= PAIR_GATE_Z:
                side = desired_side
        else:
            if not unanimous or abs(z_score) <= PAIR_EXIT_Z:
                side = 0
            elif desired_side != side:
                side = desired_side if abs(z_score) >= PAIR_GATE_Z else 0
        if side == 0:
            states.pop(key, None)
            continue
        states[key] = side
        active.append((abs(z_score), left, right, side))

    active.sort(key=lambda item: (-item[0], item[1], item[2]))
    active = active[:PAIR_ACTIVE_CAP]
    target_dollars = np.zeros(log_prices.shape[0])
    confidence = np.zeros(log_prices.shape[0])
    for abs_z, left, right, side in active:
        target_dollars[left] += DEFAULT_DLR_LIMIT * side
        target_dollars[right] -= DEFAULT_DLR_LIMIT * side
        confidence[left] = max(confidence[left], abs_z)
        confidence[right] = max(confidence[right], abs_z)
    direction = np.sign(
        np.clip(target_dollars, -DEFAULT_DLR_LIMIT, DEFAULT_DLR_LIMIT)
    )
    return direction, confidence

class CompressionEngine:
    def _fit_group_lasso_support(self, residuals: np.ndarray) -> np.ndarray:
        nstocks = residuals.shape[0]
        full_support = np.ones((nstocks, nstocks), dtype=bool)
        fallback = (
            self.lasso_support.copy()
            if self.lasso_support is not None
            and self.lasso_support.shape == full_support.shape
            else full_support
        )
        if residuals.shape[1] < 2 or not np.isfinite(residuals).all():
            return fallback
        x_raw = residuals[:, :-1].T
        y_raw = residuals[:, 1:].T
        x = (x_raw - np.mean(x_raw, axis=0)) / (
            np.std(x_raw, axis=0) + EPS
        )
        y = (y_raw - np.mean(y_raw, axis=0)) / (
            np.std(y_raw, axis=0) + EPS
        )
        xtx = x.T @ x / len(x)
        xty = x.T @ y / len(x)
        if (
            self.lasso_row is None
            or self.lasso_column is None
            or self.lasso_row.shape != xty.shape
            or self.lasso_column.shape != xty.shape
        ):
            row = np.zeros_like(xty)
            column = np.zeros_like(xty)
        else:
            row = np.asarray(self.lasso_row, dtype=float).copy()
            column = np.asarray(self.lasso_column, dtype=float).copy()
        np.fill_diagonal(row, 0.0)
        np.fill_diagonal(column, 0.0)
        accelerated_row = row.copy()
        accelerated_column = column.copy()
        momentum = 1.0
        try:
            base_lipschitz = float(np.linalg.eigvalsh(xtx)[-1] + RIDGE)
        except np.linalg.LinAlgError:
            return fallback
        step = 1.0 / max(
            2.0 * base_lipschitz + GROUP_LASSO_IDENTIFICATION_RIDGE,
            1e-12,
        )
        for _ in range(GROUP_LASSO_MAX_ITER):
            total = accelerated_row + accelerated_column
            common_gradient = xtx @ total - xty + RIDGE * total
            row_proposal = accelerated_row - step * (
                common_gradient
                + GROUP_LASSO_IDENTIFICATION_RIDGE * accelerated_row
            )
            column_proposal = accelerated_column - step * (
                common_gradient
                + GROUP_LASSO_IDENTIFICATION_RIDGE * accelerated_column
            )
            np.fill_diagonal(row_proposal, 0.0)
            np.fill_diagonal(column_proposal, 0.0)
            updated_row = group_soft_threshold(
                row_proposal, step * GROUP_LASSO_ALPHA, axis=1
            )
            updated_column = group_soft_threshold(
                column_proposal, step * GROUP_LASSO_ALPHA, axis=0
            )
            np.fill_diagonal(updated_row, 0.0)
            np.fill_diagonal(updated_column, 0.0)
            change = math.sqrt(
                np.linalg.norm(updated_row - row) ** 2
                + np.linalg.norm(updated_column - column) ** 2
            ) / max(
                1.0,
                math.sqrt(
                    np.linalg.norm(row) ** 2 + np.linalg.norm(column) ** 2
                ),
            )
            new_momentum = 0.5 * (
                1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)
            )
            acceleration = (momentum - 1.0) / new_momentum
            accelerated_row = updated_row + acceleration * (updated_row - row)
            accelerated_column = updated_column + acceleration * (
                updated_column - column
            )
            row = updated_row
            column = updated_column
            momentum = new_momentum
            if change < GROUP_LASSO_TOLERANCE:
                break
        if not np.isfinite(row).all() or not np.isfinite(column).all():
            return fallback
        self.lasso_row = row
        self.lasso_column = column
        active_rows = (
            np.linalg.norm(row, axis=1) > GROUP_LASSO_ACTIVE_THRESHOLD
        )
        active_columns = (
            np.linalg.norm(column, axis=0) > GROUP_LASSO_ACTIVE_THRESHOLD
        )
        support = active_rows[:, None] | active_columns[None, :]
        self.lasso_support = support.copy()
        return support


base = SimpleNamespace(
    BETA_CLIP=BETA_CLIP,
    BETA_SHORT_WINDOW=BETA_SHORT_WINDOW,
    DAY6_PROTECT_TOP_K=DAY6_PROTECT_TOP_K,
    LEADLAG_MIN_OBS=LEADLAG_MIN_OBS,
    MARKET_LEADER_WEIGHT=MARKET_LEADER_WEIGHT,
    PAIRNET_ADF_MAX=PAIRNET_ADF_MAX,
    PAIRNET_REFRESH=PAIRNET_REFRESH,
    PAIRNET_TOP_K=PAIRNET_TOP_K,
    PAIRNET_Z_WINDOW=PAIRNET_Z_WINDOW,
    PAIR_CONVICTION_RATIO=PAIR_CONVICTION_RATIO,
    PAIR_EARLY_OVERRIDE_CAP=PAIR_EARLY_OVERRIDE_CAP,
    PAIR_EXTRA_START=PAIR_EXTRA_START,
    PAIR_FULL_OVERRIDE_CAP=PAIR_FULL_OVERRIDE_CAP,
    PAIR_MID_LONG_WINDOW=PAIR_MID_LONG_WINDOW,
    PAIR_MID_SHORT_WINDOW=PAIR_MID_SHORT_WINDOW,
    PAIR_MIN_FORMATION_OBS=PAIR_MIN_FORMATION_OBS,
    PAIR_RECENT_SELECTION_WINDOW=PAIR_RECENT_SELECTION_WINDOW,
    PAIR_REFRESH_DAYS=PAIR_REFRESH_DAYS,
    REV_WEIGHT=REV_WEIGHT,
    RIDGE=RIDGE,
    book_target_sign=book_target_sign,
    cs_z=cs_z,
    fit_pair_models=fit_pair_models,
    lowrank_lag1_signal=lowrank_lag1_signal,
    ols_beta_and_se=ols_beta_and_se,
    position_limits=position_limits,
    post_selection_ridge=post_selection_ridge,
    residual_reversal=residual_reversal,
    CompressionEngine=CompressionEngine,
)

def signal_beta_exact(stocks: np.ndarray, market: np.ndarray) -> np.ndarray:
    long_beta = np.clip(
        base.ols_beta_and_se(stocks, market)[0], base.BETA_CLIP[0], base.BETA_CLIP[1]
    )
    if len(market) < 2 * base.BETA_SHORT_WINDOW:
        return long_beta
    recent_beta, recent_se = base.ols_beta_and_se(
        stocks[:, -base.BETA_SHORT_WINDOW :], market[-base.BETA_SHORT_WINDOW :]
    )
    prior_beta, prior_se = base.ols_beta_and_se(
        stocks[:, -2 * base.BETA_SHORT_WINDOW : -base.BETA_SHORT_WINDOW],
        market[-2 * base.BETA_SHORT_WINDOW : -base.BETA_SHORT_WINDOW],
    )
    difference = recent_beta - prior_beta
    noise_variance = recent_se * recent_se + prior_se * prior_se
    drift_weight = np.maximum(
        0.0, 1.0 - noise_variance / (difference * difference + EPS)
    )
    return np.clip(
        long_beta + drift_weight * (recent_beta - long_beta),
        base.BETA_CLIP[0], base.BETA_CLIP[1]
    )

def residuals_exact(stocks: np.ndarray, market: np.ndarray) -> np.ndarray:
    resid = stocks - signal_beta_exact(stocks, market)[:, None] * market[None, :]
    return resid - np.mean(resid, axis=0, keepdims=True)

def level_reversion_signal(resid: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(resid * resid, axis=1)) + EPS
    total = np.zeros(resid.shape[0])
    for horizon in LEVEL_HORIZONS:
        width = min(horizon, resid.shape[1])
        total += base.cs_z(
            -np.sum(resid[:, -width:], axis=1) / (scale * np.sqrt(width))
        )
    return base.cs_z(total)

def adf_t_every_pair_exact(log_prices: np.ndarray) -> np.ndarray:
    """Robust ADF(1) implementation copied from the frozen Simple2 source."""
    left, right = np.triu_indices(log_prices.shape[1], 1)
    spread = log_prices[:, left] - log_prices[:, right]
    statistics = np.full(spread.shape[1], np.nan)
    change = np.diff(spread, axis=0)
    dependent = change[1:]
    level = spread[1:-1]
    lagged = change[:-1]
    rows = len(dependent)
    if rows <= 4:
        return statistics
    design = np.empty((spread.shape[1], 3, 3))
    design[:, 0, 0] = rows
    design[:, 0, 1] = design[:, 1, 0] = level.sum(0)
    design[:, 0, 2] = design[:, 2, 0] = lagged.sum(0)
    design[:, 1, 1] = (level * level).sum(0)
    design[:, 1, 2] = design[:, 2, 1] = (level * lagged).sum(0)
    design[:, 2, 2] = (lagged * lagged).sum(0)
    moment = np.stack(
        (dependent.sum(0), (level * dependent).sum(0), (lagged * dependent).sum(0)),
        axis=1,
    )
    finite = np.isfinite(design).all(axis=(1, 2)) & np.isfinite(moment).all(axis=1)
    screened = np.flatnonzero(finite)
    if not len(screened):
        return statistics
    ranks = np.zeros(spread.shape[1], dtype=int)
    cond = np.full(spread.shape[1], np.inf)
    try:
        ranks[screened] = np.linalg.matrix_rank(design[screened])
        cond[screened] = np.linalg.cond(design[screened])
    except np.linalg.LinAlgError:
        for idx in screened:
            try:
                ranks[idx] = np.linalg.matrix_rank(design[idx])
                cond[idx] = np.linalg.cond(design[idx])
            except np.linalg.LinAlgError:
                pass
    valid = np.flatnonzero((ranks == 3) & np.isfinite(cond) & (cond <= 1e12))
    if not len(valid):
        return statistics
    try:
        coef = np.linalg.solve(design[valid], moment[valid, ..., None])[..., 0]
        errors = dependent[:, valid] - (
            coef[:, 0][None, :]
            + level[:, valid] * coef[:, 1][None, :]
            + lagged[:, valid] * coef[:, 2][None, :]
        )
        variance = (errors * errors).sum(0) / (rows - 3)
        inv = np.linalg.inv(design[valid])
        se = np.sqrt(np.maximum(variance * inv[:, 1, 1], 1e-30))
        statistics[valid] = coef[:, 1] / se
    except np.linalg.LinAlgError:
        for idx in valid:
            try:
                coef = np.linalg.solve(design[idx], moment[idx])
                error = dependent[:, idx] - (
                    coef[0] + level[:, idx] * coef[1] + lagged[:, idx] * coef[2]
                )
                variance = float(error @ error) / (rows - 3)
                inv = np.linalg.inv(design[idx])
                statistics[idx] = coef[1] / math.sqrt(
                    max(variance * inv[1, 1], 1e-30)
                )
            except np.linalg.LinAlgError:
                pass
    return statistics

def breadth_direction(side: np.ndarray) -> float:
    breadth = float(np.mean(side))
    return float(np.sign(breadth)) if abs(breadth) >= ALGO_BREADTH_GATE else 0.0

def algo_tilt_position(core_side: np.ndarray, prior_side: np.ndarray,
                       price: float, share_limit: int) -> int:
    core_direction = breadth_direction(core_side)
    prior_direction = breadth_direction(prior_side)
    tilt = core_direction if prior_direction == 0.0 or prior_direction == core_direction else 0.0
    dollars = float(np.clip(ALGO_TILT_DLR * tilt, -ALGO_DLR_LIMIT, ALGO_DLR_LIMIT))
    return int(np.clip(int(dollars / price), -share_limit, share_limit))

def idle_algo_position(side: np.ndarray, market: np.ndarray, share_limit: int) -> int:
    net_votes = int(np.sum(side))
    if abs(net_votes) >= ALGO_IDLE_MIN_NET_VOTES:
        return int(np.sign(net_votes) * share_limit)
    if len(market) < ALGO_SEASONAL_MIN_OBS:
        return 0
    ladder_sides = [
        int(np.sign(sum(int(np.sign(market[-lag])) for lag in ladder)))
        for ladder in ALGO_IDLE_SEASONAL_LADDERS
    ]
    if ladder_sides[0] != 0 and all(value == ladder_sides[0] for value in ladder_sides):
        return int(ladder_sides[0] * share_limit)
    return 0

@dataclass
class S2AuditStep:
    positions: dict[str, np.ndarray]
    sides: dict[str, np.ndarray]
    exact_algo: int
    weights: np.ndarray
    core_side: np.ndarray
    prior_side: np.ndarray
    overlay_nonzero: int
    pairnet_fit_nt: int
    hardpair_fit_nt: int
    support_fraction: float

class Simple2SourceReproduction:
    def __init__(self) -> None:
        self.blend_ready = False
        self.replaying = False
        self.replay_rows: list[tuple[int, np.ndarray]] = []
        self.features: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.weights = BLEND_PRIOR_WEIGHTS.copy()
        self.pending_features: np.ndarray | None = None
        self.pending_nt = 0
        self.live_history: np.ndarray | None = None
        self.last_result: S2AuditStep | None = None
        self._reset_core(51)

    def _reset_core(self, nins: int | None = None) -> None:
        if nins is not None:
            self.nins = nins
        self.hard_models: list[tuple[Any, ...]] = [(), (), (), ()]
        self.hard_states: list[dict[tuple[int, int], int]] = [{}, {}, {}, {}]
        self.hard_fit_nt = 0
        self.lasso_row: np.ndarray | None = None
        self.lasso_column: np.ndarray | None = None
        self.lasso_support: np.ndarray | None = None
        self.pairnet_fit_nt: int | None = None
        self.pairnet_pairs: np.ndarray | None = None
        self.pairnet_weights: np.ndarray | None = None
        self.last_nt = 0
        self.last_history: np.ndarray | None = None
        self.last_output: np.ndarray | None = None
        self.last_result = None

    def _remember(self, history: np.ndarray, positions: np.ndarray) -> np.ndarray:
        self.last_history = history.copy()
        self.last_output = np.asarray(positions, dtype=int).copy()
        return self.last_output.copy()

    def _extends_last(self, history: np.ndarray) -> bool:
        last = self.last_history
        return last is None or bool(
            history.shape[0] == last.shape[0]
            and history.shape[1] > last.shape[1]
            and np.array_equal(history[:, : last.shape[1]], last)
        )

    def _extends_live(self, history: np.ndarray) -> bool:
        live = self.live_history
        if live is None:
            return False
        if history.shape == live.shape:
            return bool(np.array_equal(history, live))
        return bool(
            history.shape[0] == live.shape[0]
            and history.shape[1] == live.shape[1] + 1
            and np.array_equal(history[:, : live.shape[1]], live)
        )

    def _fit_group_lasso_support(self, residuals: np.ndarray) -> np.ndarray:
        return base.CompressionEngine._fit_group_lasso_support(self, residuals)

    def _refresh_pair_books(self, history: np.ndarray, nt: int) -> None:
        if nt < base.PAIR_MIN_FORMATION_OBS:
            return
        due = not self.hard_models[0] or nt - self.hard_fit_nt >= base.PAIR_REFRESH_DAYS
        windows = (None, base.PAIR_RECENT_SELECTION_WINDOW,
                   base.PAIR_MID_SHORT_WINDOW, base.PAIR_MID_LONG_WINDOW)
        minimums = (base.PAIR_MIN_FORMATION_OBS, base.PAIR_MIN_FORMATION_OBS,
                    base.PAIR_EXTRA_START, base.PAIR_EXTRA_START)
        for idx, (window, minimum) in enumerate(zip(windows, minimums)):
            if nt >= minimum and (due or not self.hard_models[idx]):
                self.hard_models[idx] = base.fit_pair_models(history, window)
        if due:
            self.hard_fit_nt = nt

    def _apply_pair_overlay(self, history: np.ndarray, prices: np.ndarray,
                            book: np.ndarray, share_limits: np.ndarray,
                            gate_signal: np.ndarray) -> np.ndarray:
        if not any(self.hard_models):
            return book
        log_prices = np.log(history)
        directions, confidences = [], []
        for models, states in zip(self.hard_models, self.hard_states):
            if models:
                direction, confidence = base.book_target_sign(log_prices, models, states)
            else:
                direction = np.zeros(log_prices.shape[0])
                confidence = np.zeros(log_prices.shape[0])
            directions.append(direction)
            confidences.append(confidence)
        combined_sign = np.sign(np.sum(np.asarray(directions), axis=0))
        pair_confidence = np.zeros(log_prices.shape[0])
        for idx in np.flatnonzero(combined_sign):
            supporting = [
                confidence[idx]
                for direction, confidence in zip(directions, confidences)
                if direction[idx] == combined_sign[idx] and confidence[idx] > 0
            ]
            if supporting:
                pair_confidence[idx] = min(supporting)
        pair_pos = np.clip(
            (combined_sign * DEFAULT_DLR_LIMIT / prices).astype(int),
            -share_limits, share_limits
        )
        protected = set(
            (1 + np.argsort(-np.abs(gate_signal))[: base.DAY6_PROTECT_TOP_K]).tolist()
        )
        use_gate = history.shape[1] >= base.PAIR_EXTRA_START
        candidates: list[tuple[float, int]] = []
        for idx in np.flatnonzero(combined_sign):
            if idx in protected:
                continue
            required = base.PAIR_CONVICTION_RATIO * abs(gate_signal[idx - 1]) if use_gate else 0.0
            if pair_confidence[idx] < required:
                continue
            if np.sign(pair_pos[idx]) != np.sign(book[idx]):
                candidates.append((float(pair_confidence[idx] - required), int(idx)))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        active_books = sum(bool(models) for models in self.hard_models)
        cap = base.PAIR_EARLY_OVERRIDE_CAP if active_books <= 2 else base.PAIR_FULL_OVERRIDE_CAP
        result = book.copy()
        for _, idx in candidates[:cap]:
            result[idx] = pair_pos[idx]
        return result

    def _pairnet_signal(self, log_prices: np.ndarray, nt: int) -> np.ndarray:
        nstocks = log_prices.shape[1]
        left, right = np.triu_indices(nstocks, 1)
        if self.pairnet_pairs is None or self.pairnet_fit_nt is None or nt - self.pairnet_fit_nt >= base.PAIRNET_REFRESH:
            statistics = adf_t_every_pair_exact(log_prices)
            eligible = np.flatnonzero(np.isfinite(statistics) & (statistics < base.PAIRNET_ADF_MAX))
            selected: list[int] = []
            claimed: set[int] = set()
            for candidate in eligible[np.argsort(statistics[eligible])]:
                first, second = int(left[candidate]), int(right[candidate])
                if first in claimed or second in claimed:
                    continue
                selected.append(int(candidate))
                claimed.update((first, second))
                if len(selected) >= base.PAIRNET_TOP_K:
                    break
            self.pairnet_pairs = np.asarray(selected, dtype=int)
            self.pairnet_weights = np.abs(statistics[self.pairnet_pairs])
            self.pairnet_fit_nt = nt
        if self.pairnet_pairs is None or not len(self.pairnet_pairs):
            return np.zeros(nstocks)
        window = log_prices[-min(base.PAIRNET_Z_WINDOW, len(log_prices)) :]
        pairs = self.pairnet_pairs
        spread = window[:, left[pairs]] - window[:, right[pairs]]
        z_score = (spread[-1] - np.mean(spread, axis=0)) / (np.std(spread, axis=0) + EPS)
        pressure = np.zeros(nstocks)
        np.add.at(pressure, left[pairs], -z_score * self.pairnet_weights)
        np.add.at(pressure, right[pairs], z_score * self.pairnet_weights)
        return base.cs_z(pressure)

    def _lead_lag_signals(self, market: np.ndarray, resid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        empty = np.zeros(resid.shape[0])
        rows = resid.shape[1] - 1
        if rows < base.LEADLAG_MIN_OBS:
            return empty, empty.copy()
        leaders = np.vstack((market[None, :], resid))
        x_raw = leaders[:, -(rows + 1) : -1].T
        y_raw = resid[:, -rows:].T
        x_mean = np.mean(x_raw, axis=0)
        x_std = np.std(x_raw, axis=0) + EPS
        x = (x_raw - x_mean) / x_std
        y = (y_raw - np.mean(y_raw, axis=0)) / (np.std(y_raw, axis=0) + EPS)
        xtx, xty = x.T @ x / rows, x.T @ y / rows
        try:
            dense = np.linalg.solve(xtx + base.RIDGE * np.eye(xtx.shape[0]), xty)
            sparse = base.post_selection_ridge(xtx, xty, self._fit_group_lasso_support(resid))
        except np.linalg.LinAlgError:
            return empty, empty.copy()
        sparse[0] *= base.MARKET_LEADER_WEIGHT
        dense[0] *= base.MARKET_LEADER_WEIGHT
        latest = (leaders[:, -1] - x_mean) / x_std
        return base.cs_z(latest @ sparse), base.cs_z(latest @ dense)

    def _core_books(self, history: np.ndarray, prices: np.ndarray,
                    limits: np.ndarray, share_limits: np.ndarray,
                    market: np.ndarray, resid: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        signal, gate = self._lead_lag_signals(market, resid)
        reversal = base.REV_WEIGHT * base.cs_z(base.residual_reversal(resid))
        signal = base.cs_z(signal + reversal)
        gate = base.cs_z(gate + reversal)
        book = np.zeros(len(prices), dtype=int)
        book[1:] = np.clip(
            (limits[1:] * np.sign(signal) / prices[1:]).astype(int),
            -share_limits[1:], share_limits[1:]
        )
        raw_overlay = self._apply_pair_overlay(history, prices, book, share_limits, gate)
        return book, raw_overlay, signal, gate

    def _blend_features(self, history: np.ndarray, nt: int,
                        core_side: np.ndarray, overlay_innovation: np.ndarray,
                        market: np.ndarray, stocks: np.ndarray, resid: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        lowrank = base.lowrank_lag1_signal(stocks, market, resid)
        pairnet = self._pairnet_signal(np.log(history[1:]).T, nt)
        level = level_reversion_signal(resid)
        level_tail = np.where(np.abs(level) >= LEVEL_TAIL_Z, level, 0.0)
        pairnet_rank = pairnet * (rankdata(np.abs(pairnet), method="average") / len(pairnet))
        features = np.column_stack((
            core_side, overlay_innovation, lowrank, pairnet,
            level, level_tail, pairnet_rank
        ))
        prior_score = core_side + lowrank + pairnet + 0.25 * level
        prior_side = np.sign(prior_score)
        prior_side = np.where(prior_side == 0.0, core_side, prior_side)
        return features, prior_side, {
            "lowrank": lowrank, "pairnet": pairnet, "level": level,
            "level_tail": level_tail, "pairnet_rank": pairnet_rank,
        }

    def _rebuild_blend_history(self, history: np.ndarray) -> None:
        nt = history.shape[1]
        self.replay_rows = []
        self._reset_core(history.shape[0])
        self.replaying = True
        try:
            for decision_nt in range(MIN_OBS, nt):
                self._advance(history[:, :decision_nt])
        finally:
            self.replaying = False
        rows = self.replay_rows[-BLEND_WINDOW:]
        self.features = [features.copy() for _, features in rows]
        self.targets = [
            np.sign(history[1:, decision_nt] / history[1:, decision_nt - 1] - 1.0)
            for decision_nt, _ in rows
        ]
        self.replay_rows = []
        self.weights = BLEND_PRIOR_WEIGHTS.copy()
        self.pending_features = None
        self.pending_nt = 0
        self.blend_ready = True

    def _label_pending(self, history: np.ndarray) -> None:
        nt = history.shape[1]
        if self.pending_features is None or nt != self.pending_nt + 1:
            return
        realised = history[1:, -1] / history[1:, -2] - 1.0
        self.features.append(self.pending_features.copy())
        self.targets.append(np.sign(realised))
        self.features = self.features[-BLEND_WINDOW:]
        self.targets = self.targets[-BLEND_WINDOW:]

    def _fit_weights(self) -> np.ndarray:
        if len(self.features) < BLEND_WINDOW:
            self.weights = BLEND_PRIOR_WEIGHTS.copy()
            return self.weights.copy()
        x = np.asarray(self.features[-BLEND_WINDOW:], dtype=float).reshape(-1, 7)
        y = np.asarray(self.targets[-BLEND_WINDOW:], dtype=float).reshape(-1)
        scale = np.sqrt(np.mean(x * x, axis=0)) + 1e-12
        scaled = x / scale

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            margin = y * (scaled @ beta)
            loss = float(np.mean(np.logaddexp(0.0, -margin)) + 0.5 * BLEND_L2 * (beta @ beta))
            gradient = -(scaled.T @ (y * expit(-margin))) / len(y) + BLEND_L2 * beta
            return loss, gradient

        result = minimize(
            objective, np.maximum(self.weights * scale, 0.0),
            method="L-BFGS-B", jac=True, bounds=((0.0, None),) * 7,
            options={"maxiter": BLEND_MAX_ITER, "ftol": 1e-11, "gtol": 1e-8},
        )
        weights = np.maximum(result.x, 0.0) / scale
        if result.success and np.all(np.isfinite(weights)) and np.sum(weights) >= 1e-14:
            self.weights = weights
        return self.weights.copy()

    def _advance(self, history: np.ndarray) -> S2AuditStep | None:
        history = np.asarray(history, dtype=float)
        nins, nt = history.shape
        if (
            self.last_output is not None and self.last_history is not None
            and history.shape == self.last_history.shape
            and np.array_equal(history, self.last_history)
        ):
            return self.last_result
        if nt <= self.last_nt or not self._extends_last(history):
            self._reset_core(nins)
        self.last_nt = nt
        prices = history[:, -1]
        if nt < MIN_OBS or nins < 2 or np.any(prices <= 0):
            self._remember(history, np.zeros(nins, dtype=int))
            return None

        limits = base.position_limits(nins)
        share_limits = (limits / prices).astype(int)
        self._refresh_pair_books(history, nt)
        returns = np.diff(np.log(history), axis=1)
        market, stocks = returns[0], returns[1:]
        resid = residuals_exact(stocks, market)
        base_core_pos, raw_overlay_pos, core_signal, _gate = self._core_books(
            history, prices, limits, share_limits, market, resid
        )
        core_side = np.sign(base_core_pos[1:])
        overlay_innovation = 0.5 * (np.sign(raw_overlay_pos[1:]) - core_side)
        features, prior_side, components = self._blend_features(
            history, nt, core_side, overlay_innovation, market, stocks, resid
        )
        if self.replaying:
            self.replay_rows.append((nt, features))
            self._remember(history, base_core_pos)
            return None

        exact_algo = algo_tilt_position(core_side, prior_side, prices[0], int(share_limits[0]))
        self._label_pending(history)
        fitted_weights = self._fit_weights()
        scores = {
            "S2_exact": features @ fitted_weights,
            "S2_fixed7": features @ BLEND_PRIOR_WEIGHTS,
            "S2_pruned3": base.cs_z(
                core_signal + components["pairnet"] + 0.5 * components["lowrank"]
            ),
        }
        sides = {key: np.sign(score) for key, score in scores.items()}
        for key in sides:
            sides[key] = np.where(sides[key] == 0.0, core_side, sides[key])
        if exact_algo == 0:
            exact_algo = idle_algo_position(sides["S2_exact"], market, int(share_limits[0]))
        if exact_algo == 0 and nt >= 2:
            latest = np.log(history[1:, -1] / history[1:, -2])
            completion_side = int(np.sign(np.median(latest)))
            if completion_side:
                exact_algo = completion_side * int(share_limits[0])

        positions: dict[str, np.ndarray] = {}
        for key, side in sides.items():
            pos = np.zeros(nins, dtype=int)
            pos[0] = exact_algo
            pos[1:] = np.clip(
                (limits[1:] * side / prices[1:]).astype(int),
                -share_limits[1:], share_limits[1:]
            )
            positions[key] = pos
        self.pending_features = features.copy()
        self.pending_nt = nt
        self.live_history = history.copy()
        self._remember(history, positions["S2_exact"])
        result = S2AuditStep(
            positions=positions,
            sides={key: value.copy() for key, value in sides.items()},
            exact_algo=int(exact_algo), weights=fitted_weights.copy(),
            core_side=core_side.copy(), prior_side=prior_side.copy(),
            overlay_nonzero=int(np.count_nonzero(overlay_innovation)),
            pairnet_fit_nt=int(self.pairnet_fit_nt or 0),
            hardpair_fit_nt=int(self.hard_fit_nt),
            support_fraction=float(np.mean(self.lasso_support)) if self.lasso_support is not None else 1.0,
        )
        self.last_result = result
        return result

    def step(self, history: np.ndarray) -> S2AuditStep:
        history = np.asarray(history, dtype=float)
        nt = history.shape[1]
        if not self.replaying and nt >= MIN_OBS and (
            not self.blend_ready or not self._extends_live(history)
        ):
            self._rebuild_blend_history(history)
        result = self._advance(history)
        if result is None:
            raise RuntimeError("live Simple2 step produced no result")
        return result

_STRATEGY = Simple2SourceReproduction()


def getMyPosition(prcSoFar):
    """Return the desired 51-instrument integer position vector."""
    history = np.asarray(prcSoFar, dtype=float)
    if history.ndim != 2:
        raise ValueError("prcSoFar must be a two-dimensional price array")
    nins, nt = history.shape
    if nt == 0 or nt < MIN_OBS or nins < 2:
        return np.zeros(nins, dtype=int)
    if np.any(history[:, -1] <= 0.0):
        return np.zeros(nins, dtype=int)
    result = _STRATEGY.step(history)
    output = np.asarray(result.positions["S2_exact"], dtype=int)
    if output.shape != (nins,):
        raise RuntimeError("strategy returned an invalid position shape")
    return output.copy()

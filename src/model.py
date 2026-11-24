"""Train the models and score current minor leaguers.

Models (one set for hitters, one for pitchers), all gradient-boosted trees:
  1. P(MLB)   - chance of debuting within 7 years (trained on every MiLB player 2005-2019)
  2. ETA      - years until debut (trained on players who made it)
  3. Outcomes - first-3-seasons MLB stats and 6-season WAR, each with 3 quantile
                models: floor (low percentile), expected (median), ceiling (high percentile)
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from . import features as F

# Floor / expected / ceiling. Raw 10/90 quantile models came out a bit narrow in
# cross-validation (~73% coverage), so 7/93 is used to give ~80% real coverage.
QUANTS = (0.07, 0.5, 0.93)
HORIZON = 7  # years a player has to reach MLB to count as "made it"

H_TARGETS = ["avg_idx", "obp_idx", "slg_idx", "hr_idx", "sb_idx", "k_idx", "bb_idx",
             "wrc_plus", "war3", "war6"]
P_TARGETS = ["era_idx", "fip_idx", "whip_idx", "k9_idx", "bb9_idx", "gs_share3", "war3", "war6"]

# Zero-heavy targets where low-quantile boosting gets stuck at 0: their floor/ceiling
# come from how real outcomes spread around cross-validated predictions instead.
RATIO_TARGETS = {"sb_idx"}

GBM = dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40,
           l2_regularization=1.0, random_state=7)


def _make(kind, cat_mask, q=None):
    if kind == "clf":
        return HistGradientBoostingClassifier(categorical_features=cat_mask, **GBM)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                         categorical_features=cat_mask, **GBM)


def _wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(cw, q)])


def _fit(model, X, y, w=None, X_pred=None):
    with threadpool_limits(1):
        model.fit(X, y, sample_weight=w)
        if X_pred is None:
            return model
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_pred)[:, 1]
        return model.predict(X_pred)


class GroupModel:
    """All models for one player type ('H' hitters or 'P' pitchers)."""

    def __init__(self, kind):
        self.kind = kind
        self.features = F.H_FEATURES if kind == "H" else F.P_FEATURES
        self.targets = H_TARGETS if kind == "H" else P_TARGETS
        self.size_col = "pa3" if kind == "H" else "ip3"
        self.size_full = 600 if kind == "H" else 150
        self.cat_mask = [f == "pos_grp" for f in self.features]
        self.models, self.metrics = {}, {}

    def X(self, d):
        return d[self.features].to_numpy(float)

    # ------------------------------------------------------------------
    def datasets(self, snap, outcomes, veterans, last_season):
        snap = snap[~snap.pid.isin(veterans)]
        snap = snap[(snap.w_3y_raw >= (100 if self.kind == "H" else 80)) &
                    (snap.w_season >= (20 if self.kind == "H" else 15))]
        d = snap.merge(outcomes, on="pid", how="left")
        # P(MLB) + ETA: snapshots old enough that the 7-year outcome is known
        pm = d[(d.season <= last_season - HORIZON) & ~(d.debut <= d.season)].copy()
        pm["made"] = (pm.debut - pm.season <= HORIZON).astype(int)
        eta = pm[(pm.made == 1) & (pm.season <= last_season - HORIZON - 2)].copy()
        eta["years"] = eta.debut - eta.season
        # Outcomes: pre-debut snapshots of players who made it
        oc = d[(d.debut > d.season) & (d.debut - d.season <= HORIZON)].copy()
        oc["w"] = 1.0 / oc.groupby("pid").pid.transform("size")
        return pm, eta, oc

    def _target_rows(self, oc, t):
        rows = oc[oc[t].notna()]
        w = rows.w
        if t not in ("war3", "war6"):
            rows = rows[rows[self.size_col] >= (150 if self.kind == "H" else 40)]
            w = rows.w * (rows[self.size_col].clip(upper=self.size_full) / self.size_full)
        return rows, w

    # ------------------------------------------------------------------
    def fit(self, pm, eta, oc, evaluate=True):
        if evaluate:
            self._evaluate(pm, eta, oc)
        jobs = [("p_mlb", _make("clf", self.cat_mask), self.X(pm), pm.made, None),
                ("eta", _make("reg", self.cat_mask, 0.5), self.X(eta), eta.years, None)]
        for t in self.targets:
            rows, w = self._target_rows(oc, t)
            for q in (QUANTS if t not in RATIO_TARGETS else (0.5,)):
                jobs.append(((t, q), _make("reg", self.cat_mask, q), self.X(rows), rows[t], w))
        fitted = Parallel(n_jobs=-1)(delayed(_fit)(m, X, y, w) for _, m, X, y, w in jobs)
        self.models = {name: m for (name, *_), m in zip(jobs, fitted)}
        self.ratio_bands = {t: self._ratio_band(oc, t) for t in self.targets if t in RATIO_TARGETS}
        self.train_n = dict(p_mlb=len(pm), eta=len(eta), outcomes=int(oc.pid.nunique()))
        return self

    def _ratio_band(self, oc, t):
        """Spread of actual/predicted for out-of-fold medians, by prediction quintile."""
        rows, w = self._target_rows(oc, t)
        X, y, w = self.X(rows), rows[t].to_numpy(), w.to_numpy()
        splits = list(GroupKFold(5).split(rows, groups=rows.pid))
        res = Parallel(n_jobs=-1)(delayed(_fit)(_make("reg", self.cat_mask, 0.5), X[tr], y[tr], w[tr], X[te])
                                  for tr, te in splits)
        m = np.zeros(len(y))
        for (_, te), r in zip(splits, res):
            m[te] = r
        m = np.maximum(m, 1e-3)
        edges = np.quantile(m, [0.2, 0.4, 0.6, 0.8])
        b = np.searchsorted(edges, m)
        ratio = y / m
        bands = [(_wquantile(ratio[b == i], w[b == i], QUANTS[0]),
                  _wquantile(ratio[b == i], w[b == i], QUANTS[2])) for i in range(5)]
        return edges, bands

    def _evaluate(self, pm, eta, oc):
        """5-fold cross-validation grouped by player (a player is never in train and test)."""
        gkf = GroupKFold(5)
        tasks, slots = [], []

        def add(name, model, d, y, w=None):
            for tr, te in gkf.split(d, groups=d.pid):
                tasks.append(delayed(_fit)(model, self.X(d.iloc[tr]), y.iloc[tr],
                                           None if w is None else w.iloc[tr], self.X(d.iloc[te])))
                slots.append((name, te))

        add("p_mlb", _make("clf", self.cat_mask), pm, pm.made)
        add("eta", _make("reg", self.cat_mask, 0.5), eta, eta.years)
        eval_targets = ["war6", "wrc_plus" if self.kind == "H" else "era_idx"]
        data = {}
        for t in eval_targets:
            rows, w = self._target_rows(oc, t)
            data[t] = (rows, w)
            for q in QUANTS:
                add((t, q), _make("reg", self.cat_mask, q), rows, rows[t], w)
        results = Parallel(n_jobs=-1)(tasks)

        pred = {}
        sizes = {"p_mlb": len(pm), "eta": len(eta), **{(t, q): len(data[t][0]) for t in eval_targets for q in QUANTS}}
        for (name, te), r in zip(slots, results):
            pred.setdefault(name, np.zeros(sizes[name]))[te] = r
        self.metrics["p_mlb_auc"] = round(roc_auc_score(pm.made, pred["p_mlb"]), 3)
        bins = pd.cut(pred["p_mlb"], [0, .1, .3, .5, .7, .9, 1], include_lowest=True)
        self.metrics["p_mlb_calibration"] = {
            str(b): [round(float(pred["p_mlb"][bins == b].mean()), 3), round(float(pm.made[bins == b].mean()), 3)]
            for b in bins.categories if (bins == b).sum() > 0}
        self.metrics["eta_mae_years"] = round(mean_absolute_error(eta.years, np.clip(np.round(pred["eta"]), 1, 7)), 2)
        for t in eval_targets:
            rows, w = data[t]
            lo, mid, hi = (pred[(t, q)] for q in QUANTS)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            y = rows[t].to_numpy()
            last = (rows.debut - rows.season == 1).to_numpy()  # final MiLB season before debut
            self.metrics[t] = dict(
                spearman=round(float(spearmanr(mid, y).statistic), 3),
                spearman_final_season=round(float(spearmanr(mid[last], y[last]).statistic), 3),
                coverage=round(float(np.average((y >= lo) & (y <= hi), weights=w)), 3))

    # ------------------------------------------------------------------
    def predict(self, snap):
        X = self.X(snap)
        out = pd.DataFrame(index=snap.index)
        out["p_mlb"] = self.models["p_mlb"].predict_proba(X)[:, 1]
        out["eta_years"] = np.clip(np.round(self.models["eta"].predict(X)), 1, HORIZON)
        for t in self.targets:
            if t in RATIO_TARGETS:
                mid = np.maximum(self.models[(t, 0.5)].predict(X), 0)
                edges, bands = self.ratio_bands[t]
                lo_r, hi_r = np.array(bands)[np.searchsorted(edges, np.maximum(mid, 1e-3))].T
                qs = np.sort(np.column_stack([mid * lo_r, mid, mid * hi_r]), axis=1)
            else:
                qs = np.sort(np.column_stack([self.models[(t, q)].predict(X) for q in QUANTS]), axis=1)
            for i, name in enumerate(("lo", "mid", "hi")):
                out[f"{t}_{name}"] = qs[:, i]
        return out

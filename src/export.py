"""Turn model output into readable stat lines and write the website data."""
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
SITE_DATA = ROOT / "docs" / "data"

LABELS = [(25, "Superstar"), (17, "All-Star"), (10, "Above-average regular"),
          (5, "Everyday regular"), (1.5, "Role player"), (-99, "Depth / up-and-down")]
LABELS_P = [(25, "Ace"), (17, "All-Star arm"), (10, "No. 2-3 starter"),
            (5, "Mid-rotation / high-leverage RP"), (1.5, "Back-end SP / reliever"),
            (-99, "Depth / up-and-down")]


def label(war6, kind):
    for cut, name in (LABELS if kind == "H" else LABELS_P):
        if war6 >= cut:
            return name


def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def _trip(row, col, scale=1.0, nd=3, lower_better=False):
    """[floor, expected, ceiling] for one target."""
    lo, mid, hi = (row[f"{col}_{q}"] * scale for q in ("lo", "mid", "hi"))
    vals = [hi, mid, lo] if lower_better else [lo, mid, hi]
    return [_r(v, nd) for v in vals]


def hitter_projection(row, lg):
    p = {
        "AVG": _trip(row, "avg_idx", lg["avg"]),
        "OBP": _trip(row, "obp_idx", lg["obp"]),
        "SLG": _trip(row, "slg_idx", lg["slg"]),
        "HR/600": _trip(row, "hr_idx", lg["hr"] * 600, 0),
        "SB/600": _trip(row, "sb_idx", lg["sb"] * 600, 0),
        "K%": _trip(row, "k_idx", lg["k"] * 100, 1, lower_better=True),
        "BB%": _trip(row, "bb_idx", lg["bb"] * 100, 1),
        "wRC+": _trip(row, "wrc_plus", 1, 0),
        "WAR/yr": _trip(row, "war3", 1 / 3, 1),
    }
    p["OPS"] = [_r(a + b) for a, b in zip(p["OBP"], p["SLG"])]
    return p


def pitcher_projection(row, lg):
    return {
        "ERA": _trip(row, "era_idx", lg["era"], 2, lower_better=True),
        "FIP": _trip(row, "fip_idx", lg["fip"], 2, lower_better=True),
        "WHIP": _trip(row, "whip_idx", lg["whip"], 2, lower_better=True),
        "K/9": _trip(row, "k9_idx", lg["k9"], 1),
        "BB/9": _trip(row, "bb9_idx", lg["bb9"], 1, lower_better=True),
        "WAR/yr": _trip(row, "war3", 1 / 3, 1),
    }


def role(gs_share):
    return "SP" if gs_share >= 0.5 else ("SP/RP" if gs_share >= 0.2 else "RP")


# --------------------------------------------------------------------------
def find_comps(train, pool, kind, k=3):
    """Most similar pre-debut MiLB profiles among players with a full 6-year MLB record."""
    cols = (["age", "lvl_max", "ops_plus", "k_rel", "bb_rel", "iso_rel", "age_rel", "pos_grp"]
            if kind == "H" else
            ["age", "lvl_max", "fip_minus", "k_rel", "bb_rel", "gs_share", "age_rel"])
    weights = np.array([1.5, 1.5, 1.2, 1, 1, 1, 1, 0.5] if kind == "H" else [1.5, 1.5, 1.2, 1, 1, 1, 1])
    ref = train[train.war6.notna()].dropna(subset=cols).reset_index(drop=True)
    mu, sd = ref[cols].mean(), ref[cols].std()
    Z = lambda d: ((d[cols].fillna(mu) - mu) / sd).to_numpy() * weights
    nn = NearestNeighbors(n_neighbors=40).fit(Z(ref))
    _, idx = nn.kneighbors(Z(pool))
    out = []
    for row in idx:
        seen, comps = set(), []
        for i in row:
            r = ref.iloc[i]
            if r.pid in seen:
                continue
            seen.add(r.pid)
            stat = (f"{r.wrc_plus:.0f} wRC+" if kind == "H" else f"{r.era_raw:.2f} ERA") \
                if pd.notna(r.get("wrc_plus" if kind == "H" else "era_raw")) else ""
            comps.append({"name": r["name"], "debut": int(r.debut), "war6": _r(r.war6, 1),
                          "yrs1_3": stat, "at": f"{int(r.season)}, age {int(r.age)}"})
            if len(comps) == k:
                break
        out.append(comps)
    return out


def milb_history(rows, kind):
    rows = rows.sort_values(["season", "lvl"])
    hist = []
    for _, r in rows.iterrows():
        if kind == "H":
            ab = max(r.AB, 1)
            obp = (r.H + r.BB + r.HBP) / max(r.AB + r.BB + r.HBP + r.SF, 1)
            hist.append([int(r.season), r.level, r.team if isinstance(r.team, str) else "2+ teams", int(r.age), int(r.PA),
                         _r(r.H / ab), _r(obp), _r(r.TB / ab), int(r.HR), int(r.SB),
                         _r(100 * r.SO / r.PA, 1), _r(100 * r.BB / r.PA, 1)])
        else:
            ip = r.OUTS / 3
            hist.append([int(r.season), r.level, r.team if isinstance(r.team, str) else "2+ teams", int(r.age),
                         f"{r.OUTS // 3}.{r.OUTS % 3}", int(r.G), int(r.GS),
                         _r(9 * r.ER / ip if ip else np.nan, 2),
                         _r((r.H + r.BB) / ip if ip else np.nan, 2),
                         _r(9 * r.SO / ip if ip else np.nan, 1),
                         _r(9 * r.BB / ip if ip else np.nan, 1)])
    return hist


def build_players(kind, pool, preds, comps, lines, lg, bios, orgs):
    LEVEL_NAME = {0: "ROK", 1: "A-", 2: "A", 3: "A+", 4: "AA", 5: "AAA"}
    players = []
    by_pid = dict(tuple(lines[lines.pid.isin(set(pool.pid))].groupby("pid")))
    for (i, s), (_, p), cp in zip(pool.iterrows(), preds.iterrows(), comps):
        bio = bios.get(int(s.pid), {})
        row = {**s.to_dict(), **p.to_dict()}
        war = _trip(row, "war6", 1, 1)
        war_mean = 0.3 * row["war6_lo"] + 0.4 * row["war6_mid"] + 0.3 * row["war6_hi"]
        debuted = pd.notna(s.get("mlb_debut"))
        p_mlb = 1.0 if debuted else float(p.p_mlb)
        eta = int(s.mlb_debut) if debuted else int(s.season + p.eta_years)
        org, team = orgs(bio, s)
        pos = bio.get("pos") or ("P" if kind == "P" else "")
        if kind == "P":
            pos = role(row["gs_share3_mid"]) if pos in ("P", "") else pos
        players.append({
            "id": int(s.pid), "name": bio.get("name") or s["name"],
            "type": kind, "pos": pos, "org": org, "team": team,
            "level": LEVEL_NAME[int(s.lvl_max)], "age": int(s.age),
            "bt": bio.get("bt", ""), "born": bio.get("born", ""),
            "eta": eta, "debuted": bool(debuted), "p_mlb": round(p_mlb, 3),
            "proj": hitter_projection(row, lg) if kind == "H" else pitcher_projection(row, lg),
            "war6": war, "war6_mean": round(war_mean, 1),
            "outlook": [label(w, kind) for w in war],
            "score": round(p_mlb * max(war_mean, 0), 2),
            "comps": cp,
            "milb": milb_history(by_pid[s.pid], kind),
        })
    return players


def stamp_site():
    """Give app.js / style.css / data a new ?v= so browsers never mix cached old files with new ones."""
    index = SITE_DATA.parent / "index.html"
    html = index.read_text(encoding="utf-8")
    index.write_text(re.sub(r"\?v=[\w-]*", "?v=" + datetime.now().strftime("%Y%m%d%H%M"), html),
                     encoding="utf-8")


def write_site(players, meta):
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    players = sorted(players, key=lambda p: -p["score"])
    rank = 0
    for p in players:
        p["rank"] = None
        if p["org"] not in ("", "Free agent") and rank < 100:
            rank += 1
            p["rank"] = rank
    meta["generated"] = date.today().isoformat()
    meta["n_players"] = len(players)
    compact = lambda d: json.dumps(d, separators=(",", ":"), ensure_ascii=False)
    (SITE_DATA / "players.json").write_text(compact(players), encoding="utf-8")
    (SITE_DATA / "top100.json").write_text(compact([p for p in players if p["rank"]]), encoding="utf-8")
    stamp_site()
    (SITE_DATA / "meta.json").write_text(json.dumps(meta, indent=1))
    return players

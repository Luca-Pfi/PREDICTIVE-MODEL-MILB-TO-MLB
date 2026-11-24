"""Turn raw API tables into model-ready data.

Key ideas
- Every MiLB stat line is compared to its league that season (K% vs league K%,
  OPS vs league OPS, age vs league average age...), so a .800 OPS in a pitcher's
  league counts more than in a hitter's league.
- A "snapshot" is a player's profile at the end of one MiLB season: his last
  three seasons blended (recent seasons weigh more), his level, age, experience.
- Outcomes come from MLB: first 3 seasons after debut (rates relative to MLB
  league average, so they are era-neutral) and WAR over the first 6 seasons.
"""
import numpy as np
import pandas as pd

from . import fetch

LEVEL_ORD = {"ROK": 0, "A-": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5}
POS_GROUP = {"C": 0, "1B": 1, "DH": 1, "2B": 2, "3B": 2, "IF": 2, "TWP": 2,
             "SS": 3, "CF": 4, "OF": 5, "LF": 5, "RF": 5}
RECENCY = 0.6          # weight multiplier per season back in time
WINDOW = 3             # seasons blended into a snapshot


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _outs(ip):
    """'123.2' innings -> 371 outs."""
    if ip in (None, ""):
        return 0
    whole, _, frac = str(ip).partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


# --------------------------------------------------------------------------
# 1. raw lines
# --------------------------------------------------------------------------
def load_lines(last_season):
    hit, pit, saber_h, saber_p = [], [], [], []
    for season in range(fetch.FIRST_SEASON, last_season + 1):
        for sport, level in fetch.LEVELS.items():
            if (sport != 1 and season == 2020) or (sport == 15 and season > 2020):
                continue
            for s in fetch.season_stats(season, sport, "hitting"):
                st = s["stat"]
                pos = s.get("position", {}).get("abbreviation", "")
                hit.append(dict(
                    pid=s["player"]["id"], name=s["player"].get("fullName", ""), season=season,
                    level=level, age=_f(st.get("age")), pos=pos,
                    team_id=s.get("team", {}).get("id"), team=s.get("team", {}).get("name"),
                    league_id=s.get("league", {}).get("id"),
                    G=st.get("gamesPlayed", 0), PA=st.get("plateAppearances", 0),
                    AB=st.get("atBats", 0), H=st.get("hits", 0), B2=st.get("doubles", 0),
                    B3=st.get("triples", 0), HR=st.get("homeRuns", 0), BB=st.get("baseOnBalls", 0),
                    HBP=st.get("hitByPitch", 0), SO=st.get("strikeOuts", 0),
                    SB=st.get("stolenBases", 0), CS=st.get("caughtStealing", 0),
                    SF=st.get("sacFlies", 0)))
            for s in fetch.season_stats(season, sport, "pitching"):
                st = s["stat"]
                pit.append(dict(
                    pid=s["player"]["id"], name=s["player"].get("fullName", ""), season=season,
                    level=level, age=_f(st.get("age")),
                    team_id=s.get("team", {}).get("id"), team=s.get("team", {}).get("name"),
                    league_id=s.get("league", {}).get("id"),
                    G=st.get("gamesPitched", st.get("gamesPlayed", 0)), GS=st.get("gamesStarted", 0),
                    OUTS=st.get("outs") or _outs(st.get("inningsPitched")),
                    BF=st.get("battersFaced", 0), H=st.get("hits", 0), HR=st.get("homeRuns", 0),
                    BB=st.get("baseOnBalls", 0), HBP=st.get("hitBatsmen", 0),
                    SO=st.get("strikeOuts", 0), ER=st.get("earnedRuns", 0),
                    GO=st.get("groundOuts", 0), AO=st.get("airOuts", 0)))
        for s in fetch.season_stats(season, 1, "hitting", "sabermetrics"):
            st = s["stat"]
            saber_h.append(dict(pid=s["player"]["id"], season=season,
                                WAR=_f(st.get("war")), WRCP=_f(st.get("wRcPlus"))))
        for s in fetch.season_stats(season, 1, "pitching", "sabermetrics"):
            st = s["stat"]
            saber_p.append(dict(pid=s["player"]["id"], season=season,
                                WAR=_f(st.get("war")), FIP=_f(st.get("fip"))))
    hit, pit = pd.DataFrame(hit), pd.DataFrame(pit)
    sh = pd.DataFrame(saber_h).groupby(["pid", "season"], as_index=False).agg(
        WAR=("WAR", "sum"), WRCP=("WRCP", "mean"))
    sp = pd.DataFrame(saber_p).groupby(["pid", "season"], as_index=False).agg(
        WAR=("WAR", "sum"), FIP=("FIP", "mean"))
    return hit, pit, sh, sp


# --------------------------------------------------------------------------
# 2. league-relative MiLB lines
# --------------------------------------------------------------------------
def _env_key(df, weight_col, min_weight):
    """Normalise by league when the league is known and big enough, else by level."""
    size = df.groupby(["season", "league_id"])[weight_col].transform("sum")
    use_league = df["league_id"].notna() & (size >= min_weight)
    return np.where(use_league, "L" + df["league_id"].astype("Int64").astype(str),
                    "S" + df["level"].astype(str))


def hitter_lines(hit):
    h = hit[(hit.level != "MLB") & (hit.pos != "P") & (hit.PA > 0)].copy()
    h["env"] = _env_key(h, "PA", 3000)
    h["TB"] = h.H + h.B2 + 2 * h.B3 + 3 * h.HR
    sums = ["PA", "AB", "H", "TB", "HR", "BB", "HBP", "SO", "SB", "CS", "SF"]
    h["agePA"] = h.age * h.PA
    lg = h.groupby(["season", "env"])[sums + ["agePA"]].transform("sum")
    def rates(d):
        return dict(
            k=d.SO / d.PA, bb=d.BB / d.PA, hr=d.HR / d.PA, sb=(d.SB + 0.5 * d.CS) / d.PA,
            iso=(d.TB - d.H) / d.AB.clip(lower=1),
            babip=(d.H - d.HR) / (d.AB - d.SO - d.HR + d.SF).clip(lower=1),
            obp=(d.H + d.BB + d.HBP) / (d.AB + d.BB + d.HBP + d.SF).clip(lower=1),
            slg=d.TB / d.AB.clip(lower=1))
    me, lgr = rates(h), rates(lg)
    for c in ["k", "bb", "hr", "sb", "iso", "babip"]:
        h[c + "_rel"] = me[c] - lgr[c]
    h["ops_plus"] = 100 * (me["obp"] / lgr["obp"] + me["slg"] / lgr["slg"] - 1)
    h["age_rel"] = h.age - lg.agePA / lg.PA
    h["lvl"] = h.level.map(LEVEL_ORD)
    h["w"] = h.PA
    return h


def pitcher_lines(pit):
    p = pit[(pit.level != "MLB") & (pit.BF > 0)].copy()
    p["env"] = _env_key(p, "BF", 3000)
    p["ageBF"] = p.age * p.BF
    sums = ["BF", "OUTS", "H", "HR", "BB", "HBP", "SO", "ER", "GO", "AO"]
    lg = p.groupby(["season", "env"])[sums + ["ageBF"]].transform("sum")
    def rates(d):
        ip = (d.OUTS / 3).clip(lower=1 / 3)
        return dict(
            k=d.SO / d.BF, bb=(d.BB + d.HBP) / d.BF, hr=d.HR / d.BF,
            era=9 * d.ER / ip, whip=(d.H + d.BB) / ip,
            fip_raw=(13 * d.HR + 3 * (d.BB + d.HBP) - 2 * d.SO) / ip,
            gb=d.GO / (d.GO + d.AO).clip(lower=1),
            babip=(d.H - d.HR) / (d.BF - d.SO - d.HR - d.BB - d.HBP).clip(lower=1))
    me, lgr = rates(p), rates(lg)
    cfip = lgr["era"] - lgr["fip_raw"]
    for c in ["k", "bb", "hr", "gb", "babip", "whip"]:
        p[c + "_rel"] = me[c] - lgr[c]
    p["kbb_rel"] = p.k_rel - p.bb_rel
    p["era_minus"] = (100 * me["era"] / lgr["era"]).clip(0, 400)
    p["fip_minus"] = (100 * (me["fip_raw"] + cfip) / lgr["era"]).clip(0, 400)
    p["age_rel"] = p.age - lg.ageBF / lg.BF
    p["lvl"] = p.level.map(LEVEL_ORD)
    p["gs_share"] = p.GS / p.G.clip(lower=1)
    p["w"] = p.BF
    return p


# --------------------------------------------------------------------------
# 3. snapshots: one row per player per MiLB season
# --------------------------------------------------------------------------
def _snapshots(lines, rel_cols, top_cols, extra_mean_cols):
    lines = lines.copy()
    lines["lvl_w"] = lines.lvl
    mean_cols = rel_cols + ["age_rel", "lvl_w"] + extra_mean_cols
    for c in mean_cols:
        lines["x_" + c] = lines[c] * lines.w
    # per player-season aggregate (all levels that season)
    ps = lines.groupby(["pid", "season"]).agg(
        name=("name", "last"), age=("age", "max"), lvl_max=("lvl", "max"),
        w_season=("w", "sum"), **{"x_" + c: ("x_" + c, "sum") for c in mean_cols}).reset_index()

    # performance at the highest level reached that season
    top = lines.loc[lines.lvl == lines.groupby(["pid", "season"]).lvl.transform("max")]
    top = top.groupby(["pid", "season"]).agg(
        w_top=("w", "sum"), **{"xt_" + c: ("x_" + c, "sum") for c in top_cols}).reset_index()
    for c in top_cols:
        top[c + "_top"] = top["xt_" + c] / top.w_top
    top = top.drop(columns=["xt_" + c for c in top_cols])

    # career upper-minors (AA/AAA) running totals
    up = lines[lines.lvl >= 4].groupby(["pid", "season"]).agg(
        w_up=("w", "sum"), **{"xu_" + c: ("x_" + c, "sum") for c in top_cols}).reset_index()

    ps = ps.sort_values(["pid", "season"])
    ps["yrs_pro"] = ps.groupby("pid").cumcount() + 1
    ps["w_career"] = ps.groupby("pid").w_season.cumsum()

    # recency-weighted blend of the last WINDOW seasons
    parts = []
    for k in range(WINDOW + 1):
        sh = ps[["pid", "season", "w_season"] + ["x_" + c for c in mean_cols]].copy()
        sh["S"] = sh.season + k
        sh.loc[(sh.season < 2020) & (sh.S >= 2020), "S"] += 1  # skip the 2020 gap
        sh["d"] = k
        parts.append(sh)
    blend = pd.concat(parts)
    valid = ps[["pid", "season"]].rename(columns={"season": "S"})
    blend = blend.merge(valid, on=["pid", "S"])
    blend = blend[blend.d < WINDOW]
    rw = RECENCY ** blend.d
    blend["w_season"] *= rw
    for c in mean_cols:
        blend["x_" + c] *= rw
    blend["w_raw"] = blend.w_season / rw
    agg = blend.drop(columns=["season", "d"]).groupby(["pid", "S"]).sum().reset_index()
    for c in mean_cols:
        agg[c] = agg["x_" + c] / agg.w_season
    agg = agg.rename(columns={"S": "season", "w_season": "w_3y", "w_raw": "w_3y_raw"})
    agg = agg[["pid", "season", "w_3y", "w_3y_raw"] + mean_cols]

    snap = ps[["pid", "season", "name", "age", "lvl_max", "w_season", "yrs_pro", "w_career"]]
    snap = snap.merge(agg, on=["pid", "season"]).merge(top, on=["pid", "season"], how="left")
    snap = snap.merge(up, on=["pid", "season"], how="left").sort_values(["pid", "season"])
    snap[["w_up"] + ["xu_" + c for c in top_cols]] = snap[["w_up"] + ["xu_" + c for c in top_cols]].fillna(0)
    g = snap.groupby("pid")
    snap["w_up_career"] = g.w_up.cumsum()
    for c in top_cols:
        cum = g["xu_" + c].cumsum()
        snap[c + "_up"] = np.where(snap.w_up_career > 0, cum / snap.w_up_career.replace(0, 1), np.nan)
    snap["log_w_career"] = np.log1p(snap.w_career)
    snap["log_w_up"] = np.log1p(snap.w_up_career)
    return snap.drop(columns=["w_up"] + ["xu_" + c for c in top_cols])


H_REL = ["k_rel", "bb_rel", "iso_rel", "babip_rel", "hr_rel", "sb_rel", "ops_plus"]
P_REL = ["k_rel", "bb_rel", "kbb_rel", "hr_rel", "gb_rel", "babip_rel", "whip_rel",
         "era_minus", "fip_minus"]


def hitter_snapshots(hl):
    snap = _snapshots(hl, H_REL, ["ops_plus", "k_rel", "iso_rel"], [])
    pos = (hl.sort_values("PA").groupby(["pid", "season"]).pos.last()
           .map(POS_GROUP).rename("pos_grp").reset_index())
    snap = snap.merge(pos, on=["pid", "season"], how="left")
    snap["pos_grp"] = snap.pos_grp.fillna(2)
    return snap


def pitcher_snapshots(pl):
    snap = _snapshots(pl, P_REL, ["fip_minus", "k_rel", "bb_rel"], ["gs_share"])
    ipg = pl.groupby(["pid", "season"]).agg(outs=("OUTS", "sum"), g=("G", "sum")).reset_index()
    ipg["ip_per_g"] = ipg.outs / 3 / ipg.g.clip(lower=1)
    return snap.merge(ipg[["pid", "season", "ip_per_g"]], on=["pid", "season"], how="left")


def load_draft(last_season):
    rows = []
    for y in range(fetch.FIRST_SEASON - 8, last_season + 1):
        rows += fetch.draft(y)
    d = pd.DataFrame(rows)
    d["bonus"] = d.bonus / 1e6
    return d.sort_values("draft_year")


def add_draft(snap, draft):
    """Latest draft selection known at the time of each snapshot (no look-ahead)."""
    s = snap.sort_values("season").reset_index(drop=True)
    d = draft.rename(columns={"draft_year": "season_d"})
    m = pd.merge_asof(s, d, left_on="season", right_on="season_d", by="pid", direction="backward")
    m["yrs_since_draft"] = m.season - m.season_d
    return m.drop(columns=["season_d"]).sort_values(["pid", "season"]).reset_index(drop=True)


DRAFT = ["draft_pick", "bonus", "yrs_since_draft"]
H_FEATURES = ["age", "age_rel", "lvl_max", "lvl_w", "yrs_pro", "w_season", "w_3y",
              "log_w_career", "log_w_up", *H_REL, "ops_plus_top", "k_rel_top", "iso_rel_top",
              "w_top", "ops_plus_up", "k_rel_up", "iso_rel_up", "pos_grp", *DRAFT]
P_FEATURES = ["age", "age_rel", "lvl_max", "lvl_w", "yrs_pro", "w_season", "w_3y",
              "log_w_career", "log_w_up", *P_REL, "gs_share", "ip_per_g",
              "fip_minus_top", "k_rel_top", "bb_rel_top", "w_top",
              "fip_minus_up", "k_rel_up", "bb_rel_up", *DRAFT]


# --------------------------------------------------------------------------
# 4. MLB outcomes
# --------------------------------------------------------------------------
def _years(df, debut, n):
    d = df.merge(debut, on="pid")
    return d[(d.season >= d.debut) & (d.season < d.debut + n)]


def hitter_outcomes(hit, saber_h, last_season):
    m = hit[(hit.level == "MLB") & (hit.pos != "P") & (hit.PA > 0)].copy()
    m["TB"] = m.H + m.B2 + 2 * m.B3 + 3 * m.HR
    cnt = ["PA", "AB", "H", "TB", "HR", "BB", "HBP", "SO", "SB", "SF"]
    lg = m.groupby("season")[cnt].sum()
    lg_rates = _h_rates(lg)
    m = m.merge(saber_h, on=["pid", "season"], how="left")
    for c, v in lg_rates.items():
        m["lg_" + c] = m.season.map(v)
    debut = m.groupby("pid").season.min().rename("debut").reset_index()
    y3 = _years(m, debut, 3)
    for c in lg_rates:
        y3["xlg_" + c] = y3["lg_" + c] * y3.PA
    y3["xWRCP"] = y3.WRCP * y3.PA
    a = y3.groupby("pid").sum(numeric_only=True)
    r = _h_rates(a)
    out = pd.DataFrame(index=a.index)
    out["pa3"] = a.PA
    for c in lg_rates:          # rate relative to the MLB average of those seasons
        out[c + "_idx"] = r[c] / (a["xlg_" + c] / a.PA)
    out["wrc_plus"] = a.xWRCP / a.PA
    out["war3"] = a.WAR
    y6 = _years(m, debut, 6).groupby("pid").WAR.sum()
    out = out.join(debut.set_index("pid")).join(y6.rename("war6"))
    out.loc[out.debut + 5 > last_season, "war6"] = np.nan
    out.loc[out.debut + 2 > last_season, [c for c in out if c not in ("debut", "war6")]] = np.nan
    return out.reset_index(), lg_rates


def _h_rates(d):
    return dict(
        avg=d.H / d.AB, obp=(d.H + d.BB + d.HBP) / (d.AB + d.BB + d.HBP + d.SF),
        slg=d.TB / d.AB, hr=d.HR / d.PA, sb=d.SB / d.PA, k=d.SO / d.PA, bb=d.BB / d.PA)


def _p_rates(d):
    ip = d.OUTS / 3
    return dict(era=9 * d.ER / ip, whip=(d.H + d.BB) / ip, k9=9 * d.SO / ip,
                bb9=9 * d.BB / ip, hr9=9 * d.HR / ip)


def pitcher_outcomes(pit, saber_p, last_season):
    m = pit[(pit.level == "MLB") & (pit.BF > 0)].copy()
    cnt = ["OUTS", "ER", "H", "BB", "SO", "HR", "BF"]
    lg = m.groupby("season")[cnt].sum()
    lg_rates = _p_rates(lg)
    lg_rates["fip"] = lg_rates["era"]
    m = m.merge(saber_p, on=["pid", "season"], how="left")
    m["IP"] = m.OUTS / 3
    for c, v in lg_rates.items():
        m["lg_" + c] = m.season.map(v)
    debut = m.groupby("pid").season.min().rename("debut").reset_index()
    y3 = _years(m, debut, 3)
    for c in lg_rates:
        y3["xlg_" + c] = y3["lg_" + c] * y3.IP
    y3["xFIP"] = y3.FIP * y3.IP
    a = y3.groupby("pid").sum(numeric_only=True)
    r = _p_rates(a)
    r["fip"] = a.xFIP / a.IP
    out = pd.DataFrame(index=a.index)
    out["ip3"] = a.IP
    for c in lg_rates:
        out[c + "_idx"] = r[c] / (a["xlg_" + c] / a.IP)
    out["gs_share3"] = a.GS / a.G.clip(lower=1)
    out["era_raw"] = r["era"]
    out["war3"] = a.WAR
    y6 = _years(m, debut, 6).groupby("pid").WAR.sum()
    out = out.join(debut.set_index("pid")).join(y6.rename("war6"))
    out.loc[out.debut + 5 > last_season, "war6"] = np.nan
    out.loc[out.debut + 2 > last_season, [c for c in out if c not in ("debut", "war6")]] = np.nan
    return out.reset_index(), lg_rates


def mlb_career_totals(hit, pit):
    """MLB AB / IP so far, used to decide who is still rookie eligible."""
    ab = hit[hit.level == "MLB"].groupby("pid").AB.sum().rename("mlb_ab")
    ip = (pit[pit.level == "MLB"].groupby("pid").OUTS.sum() / 3).rename("mlb_ip")
    first_h = hit[(hit.level == "MLB") & (hit.PA > 0)].groupby("pid").season.min().rename("mlb_debut_h")
    first_p = pit[(pit.level == "MLB") & (pit.BF > 0)].groupby("pid").season.min().rename("mlb_debut_p")
    return pd.concat([ab, ip, first_h, first_p], axis=1)

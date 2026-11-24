"""Full pipeline: download -> features -> train -> predict -> website data.

    python build.py            # uses cached API data when present
    python build.py --refresh  # re-download the latest season first
"""
import argparse
from datetime import date

import pandas as pd

from src import export as E
from src import features as F
from src import fetch
from src import model as M


def default_season():
    t = date.today()
    return t.year if t.month >= 10 else t.year - 1  # last completed MiLB season


def bios_for(pids):
    out = {}
    for p in fetch.people(pids):
        bats, throws = p.get("batSide", {}).get("code", ""), p.get("pitchHand", {}).get("code", "")
        out[p["id"]] = dict(
            name=p.get("fullName", ""), pos=p.get("primaryPosition", {}).get("abbreviation", ""),
            bt=f"{bats}/{throws}", born=p.get("birthDate", ""),
            team_id=(p.get("currentTeam") or {}).get("id"),
            debut=p.get("mlbDebutDate"))
    return out


def org_lookup(season):
    teams = {}
    for sport in (1, 11, 12, 13, 14, 16):
        for t in fetch.teams(season, sport):
            teams[t["id"]] = (t.get("parentOrgName") or t.get("name", ""), t.get("name", ""))

    def orgs(bio, snap_row):
        tid = bio.get("team_id")
        if tid in teams:
            return teams[tid]
        if bio and tid is None:
            return "Free agent", ""
        return teams.get(snap_row.get("team_id"), ("", ""))
    return orgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=default_season())
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-eval", action="store_true", help="skip cross-validation")
    args = ap.parse_args()
    last = args.season

    print(f"1/5 Downloading MLB Stats API data 2005-{last}")
    fetch.download_all(last, refresh_last=args.refresh)

    print("2/5 Building league-adjusted MiLB snapshots and MLB outcomes")
    hit, pit, sab_h, sab_p = F.load_lines(last)
    hl, pl = F.hitter_lines(hit), F.pitcher_lines(pit)
    draft = F.load_draft(last)
    hs, ps = F.add_draft(F.hitter_snapshots(hl), draft), F.add_draft(F.pitcher_snapshots(pl), draft)
    ho, lg_h = F.hitter_outcomes(hit, sab_h, last)
    po, lg_p = F.pitcher_outcomes(pit, sab_p, last)
    mlb = F.mlb_career_totals(hit, pit)

    # players who debuted before 2005 look like "new" MLB players in our window: drop them
    first = pd.concat([hs[["pid", "season"]], ps[["pid", "season"]]]).groupby("pid").season.min()
    early = fetch.people(first[first <= 2008].index)
    veterans = {p["id"] for p in early if (p.get("mlbDebutDate") or "9999")[:4] < "2005"}

    print("3/5 Training models (hitters, pitchers)")
    meta = {"season": last, "metrics": {}, "train": {}}
    trained = {}
    for kind, snap, out in (("H", hs, ho), ("P", ps, po)):
        gm = M.GroupModel(kind)
        pm, eta, oc = gm.datasets(snap, out, veterans, last)
        gm.fit(pm, eta, oc, evaluate=not args.no_eval)
        meta["metrics"][kind] = gm.metrics
        meta["train"][kind] = gm.train_n
        trained[kind] = (gm, oc)
        print(f"   {kind}: {gm.train_n}  {gm.metrics}")

    print("4/5 Scoring current minor leaguers")
    players = []
    orgs = org_lookup(last)
    for kind, snap, lines, lg in (("H", hs, hl, lg_h), ("P", ps, pl, lg_p)):
        gm, oc = trained[kind]
        pool = snap[(snap.season == last) &
                    (snap.w_3y_raw >= (60 if kind == "H" else 45)) &
                    (snap.w_season >= (20 if kind == "H" else 15))].copy()
        pool = pool.join(mlb, on="pid")
        pool = pool[(pool.mlb_ab.fillna(0) < 130) & (pool.mlb_ip.fillna(0) < 50) & (pool.age <= 30)]
        pool["mlb_debut"] = pool["mlb_debut_h" if kind == "H" else "mlb_debut_p"]
        team = lines[lines.season == last].sort_values("w").groupby("pid").team_id.last()
        pool["team_id"] = pool.pid.map(team)
        pool = pool.reset_index(drop=True)
        preds = gm.predict(pool)
        comps = E.find_comps(oc, pool, kind)
        bios = bios_for(pool.pid)
        lg_now = {k: float(v.loc[last]) for k, v in lg.items()}
        players += E.build_players(kind, pool, preds, comps, lines, lg_now, bios, orgs)
        print(f"   {kind}: {len(pool)} players scored")

    print("5/5 Writing website data to docs/data/")
    meta["league_mlb"] = {"H": {k: round(float(v.loc[last]), 4) for k, v in lg_h.items()},
                          "P": {k: round(float(v.loc[last]), 3) for k, v in lg_p.items()}}
    players = E.write_site(players, meta)
    for p in [p for p in players if p["rank"]][:10]:
        print(f"   #{p['rank']:>3} {p['name']:<24} {p['pos']:<5} {p['org']:<22} ETA {p['eta']}"
              f"  MLB {p['p_mlb']:.0%}  6yr WAR {p['war6']}")


if __name__ == "__main__":
    main()

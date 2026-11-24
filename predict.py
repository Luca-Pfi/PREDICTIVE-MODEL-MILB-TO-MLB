"""Print the full MLB projection for a minor leaguer.

    python predict.py "Eli Willits"
    python predict.py willits          # partial names / no accents are fine
    python predict.py --top 25         # top-25 prospect list
    python predict.py                  # asks for a name

Reads docs/data/players.json, which `python build.py` creates.
"""
import argparse
import difflib
import json
import sys
import unicodedata
from pathlib import Path

DATA = Path(__file__).resolve().parent / "docs" / "data"
W = 74


def plain(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def load():
    f = DATA / "players.json"
    if not f.exists():
        sys.exit("No predictions found. Run `python build.py` first.")
    meta = json.loads((DATA / "meta.json").read_text())
    return json.loads(f.read_text(encoding="utf-8")), meta


def find(players, query):
    q = plain(query)
    exact = [p for p in players if plain(p["name"]) == q]
    if exact:
        return exact
    words = q.split()
    part = [p for p in players if all(w in plain(p["name"]) for w in words)]
    if part:
        return part
    names = {plain(p["name"]): p for p in players}
    close = difflib.get_close_matches(q, names, n=8, cutoff=0.6)
    return [names[n] for n in close]


def fmt(stat, v):
    if v is None:
        return "-"
    if stat in ("AVG", "OBP", "SLG", "OPS"):
        return f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"
    if stat in ("ERA", "FIP", "WHIP"):
        return f"{v:.2f}"
    if stat in ("HR/600", "SB/600", "wRC+"):
        return f"{v:.0f}"
    return f"{v:.1f}"


def show(p, meta):
    hitter = p["type"] == "H"
    line = "=" * W
    print(line)
    print(f" {p['name'].upper()}  |  {p['pos']}  |  {p['org']}")
    print(f" {p['team']} ({p['level']})  |  Age {p['age']} in {meta['season']}  |  B/T {p['bt']}  |  Born {p['born']}")
    print(line)
    rank = f"#{p['rank']}" if p.get("rank") else "outside top 100"
    eta = f"{p['eta']} (already debuted)" if p["debuted"] else str(p["eta"])
    print(f" Prospect rank: {rank:<16} ETA: {eta:<22} MLB chance: {p['p_mlb']:.0%}")
    print(f" Prospect value: {p['score']:.1f}  (MLB chance x expected 6-year WAR of {p['war6_mean']:.1f})")

    print(f"\n EXPECTED MLB STATS - average of first 3 MLB seasons"
          + (" (rates per 600 PA)" if hitter else ""))
    print(f"   {'':<10}{'Floor':>10}{'Expected':>11}{'Ceiling':>10}")
    order = (["AVG", "OBP", "SLG", "OPS", "HR/600", "SB/600", "K%", "BB%", "wRC+", "WAR/yr"] if hitter
             else ["ERA", "FIP", "WHIP", "K/9", "BB/9", "WAR/yr"])
    for stat in order:
        lo, mid, hi = p["proj"][stat]
        print(f"   {stat:<10}{fmt(stat, lo):>10}{fmt(stat, mid):>11}{fmt(stat, hi):>10}")

    print("\n CAREER OUTLOOK - WAR over first 6 MLB seasons (if the player reaches MLB)")
    for name, war, label in zip(("Floor", "Expected", "Ceiling"), p["war6"], p["outlook"]):
        print(f"   {name:<10}{war:>6.1f} WAR   {label}")

    if p["comps"]:
        print("\n SIMILAR MiLB PROFILES (players who reached MLB)")
        for c in p["comps"]:
            print(f"   {c['name']:<24} as of {c['at']:<14} -> debut {c['debut']}, "
                  f"{c['war6']:.1f} WAR in 6 yrs, {c['yrs1_3']} yrs 1-3")

    print("\n MiLB CAREER")
    if hitter:
        print(f"   {'Year':<5}{'Lvl':<5}{'Team':<26}{'Age':>4}{'PA':>5}{'AVG':>6}{'OBP':>6}{'SLG':>6}"
              f"{'HR':>4}{'SB':>4}{'K%':>6}{'BB%':>6}")
        for y, lvl, team, age, pa, avg, obp, slg, hr, sb, k, bb in p["milb"]:
            print(f"   {y:<5}{lvl:<5}{team[:25]:<26}{age:>4}{pa:>5}{fmt('AVG', avg):>6}{fmt('OBP', obp):>6}"
                  f"{fmt('SLG', slg):>6}{hr:>4}{sb:>4}{k:>6.1f}{bb:>6.1f}")
    else:
        print(f"   {'Year':<5}{'Lvl':<5}{'Team':<26}{'Age':>4}{'IP':>7}{'G':>4}{'GS':>4}{'ERA':>6}"
              f"{'WHIP':>6}{'K/9':>6}{'BB/9':>6}")
        for y, lvl, team, age, ip, g, gs, era, whip, k9, bb9 in p["milb"]:
            print(f"   {y:<5}{lvl:<5}{team[:25]:<26}{age:>4}{ip:>7}{g:>4}{gs:>4}{fmt('ERA', era):>6}"
                  f"{fmt('WHIP', whip):>6}{fmt('K/9', k9):>6}{fmt('BB/9', bb9):>6}")
    print(line)


def show_top(players, n):
    top = sorted([p for p in players if p.get("rank")], key=lambda p: p["rank"])[:n]
    print(f" {'#':>3}  {'Player':<24}{'Pos':<6}{'Org':<24}{'Lvl':<5}{'Age':>4}{'ETA':>6}{'MLB%':>6}  6-yr WAR (floor/exp/ceil)")
    for p in top:
        lo, mid, hi = p["war6"]
        print(f" {p['rank']:>3}  {p['name'][:23]:<24}{p['pos']:<6}{p['org'][:23]:<24}{p['level']:<5}"
              f"{p['age']:>4}{p['eta']:>6}{p['p_mlb']:>6.0%}  {lo:>5.1f} / {mid:>4.1f} / {hi:>4.1f}")


def main():
    ap = argparse.ArgumentParser(description="MLB projection for a minor league player")
    ap.add_argument("name", nargs="*", help="player name (full or partial)")
    ap.add_argument("--top", type=int, help="print the top N prospects")
    args = ap.parse_args()
    players, meta = load()

    if args.top:
        return show_top(players, args.top)
    query = " ".join(args.name) or input("Player name: ")
    hits = find(players, query)
    if not hits:
        sys.exit(f"No current minor leaguer found matching '{query}'.")
    hits.sort(key=lambda p: -p["score"])
    exact = all(plain(p["name"]) == plain(query) for p in hits)
    if len(hits) > 1 and not exact:
        hits = hits[:15]
        print(f"Players matching '{query}' (best prospect first):")
        for i, p in enumerate(hits, 1):
            print(f"  {i:>2}. {p['name']:<26} {p['pos']:<5} {p['org']:<24} {p['level']:<4} age {p['age']}")
        pick = input("Pick a number (Enter = 1): ").strip() if sys.stdin.isatty() else ""
        hits = [hits[int(pick) - 1] if pick.isdigit() and 0 < int(pick) <= len(hits) else hits[0]]
        print()
    for p in hits:  # several only when two players share the exact same name
        show(p, meta)


if __name__ == "__main__":
    main()

"""Download data from the public MLB Stats API (statsapi.mlb.com) with a local cache."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://statsapi.mlb.com/api/v1"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# sportId -> level code. 15 = short-season A (existed until 2020).
LEVELS = {1: "MLB", 11: "AAA", 12: "AA", 13: "A+", 14: "A", 15: "A-", 16: "ROK"}
MILB_SPORTS = [11, 12, 13, 14, 15, 16]
FIRST_SEASON = 2005

_session = requests.Session()
_session.headers["User-Agent"] = "predictive-milb-to-mlb/1.0"


def get_json(path, params=None, cache_name=None, refresh=False):
    """GET an API path, caching the JSON response on disk."""
    cache = RAW / cache_name if cache_name else None
    if cache and cache.exists() and not refresh:
        return json.loads(cache.read_text())
    for attempt in range(5):
        try:
            r = _session.get(f"{API}/{path}", params=params, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    return data


def season_stats(season, sport_id, group, stat_type="season", refresh=False):
    """All player season lines for one level/season/group (hitting|pitching)."""
    params = {"stats": stat_type, "group": group, "season": season,
              "sportId": sport_id, "playerPool": "All", "limit": 10000}
    name = f"{stat_type}_{group}_{sport_id}_{season}.json"
    data = get_json("stats", params, name, refresh)
    return data["stats"][0]["splits"] if data.get("stats") else []


def teams(season, sport_id, refresh=False):
    data = get_json("teams", {"sportId": sport_id, "season": season},
                    f"teams_{sport_id}_{season}.json", refresh)
    return data.get("teams", [])


def draft(year, refresh=False):
    """Every pick of one June amateur draft: player id, overall pick, bonus."""
    data = get_json(f"draft/{year}", None, f"draft_{year}.json", refresh)
    picks = []
    for rnd in data.get("drafts", {}).get("rounds", []):
        for p in rnd.get("picks", []):
            pid = (p.get("person") or {}).get("id")
            if pid and p.get("pickNumber"):
                picks.append(dict(pid=pid, draft_year=int(year), draft_pick=int(p["pickNumber"]),
                                  bonus=float(p.get("signingBonus") or "nan")))
    return picks


def people(ids, refresh=False):
    """Bio info (birth date, position, current team...) for many players."""
    ids = sorted(set(int(i) for i in ids))
    out = []
    chunks = [ids[i:i + 150] for i in range(0, len(ids), 150)]

    def one(chunk):
        name = f"people/{chunk[0]}_{chunk[-1]}_{len(chunk)}.json"
        data = get_json("people", {"personIds": ",".join(map(str, chunk)),
                                   "hydrate": "currentTeam"}, name, refresh)
        return data.get("people", [])

    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(one, chunks):
            out.extend(res)
    return out


def download_all(last_season, refresh_last=True):
    """Fetch every season/level/group needed by the model."""
    jobs = []
    for season in range(FIRST_SEASON, last_season + 1):
        refresh = refresh_last and season == last_season
        for group in ("hitting", "pitching"):
            jobs.append((season, 1, group, "sabermetrics", refresh))
            for sport in [1] + MILB_SPORTS:
                if sport == 15 and season > 2020:
                    continue
                if sport != 1 and season == 2020:  # no MiLB season in 2020
                    continue
                jobs.append((season, sport, group, "season", refresh))

    def run(job):
        season, sport, group, stype, refresh = job
        season_stats(season, sport, group, stype, refresh)
        return job

    done = 0
    with ThreadPoolExecutor(8) as ex:
        for _ in ex.map(run, jobs):
            done += 1
            if done % 50 == 0:
                print(f"  downloaded {done}/{len(jobs)} stat tables")
    for sport in (1, 11, 12, 13, 14, 16):
        teams(last_season, sport, refresh_last)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda y: draft(y, refresh_last and y == last_season),
                    range(FIRST_SEASON - 8, last_season + 1)))
    print(f"  downloaded {len(jobs)} stat tables")

# Predictive MiLB → MLB

Predicts how today's minor leaguers will perform in the majors, using only the public **MLB Stats API**.

The model studies MLB players from 2005 to 2026, looking at their minor-league numbers and what happened after they reached MLB. It uses that history to project every current minor leaguer:

- **ETA**: the year they're expected to debut, plus their **chance of reaching MLB**
- **Expected MLB stats** for their first 3 seasons (AVG/OBP/SLG, HR, SB, K%, BB%, wRC+ for hitters; ERA, FIP, WHIP, K/9, BB/9 for pitchers)
- **Floor and ceiling**: the low and high end of their outcomes, for those first-3-season stats and for career value (WAR over their first 6 MLB seasons)

**Website:** https://luca-pfi.github.io/PREDICTIVE-MODEL-MILB-TO-MLB/. It shows the top 100 prospects, and at the bottom you can look up any minor leaguer.

## Quick start

```bash
pip install -r requirements.txt
python build.py                    # download data, train, write predictions (~2 min, first download ~5 min)
python predict.py "Eli Willits"    # full projection in the console
python predict.py --top 25         # top 25 prospect list
```

Partial names and names without accents work too (`python predict.py made` finds Jesús Made). If several players match, you choose one from a list.

```
 ELI WILLITS  |  SS  |  Washington Nationals
 Prospect rank: #1     ETA: 2028     MLB chance: 99%

 EXPECTED MLB STATS - average of first 3 MLB seasons (rates per 600 PA)
                  Floor   Expected   Ceiling
   OPS             .627       .729      .832
   HR/600            12         17        23
   wRC+              84         95       136
 CAREER OUTLOOK - WAR over first 6 MLB seasons
   Floor       -0.6 WAR   Depth / up-and-down
   Expected    12.9 WAR   Above-average regular
   Ceiling     26.3 WAR   Superstar
 SIMILAR MiLB PROFILES: Bryce Harper, J.P. Crawford, Carlos Correa
```

## How it works

1. **Data**: every MLB, AAA, AA, High-A, Low-A, short-season and rookie-ball season since 2005, plus MLB WAR, wRC+ and FIP, and the draft (pick number and bonus).
2. **League-adjusted profile**: each MiLB season is compared to its own league (K% vs the league's K%, OPS vs the league's OPS, age vs the league's average age). The last 3 seasons are blended, with recent seasons counting more.
3. **Learn the jump**: gradient-boosted tree models learn how those profiles turned into MLB results: the chance of debuting within 7 years, years until debut, first-3-season stats and 6-season WAR.
4. **Floor and ceiling**: quantile models predict a low (~10th percentile), expected (median) and high (~90th percentile) outcome. Floor and ceiling assume the player reaches MLB.
5. **Ranking**: prospect value = chance of reaching MLB × expected 6-year WAR.

## Accuracy (5-fold cross-validation, grouped by player)

| | Hitters | Pitchers |
|---|---|---|
| Chance-of-MLB model (AUC) | 0.90 | 0.84 |
| ETA error | ±0.9 yr | ±0.9 yr |
| Real 6-yr WAR between floor and ceiling | 79% | 79% |

Predictions are well calibrated: of players given about an 80% chance to reach MLB, 78% did. Exact career value is still hard to predict from stats alone. The ranges are wide for a reason.

## Files

| File | What it does |
|---|---|
| `build.py` | Runs the whole pipeline |
| `predict.py` | Console lookup for any player |
| `src/fetch.py` | MLB Stats API downloads (cached in `data/raw/`) |
| `src/features.py` | League adjustments, MiLB snapshots, MLB outcomes |
| `src/model.py` | Model training, cross-validation, predictions |
| `src/export.py` | Stat lines, comparable players, website data |
| `docs/` | The website (GitHub Pages) and `docs/data/*.json` predictions |


**Automatic update:** every year in early October (5 October), once the minor-league season is over, it downloads the new season, retrains the models and updates the website. 

## Limitations

- Stats only: no scouting grades, injuries or tools data. Draft position is the only "scouting" signal.
- A player has to have played in 2026 to be projected. Players who missed the whole season (for example after Tommy John surgery) are not included.
- Rookie-eligible players who already debuted in 2026 are included, marked "debuted".

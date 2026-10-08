# Combine tracking to NFL performance: which movement measures hold up

*This report tests which movement measures from NFL Combine sensor tracking actually predict regular-season performance, and finds that most are too noisy or only repeat the stopwatch.*

## Summary

We tested receivers (separation on routes), offensive linemen (pressure allowed) and defensive linemen (get-off and pressure) for movement that predicts NFL results beyond the stopwatch and body size. We found none. Three results are still useful to a team:

1. **Most first-step measures are too noisy to trust.** Measures that need a detected start frame correlate only 0.04 to 0.39 between two attempts at the same drill. Measures that do not need one repeat better (0.33 to 0.68). A measure that does not repeat cannot predict anything.
2. **Some measures only repeat the 40-yard dash.** For defensive linemen, peak speed and best distance in one second are 66% explained by stopwatch and size numbers.
3. **One lead did not survive checking.** Offensive-line first-step distance pointed the right way against pressure allowed (-0.21) and kept its sign across three draft classes, but it faded under other definitions of the outcome.

## Why this question

Combine evaluation still leans on the stopwatch. Sensor tracking records how each athlete moves, which could show quickness that a single time hides. Before any such measure is used to pick or develop players, a team needs to know two things: whether the measure repeats from one attempt to the next, and whether it predicts NFL results beyond the 40 and body size. Our work asks both, position by position, with the outcomes a coach recognises: separation for receivers, and pressure allowed and pressure created for linemen.

## Data and method

Combine tracking covers 510 players (draft classes 2023 to 2025) at 10 Hz. Game tracking covers the 2023 to 2025 seasons but only for 497 draft-class players, with no opponents or ball. `player_play` supplies the outcomes, with penalty-voided plays removed. No data dictionary was available, so we inferred that `player_get_off` is a time in seconds (pressured plays average 0.88, others 1.02).

Speed and heading are recomputed from smoothed positions, because the file's `dis` and `s` columns disagree in scale.

**Outcomes.** Receivers: separation at pass release minus the average for the same route and coverage type. Offensive line: pressure allowed minus a league-wide expectation from drop-back time and rushers faced. Defensive line: get-off time and pressure rate, each minus the position average. Minimum plays: 30 routes, 100 pass blocks, 50 get-offs and 100 pass rushes.

**Statistics.** Partial rank correlations with weight, height and share of snaps at the main position removed. 95% bootstrap intervals. Each family of tests is adjusted together for multiple comparisons. We counted 56 movement tests across four families.

## Receivers (93 WRs)

No movement feature relates to separation over expected by more than 0.18 in size. Weight does (-0.41), which reflects role and body type, not movement. Adding movement features to a baseline of 40 time, vertical, broad jump and weight moves cross-validated R-squared from 0.086 to 0.091, inside a fold spread of about 0.2.

We matched each route to its closest drill (hitch to curl and comeback, out to speed out, slant to slant, in to dagger, post and corner to post-corner, go to go). None of 19 tests survives adjustment; the largest is out routes against time to re-accelerate in the speed-out drill (-0.29, adjusted p 0.58). Smooth routes show near-zero speed loss because their cuts are rounded, so this kind of measure suits only sharp breaks.

## Which measures repeat

For athletes with two attempts at the same drill, we correlated the first with the second.

![Bar chart. Offensive-line and defensive-line first-to-second-attempt correlations for seven measures. The four measures that need a detected start frame score 0.04 to 0.39. Peak speed scores 0.53 and 0.68, best distance in any 1 s scores 0.41 and 0.66, best speed gain in any 0.5 s scores 0.40 and 0.33.](figures/fig1_reliability.png)

**Figure 1.** Repeat correlation by measure (data: `figures/fig1_reliability.csv`). Offensive-line values are in blue, defensive-line in orange. Taking the best window anywhere in the attempt, instead of one after a detected start, lifted offensive-line first-step distance from 0.04 to 0.41 and defensive-line from 0.39 to 0.66. Any correlation with an outcome is capped by how well the measure repeats, so our first-pass null results (20 tests, all between -0.14 and +0.13) say little about whether first-step quickness matters.

## Overlap with the stopwatch

![Bar chart of cross-validated R-squared of each movement feature on stopwatch and size numbers. Defensive-line peak speed and best distance in 1 s score 0.66, defensive-line best speed gain 0.40, offensive-line features 0.09 to 0.15, and receiver cut features between -0.06 and -0.01.](figures/fig2_overlap.png)

**Figure 2.** Share of each feature predicted by 40 time, 10-yard split, vertical, broad jump, weight and height (data: `figures/fig2_overlap.csv`). Defensive-line peak speed and best distance largely repeat the 40. The receiver cut features overlap with nothing, but we did not measure their repeat reliability, so that may be noise.

## Linemen: the tests and the one lead

![Dot plot of 51 partial correlations with 95% intervals in three groups: receivers by route, linemen first pass and linemen second pass. Almost all intervals cross zero. Three miss zero and are marked in orange: out routes against time to re-accelerate, offensive-line best distance against pressure allowed, and defensive-line best speed gain against pressure rate.](figures/fig3_all_tests.png)

**Figure 3.** All 51 tests with partial correlations (data: `figures/fig3_all_tests.csv`). Three intervals miss zero before adjustment; after adjustment their p-values are 0.58, 0.25 and 0.16. The five receiver tests that used raw correlation are in the text above.

The second pass used three reliable measures against four outcomes. Two leads stood out, so we fixed a replication rule in advance: the sign must match in all three draft classes.

| Lead | 2023 | 2024 | 2025 | Same sign in all three |
| --- | --- | --- | --- | --- |
| Defensive-line speed gain vs pressure rate | +0.34 (n=28) | +0.28 (n=27) | -0.09 (n=26) | No |
| Offensive-line best distance vs pressure allowed | -0.19 (n=31) | -0.22 (n=36) | -0.42 (n=22) | Yes |

The defensive-line lead failed. The offensive-line lead held its sign, but the classes are not independent of the pooled figure, and every class interval is wide.

![Dot plot of the offensive-line lead under twelve checks. Original -0.21; pressure probability -0.05; raw pressure rate -0.13; sack rate +0.10; time to pressure +0.12; team average removed -0.15; mirror drill alone -0.02; pass-rush-drop drill alone -0.02; wave-slide drill alone -0.18; file's own speed -0.20; at least 200 plays -0.25; at least 300 plays -0.23.](figures/fig4_ol_robustness.png)

**Figure 4.** The offensive-line lead under other definitions (data: `figures/fig4_ol_robustness.csv`). Blue intervals miss zero. It is absent in two of three drills, weak under other outcomes, and falls to -0.15 with team averages removed. A permutation test gives p = 0.042 for this one test and 0.16 for the largest of six. We do not claim it.

## Defensive backs

Only the 497 draft-class players are tracked, so a DB's separation allowed can be measured only when the receiver he faces is also a draft-class player. We built 1,176 man-coverage pairs from 6,398 candidates, keeping those starting within 4 yards at the snap. For these, the DB-to-receiver distance equals the file's nearest-defender separation 82% of the time (14% for dropped pairs), so most are real matchups.

Only 60 DBs appear, 22 with 20 or more pairs. Outcome reliability is 0.43 to 0.51 and shifts with row order. With that sample only a correlation near 0.4 could be detected, so we did not test movement features. The question cannot be answered from this data.

## What teams can use

- **Check repeatability before using any tracking-based quickness measure,** and do not rely on one attempt.
- **Prefer measures that do not depend on a start frame,** averaged over attempts.
- **Check what a new measure adds beyond the 40 and body size.**
- **Adjust outcomes for route, coverage, drop-back time and rushers faced** before comparing players.

## What we would do next

Measure repeat reliability on every drill, not two, and use only measures above about 0.6. Test one narrow question with an outcome that has more between-player spread, using more draft classes as they arrive. With opposing players included in the tracking, repeat the defensive-back analysis.

## Limits

Samples are small (50 to 100 players per test), so effects below about 0.25 would be hard to see. The best lead was chosen after the results. Only players who stayed long enough to meet play minimums are included, so busts are largely missing. Repeat reliability was measured for two drills only. Get-off and pass-rush definitions are inferred. Receiver-to-drill matching was our judgement, fixed before the results.

**Request to organizers.** Including opposing players, or the assigned receiver, in the game tracking would allow coverage analysis, which this data cannot support.

## Appendix: reproducing

Scripts, in order: `load.py`, `features.py`, `outcomes.py`, `test_wr.py`, `test_wr_matched.py`, `trench_features.py`, `trench_outcomes.py`, `test_trench.py`, `overlap.py`, `reliability.py`, `replication.py`, `robustness_ol.py`, `db_pairs.py`, `charts.py`. Each output is saved in `results/`. A Kaggle notebook runs all of them in one pass.

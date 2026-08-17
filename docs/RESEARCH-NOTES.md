# What 1,500 Days of Data Are Actually Worth

*Quant research notes from a third-place finish at Algothon 2026 (Susquehanna × UNSW FinTech Society)*

Last week our submission, **abc123**, placed third on the Final-round technical leaderboard of Algothon 2026, scored on 500 days of market data nobody had seen. I want to write down what actually earned that result — and it isn't a clever model. It's a handful of quantitative research decisions, most of which were made **before touching any data**.

## 1. Read the objective function like a contract

The competition scored `mean(P&L) × SR²/(SR²+1)`. Most teams read that once and started building models. We differentiated it.

A few lines of calculus settle three of the four axes a strategy could compete on. The elasticity gap between mean and volatility is exactly 1 at every Sharpe — a percent of extra mean always outranks a percent of volatility reduction, and past SR ≈ 8 a dollar of mean is a dollar of score while the entire remaining value of risk-adjustment work is bounded by μ/(1+SR²) — a couple of percent. Commissions capped out around $104 a day against a book earning ~$1,100 a day: turnover was nearly free. Position caps fixed the optimal size at the cap.

So before any backtest existed, the research agenda had collapsed to a single question: **direction**. Everything else was already decided by the rules. I now believe this is the highest-ROI hour available in any structured problem: differentiate the objective before you optimize it.

## 2. Count your effective sample, not your nominal one

1,500 days × 51 assets sounds like 76,500 observations. It is one realized market history. A common factor carried ~22% of the correlation trace, and the effective number of independent cross-sectional units came out near **5**, against a nominal 50. Every threshold we set downstream assumed the small number, not the big one.

This is the least glamorous habit in quant research and the one that decides everything after it: your panel is almost never as big as it looks.

## 3. Price your own luck before admiring your results

We measured the standard error of a single scored block at roughly **130–140 points**. Then we priced selection bias directly: picking the best of a few hundred zero-skill variants is worth about **+166 points** of score, for free. That number became a hard threshold, fixed before candidates were compared.

Seven near-relatives of our engine posted higher nominal scores during development. The best margin was **+1.14 points** — against a selection price of +166. Two orders of magnitude inside the noise. We submitted none of them. The version that shipped was the one with the simplest structure and the fewest unverified parts — explicitly *not* the highest score on our own leaderboard, because our own leaderboard was, statistically, a random-number generator at that margin.

## 4. The out-of-sample verdict

The hidden-window result landed at **1,085.17** against a public-data replay mean of 1,144.89 — a gap of about **0.4 of one block's standard error**. The transfer we had declined to promise arrived inside the tolerance our own noise analysis had priced. And the gap to second place? 3.7 points — three *hundredths* of a standard error. At that resolution, ranks 2 and 3 are the same number.

That's the strange comfort of doing the noise arithmetic honestly: the leaderboard stops being a scoreboard and becomes a measurement, with error bars you computed yourself.

## What I'd tell anyone starting in QR

- Differentiate the objective before optimizing it. The rules usually decide more than your model does.
- Your effective sample is a fraction of your nominal one. Count it.
- Selection luck has a price. Compute it, fix it as a threshold in advance, and be prepared to discover that your best "improvement" doesn't clear it.
- When nothing is demonstrably better, ship the simplest thing with the fewest unverified parts.

The full 69-page research record — every constant derived, every claim graded by evidence class, every number regenerated from retained records at build time — was the real product of this competition. The medal is a lagging indicator.

Thanks to **Susquehanna International Group** and the **UNSW FinTech Society** for a competition where the scoring rule itself was worth a week of study.



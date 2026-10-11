# Methodology

## Unit of analysis and defined mechanism

One row represents one synthetic case assignment. For a selected number of cases, `n`, the generator makes `n` independent categorical draws over Court A, Court B, Court C, and Court D. Each case appears once; there is no filtering by resolution, decision, or outcome.

The null allocation probabilities are fixed in advance:

```text
p = (0.40, 0.30, 0.20, 0.10)
```

If `X_i` is the count for court `i`, the count vector follows `Multinomial(n, p)` under the null. The counts sum to `n` and are dependent on one another even though the individual assignments are independent. Expected counts are `E_i = n * p_i`; they need not be equal or integers.

The illustrative anomaly scenario draws from `q = (0.58, 0.21, 0.14, 0.07)`. This changes the data-generating mechanism while leaving the comparison rule `p` unchanged. Counts are sampled without forcing category totals. A scenario describes how data were generated; it does not guarantee a particular test result.

All labels and records are synthetic. The app provides no inference about an actual institution or person.

## Primary analysis: global multinomial comparison

The global null hypothesis is that assignments follow all four defined probabilities. The discrepancy statistic is Pearson's statistic:

```text
T = sum_i (X_i - E_i)^2 / E_i
```

The app does not use an asymptotic chi-square approximation to turn this statistic into a p-value. Instead, it generates `B` count vectors from `Multinomial(n, p)` using an independent, recorded Monte Carlo seed, and computes the same statistic for every vector. This follows the general null-simulation approach described in the [SciPy Monte Carlo test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html); the project implements its own multinomial calibration to expose simulation diagnostics.

Let `b` be the number of simulated statistics at least as large as the observed statistic, including ties. Numerical comparisons allow a conservative tie tolerance of `64 * float64_epsilon * max(1, T_observed)`, subtracted from the observed statistic. The reported Monte Carlo p-value is:

```text
p_MC = (b + 1) / (B + 1)
```

The added observation prevents a zero p-value and gives a finite-simulation correction. The smallest reportable value is `1 / (B + 1)`. A result at that floor means the simulation did not observe a statistic as extreme as the one analyzed; it is not evidence that the true tail probability is zero.

This calibration uses the discrete null distribution and remains suitable for sparse expected counts under the defined model. It cannot repair incorrect probabilities, dependent assignments, incomplete records, or a changing eligibility set.

The approximate Monte Carlo standard error is `sqrt(r * (1 - r) / B)`, where `r = b / B`. It describes uncertainty caused by finite simulation, not uncertainty about the truth of the model. This plug-in estimate is zero when `b = 0` or `b = B`; those zeros do **not** mean there is no simulation uncertainty.

An exact 95% Clopper-Pearson binomial interval accompanies it. This interval describes uncertainty in the underlying null exceedance probability estimated by the simulations, not a confidence interval for a causal effect or the probability that the null is true. The Monte Carlo correction and this interval serve different purposes. Increase `B` for finer resolution, and retain the same seed to reproduce the calculation. If a conclusion depends on a value very close to a threshold, simulation precision matters. The app and CLI default to `B = 20,000` with Monte Carlo seed `271828`.

The fixed demo probabilities are all positive. The statistical library also handles zero-probability categories: a zero observed count contributes nothing to the statistic; a positive count is flagged as a null-support violation with an infinite statistic. In that boundary case, the model's exact tail is zero while the corrected simulation result stays at its resolution floor. The flag is necessary to distinguish the two.

## Planned secondary analysis: exact binomial enrichment

Court A is the preselected target, and enrichment is the preselected direction. Under the null, its marginal count has distribution `Binomial(n, 0.40)`. For an observed target count `k`, the exact upper-tail p-value is:

```text
P(X_A >= k | n, p_A = 0.40)
```

The tail includes the observed count and is calculated from the binomial distribution directly, without a normal approximation, using [`scipy.stats.binomtest` with `alternative="greater"`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html). In particular, a zero observed count has an upper-tail p-value of one. This is a test for excess assignments; a deficit is not evidence against its one-sided alternative.

Extremely small nonzero tails can underflow floating-point arithmetic. Such results carry an explicit underflow flag and a finite natural-log p-value. The interface displays a power of ten; JSON exports use `null` for the unrepresentable numeric p-value and retain its log value. Holm adjustment is also computed in log space. An event outside a zero-probability category's null support has a genuinely zero tail and is distinguished from numerical underflow.

Preselection is part of the inferential claim. Selecting a target or test direction after viewing the results changes the question. In that situation, use the exploratory family below and disclose the selection; do not relabel a selected court as a preplanned target.

## Exploratory comparisons and multiplicity

The exploratory table computes one exact upper-tail binomial p-value for each of the four courts against that court's own null probability. It applies Holm's step-down adjustment to the entire four-test family.

For ordered raw p-values `p_(1) <= ... <= p_(m)` with `m = 4`, adjusted values are the cumulative maximum of `(m - j + 1) * p_(j)`, capped at one and returned to the original court order. Holm controls the family-wise error rate without requiring the tests to be independent, which matters because the category counts share a total.

These values are labeled exploratory. Their adjustment covers the four displayed enrichment tests. It does not cover repeated searches over seeds, sample sizes, time windows, alternative directions, or other unreported analyses. The global test is the primary model check; the preselected target test is a planned secondary analysis. This ordering clarifies their roles without imposing a gatekeeping procedure or creating a separately corrected family for a combined claim such as "either test is significant."

## Magnitude and interpretation

For each court, the app reports the observed count and share, expected count and share, the share difference in percentage points, and the observed-to-expected count ratio. These describe deviation size and direction independently of a significance threshold.

The overall total variation distance is half the sum of absolute share differences:

```text
TV = 0.5 * sum_i abs(X_i / n - p_i)
```

It lies between zero and one, with zero indicating matching observed and expected shares. It measures the fraction of probability mass that would need to be redistributed to align the two distributions; it does not identify how many individual assignments were improperly made. Unlike the Pearson statistic, TV directly summarizes the magnitude of the share discrepancy.

Every p-value is conditional on the defined mechanism and the corresponding hypothesis. It describes the extremeness of the observed statistic under that model. It does not measure the probability of fraud, guilt, manipulation, or the null hypothesis itself. Statistical incompatibility also does not identify a cause. A non-small p-value is not proof that the allocation rule holds.

This interpretation follows the [American Statistical Association's principles on p-values](https://www.amstat.org/asa/files/pdfs/P-ValueStatement.pdf): assess assumptions and effect size, report the analysis transparently, and avoid basing conclusions on a threshold alone.

Case outcomes, selected subsets of resolved cases, estimated comparison probabilities, and contingency-table questions require different data definitions and analyses. They are outside this demo. A Fisher or hypergeometric test is not used because this simulation's null is a known-probability multinomial assignment process, not a comparison conditional on contingency-table margins.

## Reproducibility and boundaries

Reproduce an analysis with its generation seed, Monte Carlo seed, number of assignments, scenario, reference probabilities, and Monte Carlo sample count. Record software versions as random-number and numerical implementations can change across versions. Generation and calibration use separate random streams so changing simulation precision does not change the synthetic assignments.

The implementation validates counts and probabilities rather than silently repairing invalid inputs. Probability sums must equal one within absolute tolerance `1e-12`; normalization only removes rounding noise within that tolerance. Validation covers distribution tails against independent calculations, weighted probabilities, zero counts, multiplicity, invalid inputs, and seeded reproducibility. The analysis requires no network calls once dependencies are installed.

The model is deliberately small. Applying similar methods elsewhere requires a defensible allocation mechanism and an appropriate sampling frame before choosing the test. This demonstration supplies both by construction; it is not a ready-made conclusion about any real assignment process.

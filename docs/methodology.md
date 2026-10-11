# Methodology

## Designed data and the classroom question

The public lesson contains a fixed, authored fixture of 60 invented cases involving one fictional company. Its assignments are literal teaching data, not a random sample, a seed-selected discovery, or records from an actual institution. Court A and the question about receiving extra assignments are chosen as part of the lesson design. This is not a claim that the displayed pattern was discovered in a prospectively collected dataset.

The full counts, in Court A through Court L order, are:

```text
(10, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4)
```

Each case appears once. The lesson does not filter on case outcome or resolution, measure a company's success in court, or infer benefit from an assignment. All company, case, and court labels are fictional.

The beginner question is: **if each case independently had a one-in-twelve chance of going to Court A, how often would 60 cases produce ten or more assignments there?** The lesson moves from the visible records to the rule, simulated allocations, the exact distribution, the inclusive tail, and a conditional interpretation. The interface uses Spanish; these technical notes use English.

## The stated assignment model

Under the model, every case independently goes to one of twelve eligible courts with equal, constant probability:

```text
p = (1/12, ..., 1/12)
(X_A, ..., X_L) ~ Multinomial(60, p)
X_A ~ Binomial(60, 1/12)
E[X_A] = 60 * (1/12) = 5
```

Expected counts are averages across repeated allocations, not quotas. Giving a case to Court A does not reduce that court's chance on the next independent draw. The twelve court counts share a total of 60 and are therefore dependent, even though individual assignments are independent. The probabilities are stated inputs, not estimates fitted to the fixture.

Real use would require a complete sampling frame selected independently of court destination, a defensible eligibility rule, and evidence that assignments have the assumed probabilities and dependence structure. Related cases, workload balancing, changing eligibility, or incomplete records can invalidate this model. A count of resolved cases is a different population. A Fisher or hypergeometric test is not the calculation here: the model specifies assignment probabilities rather than conditioning on contingency-table margins.

## Intuition from repeated allocations

Each simulation batch contains 1,000 **complete allocations of 60 cases**. For every allocation, 60 independent draws over all twelve courts produce one joint count vector that sums to 60. The code does not draw twelve independent binomial counts or force category totals. Court A's count is then read from each complete allocation.

Batch numbers start at one. Batch `b` uses Python's `random.Random` with seed `271828 + b - 1` and twelve-way `randrange` draws. Repeating the batch number with the recorded implementation reproduces all its allocations. The fixture has no generation seed: seeds apply only to simulations, which never replace its records.

The displayed match count is the number of simulated allocations with `X_A >= 10`. It is an observed simulation frequency, not the exact p-value or a second hypothesis test. The first reproducible batch happens to contain 26 such allocations; other batches need not. The exact probability implies an average of about 25.754 matches per 1,000 repetitions, not a requirement that a particular batch contain 26. These are counts of whole allocations, not counts of individual cases.

Repeating or selecting an attractive simulation batch cannot change the exact result. Exported batch metadata permits reproduction rather than treating an illustration as new evidence.

## Exact count probabilities and the inclusive tail

For an integer count `x` from zero through 60, the binomial probability is:

```text
P(X_A = x) = C(60, x) * (1/12)^x * (11/12)^(60-x)
```

The plotted bars represent these discrete probabilities. There is no fitted normal curve, density smoothing, or area-based approximation. The expected marker is at five and the observed marker is at ten. Colour highlights counts at least as large as the threshold, including the threshold itself.

The teaching question uses an upper tail:

```text
P(X_A >= 10) = sum from x=10 to 60 of P(X_A = x)
```

| Quantity | Value under the stated model |
| --- | ---: |
| Expected Court A count | 5 |
| Authored Court A count | 10 |
| Difference from expectation | +5 cases; twice the expected count |
| Probability of exactly 10 | 1.57071156% |
| Probability of 10 or more | 2.57542547% |
| Average number of allocations with 10 or more per 1,000 repetitions | 25.7542547 |

The existing exact upper-tail calculation uses [`scipy.stats.binomtest` with `alternative="greater"`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html). The inclusive result is the p-value for the planned Court A enrichment question. It is not the probability of exactly the observed count, a test for unusually few assignments, or a two-sided result. The lesson's exact-count and tail functions also allow direct numerical reproduction.

The chart may hide negligible tails to remain readable. The existing binomial-view helper retains the original bar heights, includes the threshold and expected marker, and limits total omitted probability to `1e-6`. For a positive upper tail, the omitted part is also at most `1e-6` of that tail's probability. Relative checks use log probabilities when needed. The displayed p-value always includes the full tail through 60; it is never recomputed from, or renormalized to, the visible bars.

## Interpretation and hypothetical practice

The conditional statement is: **under the stated rule, about 26 in 1,000 complete allocations of 60 cases would give Court A ten or more cases**. This frequency comes from the exact model. It does not promise a particular simulation count or predict which future allocation will be unusual.

The probability is below a 5% reading reference, so this pattern would be uncommon under that rule. The authored fixture is a demonstration of that calculation, not empirical evidence that any institution used a different rule. A p-value is not the probability of fraud, guilt, misconduct, or the null hypothesis being true. Unusual outcomes do not establish a cause; ordinary outcomes do not prove randomness or fairness. This interpretation is consistent with the [American Statistical Association's principles on p-values](https://www.amstat.org/asa/files/pdfs/P-ValueStatement.pdf).

The practice slider changes a **hypothetical threshold** `k` between zero and 15 while retaining `n = 60` and `p_A = 1/12`. It asks for `P(X_A >= k)` through the full possible count of 60. The practice range is not the distribution's support. At `k = 0`, the upper-tail probability is one; a threshold below five is not a test for a deficit.

Practice does not edit the 60 records, change the observed Court A count of ten, or specify a new allocation across the other eleven courts. Its threshold and result are stored separately in the manifest. No global or multiple-testing conclusion is inferred from moving the slider.

## Reproduction and exports

`lesson.py` owns the fixture, its counts, exact calculations, simulation batches, and export metadata. `lesson_records()` returns the teaching records; `lesson_counts()` returns a count tuple in `COURTS` order. `exact_count_probability(k)`, `exact_tail_probability(k)`, and `target_result(k=10)` reproduce the Court A calculations. `simulate_batch(batch_number)` reproduces a selected 1-based simulation batch.

```bash
python scripts/export_lesson.py --output artifacts/lesson --batch 1 --practice-threshold 12
```

The script writes `lesson.csv` and `lesson_manifest.json`. The CSV always contains the authored fixture. The manifest identifies its construction as `authored_teaching_fixture` and separates `dataset`, `model`, `observed_analysis`, `practice`, and `simulation`. It records the dataset counts and CSV SHA-256 checksum, software versions, and, when selected, the simulation's batch number, seed, complete allocation vectors, Court A counts, match count, and frequency. Omitting `--batch` exports without a simulation; omitting `--practice-threshold` avoids adding a hypothetical exercise.

The browser uses the filenames `constructed_lesson_assignments.csv` and `lesson_run_manifest.json`. Its manifest records the current practice threshold and the last simulated batch, if one has been run. Practice reuses that batch for its optional simulated frequency; it does not generate replacement allocations or change the batch's original threshold-ten summary.

Reproduction depends on the fixture version, model, batch number and seed, algorithm, and recorded software versions. The statistical calculations require no external services after dependency installation. All exported records and simulated allocations are invented; no uploaded dataset or external personal information is used.

## Optional advanced analyses

These library extensions are outside the beginner Streamlit lesson. They use the **current twelve-court fixture**, not the earlier four-court generator:

```python
from court_assignment_audit.lesson import COURTS, TARGET_PROBABILITY, lesson_counts
from court_assignment_audit.statistics import (
    exploratory_court_tests,
    global_multinomial_test,
)

counts = lesson_counts()
probabilities = (TARGET_PROBABILITY,) * len(COURTS)
global_result = global_multinomial_test(
    counts, probabilities, simulations=20_000, seed=271828
)
exploratory_results = exploratory_court_tests(counts, probabilities)
```

**Global model check.** Pearson's discrepancy, `sum((observed - expected)^2 / expected)`, compares the entire twelve-court allocation with its known probabilities. Calibration simulates complete multinomial vectors rather than using an asymptotic chi-square approximation. If `b` of `B` simulated discrepancies are at least as large as observed, including ties, the reported result is `(b + 1) / (B + 1)`. The finite correction prevents a zero simulation p-value; zero exceedances do not imply impossibility. Comparisons use the documented conservative floating-point tie tolerance in `statistics.py`.

The approximate Monte Carlo standard error is `sqrt(r * (1-r) / B)`, where `r = b/B`. An exact 95% Clopper-Pearson interval describes uncertainty in the underlying null exceedance probability. Neither is uncertainty about guilt, causation, or whether the model is true. At `b = 0` or `b = B`, the plug-in standard error is zero but the interval still reflects simulation uncertainty. A threshold-based reading should remain cautious when this interval touches or crosses the threshold. This calibration is separate from the 1,000-allocation classroom illustration and from the exact Court A tail.

**Exploratory enrichment.** One upper-tail binomial test per court forms a complete **twelve-test** family. Holm's step-down procedure orders the raw p-values, applies multipliers `12, 11, ..., 1`, takes cumulative maxima, caps at one, and returns them to court order. It controls the family-wise error rate without requiring independent court counts. The family includes courts that do not look unusual. Adjustment does not cover searching over datasets, targets, directions, filters, or other unreported analyses.

The global question, planned Court A question, and exploratory family are distinct. They can produce different readings, and no gatekeeping rule or combined verdict is asserted. Holm does not adjust the global and planned results as a joint family. Generic library results retain effect sizes and log probabilities for very small tails; numerical underflow is flagged rather than interpreted as mathematical zero. Zero-probability support violations are separately identified.

The previous four-court weighted generator and CLI remain a separate programmatic example. Their probabilities, generated records, seeds, and manifests do not describe or reproduce this lesson. For the classroom fixture, use `lesson.py` and `scripts/export_lesson.py`.

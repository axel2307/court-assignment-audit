# Court Assignment Audit

A small Streamlit portfolio project for exploring statistical anomalies in a **fully synthetic assignment process**. It combines reproducible simulation, statistical testing, effect sizes, and transparent reporting in one reviewable workflow.

Each synthetic case is independently assigned to one of four generic courts. The app compares observed assignments with an explicitly defined allocation rule. It does not analyze case outcomes, estimate the probability of misconduct, or provide conclusive legal evidence.

[Open the live demo](https://court-assignment-audit.streamlit.app/).

## Start with the story

The fictional registry gives Court A a 40% chance of receiving each new case, Court B 30%, Court C 20%, and Court D 10%. Random draws will not meet those shares exactly, so the question is whether the observed differences are unusual under that rule.

The app starts with one question chosen in advance: **did Court A receive unusually many assignments for its 40% chance?** Its observed count and expected average lead into an exact binomial chart. Each bar is the probability of one possible Court A count; the coloured bars mark counts at least as large as the observed count. The total probability of such counts explains the one-sided p-value.

The whole-allocation comparison follows separately and remains the primary model check. Compare counts across all four courts, or expand twelve complete simulated allocations for extra context. Technical details and the four-court Holm exploration are expandable. **How to read this** explains the assumptions; **Data & run log** preserves the synthetic CSV and reproducibility metadata. This teaching order changes neither the tests nor their roles.

## Run the demo

Use Python 3.12-3.13. From the project directory:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1 --server.port 3062
```

The requirements file installs the local analysis package as well as the pinned dependencies. The command above keeps the local demo at `http://127.0.0.1:3062`; the shared Streamlit configuration leaves network binding to the hosting environment.

Use the sidebar to compare the null and illustrative anomaly scenarios, change the number of assignments, or try another dataset seed. The scenario describes the generating mechanism, not a guaranteed conclusion.

To generate a reproducible synthetic example from the command line:

```bash
python -m court_assignment_audit.cli --scenario null --seed 20261011 --output-dir artifacts/demo
```

After installing dependencies, run the test suite with external Python socket connections blocked, then lint the project. Loopback is allowed for Windows' internal asynchronous event-loop communication; the tests use no external services.

```bash
python scripts/run_checks.py
ruff check .
```

## Deploy on Streamlit Community Cloud

Select the repository, the `main` branch, and `app.py` as the entrypoint. Choose Python **3.12** in Advanced settings, matching a tested CI runtime. No secrets or external data connections are needed. Community Cloud installs `requirements.txt`, including the local package needed by the app and its exported software-version metadata.

See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## The two scenarios

| Court | Defined null probability | Illustrative anomaly probability |
| --- | ---: | ---: |
| Court A | 40% | 58% |
| Court B | 30% | 21% |
| Court C | 20% | 14% |
| Court D | 10% | 7% |

The null scenario samples from the defined allocation rule. The anomaly scenario deliberately increases Court A's assignment probability. **Both are invented simulations**, and both are tested against the same null probabilities. Their realized counts remain random: the generator does not force a particular count or p-value.

The generator seed reproduces the assignments. A separate Monte Carlo seed controls calibration of the global test. Keep the seeds, sample size, scenario, simulation count, and software versions with a result when reproducing it.

## What the results mean

| Question | Reading |
| --- | --- |
| Did Court A receive unusually many cases? | A planned, exact upper-tail binomial test checks Court A against its 40% probability. The court and direction are chosen in advance; a shortfall is not tested. |
| Does the whole allocation fit the stated rule? | The primary global comparison uses Pearson's discrepancy, calibrated with complete multinomial simulations under the known probabilities. |
| Which courts show an excess when all four are examined? | The optional exploration applies Holm adjustment to the complete family of four enrichment tests. It is a separate question and need not agree with either result above. |

The Court A chart uses discrete binomial probabilities, not a fitted bell curve. Observed and expected markers locate this run against the model. The reported tail probability uses the full binomial distribution, summing from the observed count through the sample size even when the chart displays a narrower range; visible bars are never rescaled to total one. When readable, an accompanying frequency such as "40 in 100" or "1 in X" describes complete allocations of the same number of cases under the stated rule that would give Court A at least the observed count. It does not describe the probability of fraud. Very small probabilities retain scientific or logarithmic notation rather than being rounded to zero.

The planned Court A reading uses its exact upper-tail p-value below **5%**, without a Monte Carlo interval. The separate global headline uses the same reference but accounts for simulation uncertainty: it says unusual only when the corrected p-value and the 95% interval's upper endpoint are both below 5%, and no clear departure only when the lower endpoint is above 5%. An interval touching or crossing 5% receives a neutral reading. With one assignment, the global headline instead notes that a stable allocation pattern cannot be established.

Observed shares, percentage-point gaps, and observed-to-expected ratios show the size and direction of differences. Total variation summarizes the overall share discrepancy. The optional twelve examples use the same sample size and null probabilities, with fixed seed `271828`; they are the first joint multinomial draws, not examples selected to resemble or contrast with the observed data. They provide context, not an additional test or confidence band.

A small p-value indicates incompatibility with the stated null model under the test's assumptions. It is **not** the probability that the null is true, that an assignment was manipulated, or that any person is responsible. A large p-value does not establish that the process is unbiased. Null simulations can produce small p-values, and anomaly simulations can fail to produce them.

See [the methodology](docs/methodology.md) for the formulas, Monte Carlo uncertainty, multiple-testing scope, and assumptions.

## Scope and limitations

This is a teaching and portfolio application built entirely from generated records and generic labels. It has no data upload flow: every analysis starts with the explicitly defined synthetic mechanism.

The model assumes independent assignments, a fixed set of eligible courts, constant known probabilities, and complete assignment records. Real allocation systems may have eligibility restrictions, changing workloads, rotations, or dependent assignments that require a different model. Resolved-case outcomes are a different population and cannot substitute for assignment counts in this analysis.

The app evaluates one specified model; it does not establish causes. Trying many seeds, sample sizes, scenarios, or targets and reporting only a favorable result creates selection bias beyond the adjustment shown in the app.

## Project structure

```text
app.py                       Streamlit interface
src/court_assignment_audit/  Synthetic generation and statistical analysis
tests/                       Deterministic, offline statistical and input checks
docs/methodology.md          Model, interpretation, and limitations
```

The project intentionally keeps the analysis local and focused: no accounts, database, external model API, or production deployment is required.

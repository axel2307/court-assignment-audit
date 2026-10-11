# Court Assignment Audit

A small Streamlit portfolio project for exploring statistical anomalies in a **fully synthetic assignment process**. It combines reproducible simulation, statistical testing, effect sizes, and transparent reporting in one reviewable workflow.

Each synthetic case is independently assigned to one of four generic courts. The app compares observed assignments with an explicitly defined allocation rule. It does not analyze case outcomes, estimate the probability of misconduct, or provide conclusive legal evidence.

[Open the live demo](https://court-assignment-audit.streamlit.app/).

## Start with the story

The fictional registry gives Court A a 40% chance of receiving each new case, Court B 30%, Court C 20%, and Court D 10%. Random draws will not meet those shares exactly, so the question is whether the observed differences are unusual under that rule.

The app leads with a plain-language summary of the whole allocation. Read the observed-versus-expected counts next, then compare this run with twelve complete simulated allocations of the same size. The largest share gap describes the size of a difference; it is not a separate test of the court with that gap.

The planned Court A question appears separately below. P-values, the four-court Holm exploration, and calculation details are in expandable sections. **How to read this** explains the assumptions; **Data & run log** preserves the synthetic CSV and reproducibility metadata.

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
| Does the whole allocation fit the stated rule? | The primary global comparison uses Pearson's discrepancy, calibrated with complete multinomial simulations under the known probabilities. |
| Did Court A receive unusually many cases? | A planned, exact upper-tail binomial test checks Court A against its 40% probability. The court and direction are chosen in advance; a shortfall is not tested. |
| Which courts show an excess when all four are examined? | The optional exploration applies Holm adjustment to the complete family of four enrichment tests. It is a separate question and need not agree with either result above. |

The global headline uses **5% as an explicit reading guide**. It says the allocation is unusual only when the corrected Monte Carlo p-value and the upper end of its 95% simulation interval are both below 5%. It says there is no clear departure only when the interval's lower end is above 5%. An interval that touches or crosses 5% receives a neutral reading because simulation uncertainty affects that comparison. A single assignment instead receives a reminder that it cannot establish a stable allocation pattern. The planned Court A reading uses its exact upper-tail p-value below 5%, without a Monte Carlo interval.

Observed shares, percentage-point gaps, and observed-to-expected ratios show the size and direction of differences. Total variation summarizes the overall share discrepancy. The twelve visual examples use the same sample size and null probabilities, with fixed seed `271828`; they are the first joint multinomial draws, not examples selected to resemble or contrast with the observed data. They provide context, not an additional test or confidence band.

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

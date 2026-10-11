# Court Assignment Audit

An interactive statistics lesson built with Streamlit, Python, and **entirely invented data**. The public lesson uses Spanish for a classroom audience; the code and documentation use English for a data-analysis portfolio.

[Open the live lesson](https://court-assignment-audit.streamlit.app/).

## The teaching example

A fictional company appears in **60 cases**, of which **10 go to Court A**. The stated assignment rule gives each of **12 courts** the same independent chance of receiving each case. Court A would therefore receive **5 cases on average**, not a quota of exactly five.

The 60 records are a **fixed, deliberately authored teaching fixture**, not a random sample or observations from a real institution. Court A and the question about extra assignments are part of the lesson's design. No private data, real company names, or judicial outcomes are included.

The lesson builds the explanation in order:

1. Read the fictional story and see all 60 case tokens.
2. Understand the one-in-twelve rule and what an expected count means.
3. Simulate 1,000 complete allocations of 60 cases under that rule.
4. Compare the simulation with exact binomial probabilities.
5. Highlight **10 or more**, including ten, and interpret that tail probability.
6. Try hypothetical counts without changing the original records.

The exact chance of **10 or more** assignments to Court A is **2.5754%** under the stated rule: about **26 in 1,000 complete allocations of 60 cases**. The chance of **exactly 10** is different: **1.5707%**. Neither number is the probability of fraud, misconduct, or the rule being true. An ordinary result would not prove that a process was random or fair.

## Run locally

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

The requirements file installs the local package and pinned dependencies. The command uses `http://127.0.0.1:3062`; shared Streamlit configuration leaves network binding to the hosting environment. No secrets, accounts, external data connections, or uploads are needed by the lesson.

## Reproduce the lesson

The fixture never changes. Simulation batch `b`, numbered from one, uses seed `271828 + b - 1`. Repeating a batch number reproduces its 1,000 complete allocations. The first batch contains 26 allocations with at least ten cases in Court A; later batches can differ. The exact binomial probability does not depend on simulation results.

Each simulation click advances the batch within the current Streamlit session. Moving the practice slider preserves that batch.

```bash
python scripts/export_lesson.py --output artifacts/lesson --batch 1
```

This writes `lesson.csv` and `lesson_manifest.json`. The manifest distinguishes the authored fixture from simulated allocations, records the model and exact results, and includes a CSV checksum, software versions, and the selected simulation batch. Omit `--batch` to export without a simulation. Add `--practice-threshold 12` to record a hypothetical cutoff separately; the CSV still contains the same 60 records and ten Court A assignments.

The browser downloads the same fixture as `constructed_lesson_assignments.csv` and its current session record as `lesson_run_manifest.json`. To reproduce that session's metadata from the CLI, pass its batch number, if any, and the displayed practice threshold.

The app's practice slider covers counts 0-15. It changes a hypothetical upper-tail question, not the fixture, the sample size, or the assignment probabilities. Every tail calculation continues through the full possible count of 60.

## Validation

```bash
python scripts/run_checks.py
ruff check .
```

CI runs on Python 3.12 and 3.13. The test runner blocks external Python socket connections while allowing loopback for Windows' internal asynchronous event-loop communication. Tests use no external services. Statistical checks cover independently calculated tails, inclusive boundaries, probability mass, complete simulated allocations, reproducibility, invalid inputs, and the distinction between observed data and practice thresholds.

## Scope and extensions

The model assumes independent assignments, twelve eligible courts, constant known probabilities, and a complete set of cases selected without using their destination. Workload rules, changing eligibility, related cases, or selected resolved-case subsets may require a different model. The authored example illustrates a calculation; it is not empirical evidence about an assignment process.

[Methodology](docs/methodology.md) explains the exact calculation, simulation, visual safeguards, and optional global and Holm analyses for the twelve-court fixture. Those extensions answer different questions and are kept outside the beginner lesson. There is no combined verdict.

The earlier four-court weighted generator remains a separate programmatic example in `synthetic.py`, `reporting.py`, and `cli.py`. Its CLI does **not** reproduce this classroom lesson.

## Streamlit Community Cloud

Use repository branch `main`, entrypoint `app.py`, and Python **3.12** in Advanced settings. Community Cloud installs `requirements.txt`, including the local package. No secrets are required. See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Project structure

```text
app.py                               Spanish classroom interface
src/court_assignment_audit/lesson.py  Authored fixture, simulations, and lesson exports
src/court_assignment_audit/           Exact statistics and chart helpers
scripts/export_lesson.py              Reproducible lesson CSV and manifest
scripts/run_checks.py                 Tests with external Python sockets blocked
tests/                               Statistical, reproducibility, and interface checks
docs/methodology.md                   Model, interpretation, and optional extensions
```

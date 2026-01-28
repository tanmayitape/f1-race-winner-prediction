# Predicting Formula 1 Race Winners Using Pre-Race Machine Learning (2017–2024)

**Flagship Project | Imbalanced Classification | Time-aware Validation | Interpretable ML**

<p align="center">
  <img src="assets/cover.png" width="100%" alt="F1 Winner Prediction Cover"/>
</p>

<p align="center">
  <a href="#project-summary">Summary</a> ·
  <a href="#the-core-question">Core Question</a> ·
  <a href="#dataset">Dataset</a> ·
  <a href="#approach">Approach</a> ·
  <a href="#results">Results</a> ·
  <a href="#what-drives-winning">What Drives Winning</a> ·
  <a href="#how-to-run">How to Run</a>
</p>

---

## Project Summary
Formula 1 looks chaotic on Sunday, but the grid is not random. This project tests a simple, strict question:

**Can we predict the race winner *before lights-out* using only pre-race information?**

To answer it, I built a controlled modelling pipeline using **only features available before the race starts** (no pit strategy, weather shifts, safety cars, or telemetry). I trained and compared:

- Logistic Regression (baseline)
- Random Forest (non-linear ensemble)
- XGBoost (gradient boosting, strong under imbalance)

Evaluation uses a **time-aware split** to reflect real forecasting conditions:
- **Train:** 2017–2022  
- **Test:** 2023–2024

This avoids leakage and tests whether the model generalises to the newest competitive landscape.

---

## The Core Question
### Can a Formula 1 race winner be predicted pre-race, using public data only?

I treat each **driver–race** as one observation and predict a binary target:
- `isWinner = 1` if the driver won the race
- `isWinner = 0` otherwise

This creates a **severely imbalanced** problem:
- ~20 drivers per race
- **exactly 1 winner**
- accuracy alone is misleading

So the primary lens is winner-class performance: **Precision, Recall, and F1 for Class 1 (Winner)**.

---

## Dataset
- **Seasons:** 2017–2024 (8 seasons)
- **Rows:** 3,379 driver-race records
- **Target:** `isWinner` (one winner per race)

### Pre-race features used (examples)
- **Grid position**
- **Driver season points**
- **Constructor season points**
- **Constructor season wins**
- Historical/seasonal performance indicators

> Why this feature set? It’s realistic. These are the kinds of signals available for race previews and pre-race forecasting.

---

## Approach

### 1) EDA: Establish the “structure” of winning
Before modelling, I validated that pre-race structure exists:
- Wins are concentrated among a small number of constructors (team dominance is real)
- **Grid position has a steep non-linear relationship** with win probability
  - P1 has ~50% win probability, then it drops sharply

This is a good sign: structure exists, so learning is possible.

### 2) Validation design (the most important part)
A random train/test split would leak information across seasons and exaggerate performance.

Instead:
- **Train on 2017–2022**
- **Test on 2023–2024**

This simulates how you’d actually forecast new races.

### 3) Model behaviour matters (not just AUC)
In F1, predicting “multiple winners” inside the same race is not useful.
So beyond metrics, I also inspected model behaviour and probability outputs.

In the notebook, predicted probabilities are also normalised within each race to make them comparable:
- `norm_win_proba = win_proba / sum(win_proba within the race)`

That allows clean ranking of drivers per race.

---

## Results

### Headline outcome
**XGBoost produced the best overall winner-prediction behaviour**, delivering the strongest balance between precision and recall on the winner class.

### Model comparison (Test: 2023–2024)
| Model | Accuracy | ROC-AUC | Brier Score | Precision (Winner) | Recall (Winner) | F1 (Winner) |
|------|----------|--------:|------------:|-------------------:|----------------:|------------:|
| Logistic Regression | 0.8281 | 0.9411 | 0.115 | 0.22 | 0.93 | 0.35 |
| Random Forest | 0.8770 | 0.9598 | 0.078 | 0.28 | 0.91 | 0.43 |
| **XGBoost** | **0.9423** | 0.9376 | **0.050** | **0.45** | 0.63 | **0.52** |

**Interpretation (real talk):**
- Logistic Regression catches most true winners (high recall) but floods you with false positives (low precision).
- Random Forest ranks well (best AUC) but still over-predicts winners.
- **XGBoost is more selective**, produces **better calibrated probabilities** (lowest Brier), and gives the most realistic “one winner per race” behaviour.

> In short: **Modern F1 is predictable in structure, not fully predictable in execution.** An F1-Winner score of 0.52 is strong given the imbalance and missing in-race factors.

---

## What Drives Winning?
XGBoost feature importance quantifies what experienced fans already suspect, with actual weights:

1. **Grid Position (0.42)**  
   Qualifying is the single strongest pre-race driver-level signal.
2. **Constructor Season Points (0.28)**  
   Team strength creates the platform for winning.
3. **Constructor Season Wins (0.18)**  
   Consistent operational ability and race-winning capability matters.
4. **Driver Season Points (0.12)**  
   Driver form matters, but it’s constrained by car/team strength.

**Takeaway:** Cars create opportunity. Drivers convert it.

---

## Example: Applying the Model to a Real Race
To make the output concrete, the notebook includes race-level predicted win probabilities (and normalised win probabilities) so you can rank the grid **before** the race starts and compare to the actual outcome.

This makes the project usable as a **race preview tool**, not just a metrics exercise.

---

## Limitations (deliberate constraints)
This project intentionally excludes in-race variables that are unavailable pre-race:
- Weather changes
- Safety cars
- DNFs / reliability incidents
- Strategy and pit decisions
- Tyre degradation signals
- Telemetry

That limitation is the point: the goal is **pre-race forecasting under realistic information constraints**.

---

## Future Improvements
If extending this work, the biggest gains likely come from adding **pre-race accessible context**, such as:
- Forecast weather and track temperature
- Circuit-specific historical safety-car probability
- Altitude / track characteristics
- Probability calibration improvements
- Hybrid approach: pre-race prediction + limited early-race updates

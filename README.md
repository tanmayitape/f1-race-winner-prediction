# Predicting Formula 1 Race Winners Using Pre-Race Machine Learning (2017–2024)

## Project Summary
Formula 1 looks chaotic on Sunday, but the grid is not random. This project investigates a focused and realistic question:

**Can we predict the race winner *before lights-out* using only pre-race information?**

To answer this, I built a controlled machine learning pipeline using **only features available prior to race start**, excluding in-race variables such as pit strategy, safety cars, weather changes, or telemetry.

Three models were trained and compared:
- Logistic Regression (baseline)
- Random Forest (non-linear ensemble)
- XGBoost (gradient boosting, well-suited for imbalance)

Evaluation uses a **time-aware train–test split** to reflect real forecasting conditions:
- **Training:** 2017–2022  
- **Testing:** 2023–2024  

This design avoids data leakage and evaluates whether the models generalise to the most recent competitive era.

---

## The Core Question
### Can a Formula 1 race winner be predicted pre-race, using public data only?

Each **driver–race** combination is treated as one observation, with a binary target:
- `isWinner = 1` if the driver won the race  
- `isWinner = 0` otherwise  

This results in a **severely imbalanced classification problem**:
- ~20 drivers per race  
- Exactly **one winner**  

In this context, accuracy alone is misleading.  
Model performance is therefore evaluated using **winner-class Precision, Recall, and F1-score**, alongside probability-based metrics.

---

## Dataset
- **Seasons covered:** 2017–2024 (8 seasons)
- **Observations:** 3,379 driver–race records
- **Target variable:** `isWinner`

### Pre-race features used (examples)
- Grid position
- Driver season points
- Constructor season points
- Constructor season wins
- Historical and season-level performance indicators

These features were deliberately chosen to reflect **information realistically available before a race**, making the task relevant for race previews and forecasting.

---

## Tools Used

### Python (Core Analysis)
- **Pandas:** data loading, cleaning, feature engineering, and construction of the modelling dataset.
- **NumPy:** numerical operations and efficient array-based computation.

### Visualisation (EDA and Interpretation)
- **Matplotlib:** clean, publication-style plots for trends, distributions, and model evaluation.
- **Seaborn:** statistical visualisation for analysing relationships such as grid position versus win probability.

### Machine Learning
- **Scikit-learn:**
  - Logistic Regression baseline
  - Model training and evaluation utilities
  - Metrics for imbalanced classification, including Precision, Recall, F1-score, ROC-AUC, and Brier Score
- **XGBoost:**
  - Gradient-boosted tree model
  - Handles non-linear relationships and class imbalance effectively
  - Delivered the strongest overall winner-prediction behaviour

### Experiment Design and Evaluation
- **Time-aware validation (2017–2022 → 2023–2024):**
  - Prevents information leakage across seasons
  - Mirrors real-world forecasting conditions
- **Winner-focused metrics:**
  - Emphasis on winner-class performance rather than overall accuracy
  - Probability calibration assessed using Brier Score

### Development and Version Control
- **Jupyter Notebook:** interactive, reproducible analysis workflow
- **Git/GitHub:** version control, documentation, and portfolio publication

---

## Approach

### 1) Exploratory Data Analysis
EDA was used to confirm that meaningful structure exists in pre-race data:
- Race wins are highly concentrated among a small number of constructors
- **Grid position shows a steep non-linear relationship** with win probability  
  - Pole position carries close to a 50 percent win probability, followed by a sharp decline

This confirmed that the problem is learnable under realistic constraints.

### 2) Validation Design
A random split would exaggerate performance by leaking season-level information.

Instead:
- Models were trained on historical seasons only
- Performance was tested on unseen future seasons

This simulates how predictions would be made in practice.

### 3) Model Behaviour Analysis
Beyond aggregate metrics, model behaviour was examined at race level.

Predicted probabilities were **normalised within each race**:

This enables clean ranking of drivers and avoids unrealistic multiple-winner predictions.

---

## Results

### Headline Outcome
**XGBoost produced the best overall winner-prediction behaviour**, offering the strongest balance between selectivity and sensitivity for the winner class.

### Model Comparison (Test Set: 2023–2024)

| Model | Accuracy | ROC-AUC | Brier Score | Precision (Winner) | Recall (Winner) | F1 (Winner) |
|------|----------|--------:|------------:|-------------------:|----------------:|------------:|
| Logistic Regression | 0.8281 | 0.9411 | 0.115 | 0.22 | 0.93 | 0.35 |
| Random Forest | 0.8770 | 0.9598 | 0.078 | 0.28 | 0.91 | 0.43 |
| **XGBoost** | **0.9423** | 0.9376 | **0.050** | **0.45** | 0.63 | **0.52** |

### Interpretation
- Logistic Regression captures most winners but generates many false positives.
- Random Forest ranks drivers well but still over-predicts winners.
- **XGBoost is more selective**, better calibrated, and produces the most realistic pre-race predictions.

**Key insight:**  
Modern Formula 1 is **predictable in structure**, but not fully predictable in execution.  
An F1-score of 0.52 for winner prediction is strong given the imbalance and exclusion of in-race factors.

---

## What Drives Winning?
XGBoost feature importance highlights the dominant pre-race signals:

1. **Grid Position (0.42)** – qualifying performance is the strongest driver-level predictor  
2. **Constructor Season Points (0.28)** – team strength creates winning opportunity  
3. **Constructor Season Wins (0.18)** – operational race-winning capability matters  
4. **Driver Season Points (0.12)** – driver form matters, but within team constraints  

**Takeaway:** Cars create opportunity. Drivers convert it.

---

## Practical Application
The notebook includes race-level predicted probabilities and normalised rankings, allowing:
- Pre-race winner ranking
- Comparison of predicted versus actual outcomes
- Use as a structured **race preview and analysis tool**

---

## Limitations
This project intentionally excludes in-race variables that are unavailable pre-race:
- Weather changes
- Safety cars
- DNFs and reliability failures
- Pit strategy and tyre degradation
- Telemetry and live race data

These constraints are deliberate to preserve real-world forecasting validity.

---

## Future Improvements
Potential extensions include:
- Forecast weather and track temperature
- Circuit-specific historical safety-car probabilities
- Track altitude and layout characteristics
- Improved probability calibration
- Hybrid approaches combining pre-race prediction with early-race updates




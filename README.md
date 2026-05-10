# credit-risk-pd

## Overview
An end-to-end Probability of Default (PD) credit scoring model built on the German Credit dataset, 
implementing industry standard techniques used in retail banking.

## Motivation
This project was built to demonstrate practical credit risk modeling skills including 
data exploration, scorecard development and cost-sensitive modeling.

## Project Structure
- `01_EDA_and_data_loading.ipynb` — Exploratory data analysis
- `02_preprocessing.ipynb` — Data preprocessing and WoE transformation
- `03_modeling.ipynb` — Model building, validation and scorecard construction

## Methodology
### Data
- German Credit dataset (1000 observations, 20 features)
- Target variable: binary good/bad credit risk
- Class imbalance: 70% good, 30% bad

### Preprocessing
- Log transformation of credit amount to handle skewness
- Weight of Evidence (WoE) encoding of all variables
- Information Value (IV) based feature selection
- Variables with IV < 0.02 dropped

### Modeling
Two cost-sensitive logistic regression approaches were implemented and compared:
- **Model 1** — Cost matrix embedded in loss function via class weights {good: 1, bad: 5}
- **Model 2** — Standard logistic regression with cost-based threshold adjustment (threshold = 0.167)

### Validation
- 5-fold stratified cross validation
- Cost-weighted misclassification rate as primary metric
- Model 2 outperformed Model 1 (mean cost: 0.438 vs 0.454) with lower variance (std: 0.046 vs 0.068)

### Scorecard
- Logistic regression converted to points-based scorecard using PDO-odds scaling
- Base score: 600, PDO: 20 (every 20 point increase halves default odds)
- Score range on test set: 474 to 729

## Key Findings
- `checking_status` and `credit_history` are the strongest predictors of default
- Counterintuitive finding: customers with "all paid" credit history have higher default rates than those with "critical accounts" — consistent with selection bias in credit data
- Cross validation revealed Model 2 superior despite Model 1 appearing better on single test split — highlighting importance of robust evaluation on small datasets

## Limitations
- German Credit dataset is small (1000 observations) and dated
- Stress testing would require longitudinal data spanning multiple economic cycles
- Some variables (gender, foreign worker status) raise model fairness concerns
- A production model would require independent model validation

## Technologies
- Python 3.11
- pandas, numpy, scikit-learn
- optbinning (WoE/IV transformation and scorecard)
- matplotlib, seaborn

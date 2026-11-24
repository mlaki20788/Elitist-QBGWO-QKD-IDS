# Elitist Q-BGWO + RBF-SVM for DDoS Detection in Hybrid QKD Networks

Official implementation and experimental scripts for classically simulated **Elitist Quantum-Inspired Binary Grey Wolf Optimizer (Elitist Q-BGWO)** paired with **RBF-SVM** for passive post-flow DDoS detection on classical QKD control planes.

## Repository Contents
- `step6_final_journal_experiments.py`: Main benchmark script running 10 repeated stratified splits (Seeds 42–51) with training-only preprocessing (`cache_size=1000` MB, population standard deviation `ddof=0`), paired Wilcoxon signed-rank tests, aggregated confusion matrices, and imbalanced ($\approx 9.91\%$) traffic evaluation.
- `requirements.txt`: Required Python dependencies (`Python 3.13.2`, `scikit-learn 1.9.1`).

## Data Sources
- **KDD Cup 99 (10% subset):** Loaded directly via `sklearn.datasets.fetch_kddcup99(percent10=True, random_state=42)` and deduplicated prior to balanced sampling.
- **CICIDS2017 Benchmark Sample:** Loaded directly from `https://raw.githubusercontent.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/main/data/CICIDS2017_sample.csv` and deduplicated prior to balanced sampling.

## Quick Start
```bash
pip install -r requirements.txt
python step6_final_journal_experiments.py

# Elitist Q-BGWO + RBF-SVM for DDoS Detection in Hybrid QKD Networks

Official implementation and experimental scripts for classically simulated **Elitist Quantum-Inspired Binary Grey Wolf Optimizer (Elitist Q-BGWO)** paired with **RBF-SVM** for passive post-flow DDoS detection on classical QKD control planes.

## Repository Contents
- `step6_final_journal_experiments.py`: Main benchmark script running 10 repeated stratified splits (Seeds 42–51) with training-only preprocessing across deduplicated KDD Cup 99 and CICIDS2017 proxy datasets, Wilcoxon signed-rank statistical tests, aggregated confusion matrices, and imbalanced ($\approx 9.91\%$) traffic evaluation.
- `requirements.txt`: Required Python dependencies.

## Quick Start
```bash
pip install -r requirements.txt
python step6_final_journal_experiments.py

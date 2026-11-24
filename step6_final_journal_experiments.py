import platform
import time
import warnings
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score
)

warnings.filterwarnings("ignore")

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
POP_SIZE = 12
MAX_ITER = 20
W1 = 0.992
W2 = 0.008

def load_kdd99_deduplicated():
    print("Loading KDD99 (10% subset) and removing duplicate records...")
    data = fetch_kddcup99(percent10=True, random_state=42)
    df = pd.DataFrame(data.data)
    df['target'] = [label.decode('utf-8') for label in data.target]

    total_raw = len(df)
    # فیلتر کردن کلاس عادی و حملات DoS
    dos_labels = {'back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'}
    df = df[df['target'].isin(dos_labels | {'normal.'})].copy()
    
    # حذف رکوردهای تکراری برای جلوگیری از نشت داده
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    print(f"KDD99 filtered rows: {len(df)} -> Deduplicated unique rows: {len(df_dedup)} (Removed {len(df) - len(df_dedup)} duplicates)")

    y_binary = (df_dedup['target'] != 'normal.').astype(int).values
    X_df = df_dedup.drop(columns=['target'])

    rng = np.random.RandomState(42)
    idx_0 = np.where(y_binary == 0)[0]
    idx_1 = np.where(y_binary == 1)[0]
    n_s = min(3000, len(idx_0), len(idx_1))
    sel_idx = np.concatenate([
        rng.choice(idx_0, n_s, replace=False),
        rng.choice(idx_1, n_s, replace=False)
    ])
    return X_df.iloc[sel_idx].reset_index(drop=True), y_binary[sel_idx]

def load_cicids2017_clean():
    print("Loading CICIDS2017 (DoS/DDoS benchmark sample)...")
    url = "https://raw.githubusercontent.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/main/data/CICIDS2017_sample.csv"
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]
    target_col = 'Label' if 'Label' in df.columns else df.columns[-1]

    X_raw = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].astype(str).str.strip()

    non_num = X_raw.select_dtypes(exclude=[np.number]).columns
    if len(non_num) > 0:
        X_raw = X_raw.drop(columns=non_num)

    X_raw = X_raw.astype(float)
    X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid = ~X_raw.isna().any(axis=1)
    X_raw = X_raw[valid]
    y_raw = y_raw[valid]

    dos_mask = y_raw.str.upper().isin(['BENIGN', 'DOS', 'DDOS'])
    X_raw = X_raw[dos_mask]
    y_raw = y_raw[dos_mask]

    # حذف ردیف‌های کاملاً تکراری در صورت وجود
    combined = X_raw.copy()
    combined['__y__'] = (y_raw.str.upper() != 'BENIGN').astype(int).values
    combined = combined.drop_duplicates().reset_index(drop=True)

    y_binary = combined['__y__'].values
    X_df = combined.drop(columns=['__y__'])

    rng = np.random.RandomState(42)
    idx_0 = np.where(y_binary == 0)[0]
    idx_1 = np.where(y_binary == 1)[0]
    n_s = min(3000, len(idx_0), len(idx_1))
    sel_idx = np.concatenate([
        rng.choice(idx_0, n_s, replace=False),
        rng.choice(idx_1, n_s, replace=False)
    ])
    return X_df.iloc[sel_idx].reset_index(drop=True), y_binary[sel_idx]

def preprocess_split_strictly_on_train(X_tr_df, X_val_df, X_te_df, is_kdd=False):
    X_tr = X_tr_df.copy()
    X_val = X_val_df.copy()
    X_te = X_te_df.copy()

    # ۱. رمزگذاری دسته‌ای منحصراً روی داده آموزش (برای KDD99)
    if is_kdd:
        cat_cols = [1, 2, 3]
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
        X_val[cat_cols] = enc.transform(X_val[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str))

    X_tr_arr = X_tr.values.astype(float)
    X_val_arr = X_val.values.astype(float)
    X_te_arr = X_te.values.astype(float)

    # ۲. حذف ستون‌های با واریانس صفر منحصراً بر اساس داده آموزش
    active_cols = np.where(np.std(X_tr_arr, axis=0) > 0)[0]
    X_tr_arr = X_tr_arr[:, active_cols]
    X_val_arr = X_val_arr[:, active_cols]
    X_te_arr = X_te_arr[:, active_cols]

    # ۳. نرمال‌سازی Min-Max منحصراً روی داده آموزش
    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_tr_arr)
    X_val_s = scaler.transform(X_val_arr)
    X_te_s = scaler.transform(X_te_arr)

    return X_tr_s, X_val_s, X_te_s, len(active_cols)

def make_fitness_evaluator(X_tr, y_tr, X_val, y_val, total_dim):
    def evaluate(mask):
        cols = np.where(mask == 1)[0]
        if len(cols) == 0:
            return 1.0, 0.0
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', cache_size=1000, random_state=42)
        clf.fit(X_tr[:, cols], y_tr)
        acc = accuracy_score(y_val, clf.predict(X_val[:, cols]))
        fit = W1 * (1.0 - acc) + W2 * (len(cols) / total_dim)
        return fit, acc
    return evaluate

def run_pso(eval_fn, dim, seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(-1, 1, (POP_SIZE, dim))
    vel = rng.uniform(-0.5, 0.5, (POP_SIZE, dim))
    pbest_pos = pos.copy()
    pbest_fit = np.full(POP_SIZE, np.inf)
    gbest_pos = np.ones(dim, dtype=int)
    gbest_cont = pos[0].copy()
    gbest_fit = np.inf

    for _ in range(MAX_ITER):
        for i in range(POP_SIZE):
            sig = 1.0 / (1.0 + np.exp(-np.clip(pos[i], -10, 10)))
            mask = (rng.rand(dim) < sig).astype(int)
            fit, _ = eval_fn(mask)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = pos[i].copy()
            if fit < gbest_fit:
                gbest_fit = fit
                gbest_pos = mask.copy()
                gbest_cont = pos[i].copy()
        r1, r2 = rng.rand(POP_SIZE, dim), rng.rand(POP_SIZE, dim)
        vel = 0.7 * vel + 1.5 * r1 * (pbest_pos - pos) + 1.5 * r2 * (gbest_cont - pos)
        pos = np.clip(pos + vel, -4, 4)
    return gbest_pos

def run_bgwo(eval_fn, dim, seed):
    rng = np.random.RandomState(seed)
    wolves = rng.randint(0, 2, (POP_SIZE, dim))
    alpha_pos, beta_pos, delta_pos = np.zeros(dim, dtype=int), np.zeros(dim, dtype=int), np.zeros(dim, dtype=int)
    alpha_fit, beta_fit, delta_fit = np.inf, np.inf, np.inf

    for t in range(MAX_ITER):
        for i in range(POP_SIZE):
            fit, _ = eval_fn(wolves[i])
            if fit < alpha_fit:
                delta_fit, delta_pos = beta_fit, beta_pos.copy()
                beta_fit, beta_pos = alpha_fit, alpha_pos.copy()
                alpha_fit, alpha_pos = fit, wolves[i].copy()
            elif fit < beta_fit:
                delta_fit, delta_pos = beta_fit, beta_pos.copy()
                beta_fit, beta_pos = fit, wolves[i].copy()
            elif fit < delta_fit:
                delta_fit, delta_pos = fit, wolves[i].copy()

        a = 2.0 * (1.0 - t / (MAX_ITER - 1))
        for i in range(POP_SIZE):
            r1, r2 = rng.rand(dim), rng.rand(dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - wolves[i])

            r1, r2 = rng.rand(dim), rng.rand(dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - wolves[i])

            r1, r2 = rng.rand(dim), rng.rand(dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - wolves[i])

            X_avg = (X1 + X2 + X3) / 3.0
            sig = 1.0 / (1.0 + np.exp(-10 * (X_avg - 0.5)))
            wolves[i] = (rng.rand(dim) < sig).astype(int)
    return alpha_pos

def run_qbgwo(eval_fn, dim, seed, elitist=True):
    rng = np.random.RandomState(seed)
    theta = np.full((POP_SIZE, dim), np.pi / 4)

    if elitist:
        alpha_pos = np.ones(dim, dtype=int)
        alpha_fit, _ = eval_fn(alpha_pos)
        beta_pos = alpha_pos.copy()
        beta_fit = alpha_fit
        delta_pos = alpha_pos.copy()
        delta_fit = alpha_fit
    else:
        alpha_pos, beta_pos, delta_pos = np.zeros(dim, dtype=int), np.zeros(dim, dtype=int), np.zeros(dim, dtype=int)
        alpha_fit, beta_fit, delta_fit = np.inf, np.inf, np.inf

    for t in range(MAX_ITER):
        prob_one = np.sin(theta) ** 2
        wolves = (rng.rand(POP_SIZE, dim) < prob_one).astype(int)

        for i in range(POP_SIZE):
            fit, _ = eval_fn(wolves[i])
            if fit < alpha_fit:
                delta_fit, delta_pos = beta_fit, beta_pos.copy()
                beta_fit, beta_pos = alpha_fit, alpha_pos.copy()
                alpha_fit, alpha_pos = fit, wolves[i].copy()
            elif fit < beta_fit:
                delta_fit, delta_pos = beta_fit, beta_pos.copy()
                beta_fit, beta_pos = fit, wolves[i].copy()
            elif fit < delta_fit:
                delta_fit, delta_pos = fit, wolves[i].copy()

        a = 2.0 * (1.0 - t / (MAX_ITER - 1))
        target_state = (alpha_pos + beta_pos + delta_pos) / 3.0
        for i in range(POP_SIZE):
            r_mean = (rng.rand(dim) + rng.rand(dim) + rng.rand(dim)) / 3.0
            delta_theta = a * 0.08 * np.pi * (target_state - (np.sin(theta[i]) ** 2)) * r_mean
            theta[i] = np.clip(theta[i] + delta_theta, 0.05, np.pi / 2 - 0.05)

    return alpha_pos

def evaluate_on_heldout_test(mask, X_tr, y_tr, X_te, y_te):
    cols = np.where(mask == 1)[0]
    if len(cols) == 0:
        cols = np.arange(X_tr.shape[1])
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', cache_size=1000, probability=True, random_state=42)
    clf.fit(X_tr[:, cols], y_tr)

    # زمان‌سنجی دقیق inference (میانگین روی ۵ تکرار پس از warm-up)
    _ = clf.predict(X_te[:10, cols])
    t_runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        preds = clf.predict(X_te[:, cols])
        t_runs.append(((time.perf_counter() - t0) / len(y_te)) * 1e6)
    infer_us = float(np.mean(t_runs))

    probs = clf.predict_proba(X_te[:, cols])[:, 1]
    acc = accuracy_score(y_te, preds) * 100.0
    prec = precision_score(y_te, preds, zero_division=0) * 100.0
    rec = recall_score(y_te, preds, zero_division=0) * 100.0
    f1 = f1_score(y_te, preds, zero_division=0) * 100.0
    ap = average_precision_score(y_te, probs) * 100.0
    cm = confusion_matrix(y_te, preds)
    return acc, prec, rec, f1, ap, len(cols), infer_us, cm, clf

def process_single_seed(seed, X_df, y, is_kdd):
    X_temp, X_te_df, y_temp, y_te = train_test_split(X_df, y, test_size=0.20, stratify=y, random_state=seed)
    X_tr_df, X_val_df, y_tr, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=seed)

    X_tr_s, X_val_s, X_te_s, dim = preprocess_split_strictly_on_train(X_tr_df, X_val_df, X_te_df, is_kdd=is_kdd)
    eval_fn = make_fitness_evaluator(X_tr_s, y_tr, X_val_s, y_val, dim)

    seed_res = {}
    seed_cm = {}

    # 1. Full-Feature SVM
    res_full = evaluate_on_heldout_test(np.ones(dim, dtype=int), X_tr_s, y_tr, X_te_s, y_te)
    seed_res["Full-Feature SVM"] = (0.0, dim, *res_full[:7])
    seed_cm["Full-Feature SVM"] = res_full[7]

    # 2. PSO + SVM
    t0 = time.perf_counter()
    m_pso = run_pso(eval_fn, dim, seed)
    s_time = time.perf_counter() - t0
    res_pso = evaluate_on_heldout_test(m_pso, X_tr_s, y_tr, X_te_s, y_te)
    seed_res["PSO + SVM"] = (s_time, dim, *res_pso[:7])
    seed_cm["PSO + SVM"] = res_pso[7]

    # 3. BGWO + SVM
    t0 = time.perf_counter()
    m_bgwo = run_bgwo(eval_fn, dim, seed)
    s_time = time.perf_counter() - t0
    res_bgwo = evaluate_on_heldout_test(m_bgwo, X_tr_s, y_tr, X_te_s, y_te)
    seed_res["BGWO + SVM"] = (s_time, dim, *res_bgwo[:7])
    seed_cm["BGWO + SVM"] = res_bgwo[7]

    # 4. Standard Q-BGWO + SVM
    t0 = time.perf_counter()
    m_qstd = run_qbgwo(eval_fn, dim, seed, elitist=False)
    s_time = time.perf_counter() - t0
    res_qstd = evaluate_on_heldout_test(m_qstd, X_tr_s, y_tr, X_te_s, y_te)
    seed_res["Standard Q-BGWO + SVM"] = (s_time, dim, *res_qstd[:7])
    seed_cm["Standard Q-BGWO + SVM"] = res_qstd[7]

    # 5. Elitist Q-BGWO + SVM
    t0 = time.perf_counter()
    m_qeli = run_qbgwo(eval_fn, dim, seed, elitist=True)
    s_time = time.perf_counter() - t0
    res_qeli = evaluate_on_heldout_test(m_qeli, X_tr_s, y_tr, X_te_s, y_te)
    seed_res["Elitist Q-BGWO + SVM"] = (s_time, dim, *res_qeli[:7])
    seed_cm["Elitist Q-BGWO + SVM"] = res_qeli[7]

    # ارزیابی نامتوازن واقعی (10% Attack / 90% Benign) روی همان Held-out Test
    imb_res = None
    if not is_kdd:
        idx_norm = np.where(y_te == 0)[0]
        idx_att = np.where(y_te == 1)[0]
        rng_i = np.random.RandomState(seed)
        sel_att = rng_i.choice(idx_att, size=max(1, int(len(idx_norm) / 9)), replace=False)
        imb_idx = np.concatenate([idx_norm, sel_att])
        X_te_imb, y_te_imb = X_te_s[imb_idx], y_te[imb_idx]

        cols_eli = np.where(m_qeli == 1)[0]
        clf_eli = res_qeli[8]
        p_imb = clf_eli.predict(X_te_imb[:, cols_eli])
        pr_imb = clf_eli.predict_proba(X_te_imb[:, cols_eli])[:, 1]
        imb_res = (
            accuracy_score(y_te_imb, p_imb) * 100.0,
            precision_score(y_te_imb, p_imb, zero_division=0) * 100.0,
            recall_score(y_te_imb, p_imb, zero_division=0) * 100.0,
            f1_score(y_te_imb, p_imb, zero_division=0) * 100.0,
            average_precision_score(y_te_imb, pr_imb) * 100.0
        )

    return seed, seed_res, seed_cm, imb_res

def run_benchmark_suite(name, X_df, y, is_kdd=False):
    print("\n" + "="*90)
    print(f"BENCHMARK: {name} | Samples: {len(y)} | Initial Columns: {X_df.shape[1]}")
    print("="*90)

    methods = [
        "Full-Feature SVM",
        "PSO + SVM",
        "BGWO + SVM",
        "Standard Q-BGWO + SVM",
        "Elitist Q-BGWO + SVM"
    ]

    t_start = time.perf_counter()
    results_list = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_single_seed)(seed, X_df, y, is_kdd) for seed in SEEDS
    )
    print(f"\nCompleted all 10 seeds in {time.perf_counter() - t_start:.1f} seconds!")

    records = {m: [] for m in methods}
    cms = {m: np.zeros((2, 2), dtype=int) for m in methods}
    imb_list = []

    for _, s_res, s_cm, imb_r in results_list:
        for m in methods:
            records[m].append(s_res[m])
            cms[m] += s_cm[m]
        if imb_r is not None:
            imb_list.append(imb_r)

    print("\n--- SUMMARY RESULTS (Mean ± Std over 10 Monte Carlo Stratified Splits) ---")
    header = f"{'Method':<22} | {'Acc (%)':<13} | {'Prec (%)':<13} | {'Rec (%)':<13} | {'F1 (%)':<13} | {'AP (%)':<7} | {'Feats':<11} | {'Opt(s)':<6} | {'Inf(us)':<7}"
    print(header)
    print("-" * len(header))

    prop_accs = [r[2] for r in records["Elitist Q-BGWO + SVM"]]

    for m in methods:
        arr = np.array(records[m])
        # 0:opt_s, 1:dim, 2:acc, 3:prec, 4:rec, 5:f1, 6:ap, 7:feats, 8:inf_us
        opt_m = arr[:, 0].mean()
        dim_m = arr[:, 1].mean()
        acc_m, acc_s = arr[:, 2].mean(), arr[:, 2].std()
        pre_m, pre_s = arr[:, 3].mean(), arr[:, 3].std()
        rec_m, rec_s = arr[:, 4].mean(), arr[:, 4].std()
        f1_m, f1_s   = arr[:, 5].mean(), arr[:, 5].std()
        ap_m         = arr[:, 6].mean()
        ft_m, ft_s   = arr[:, 7].mean(), arr[:, 7].std()
        inf_m        = arr[:, 8].mean()

        print(f"{m:<22} | {acc_m:5.2f} ± {acc_s:4.2f} | {pre_m:5.2f} ± {pre_s:4.2f} | {rec_m:5.2f} ± {rec_s:4.2f} | {f1_m:5.2f} ± {f1_s:4.2f} | {ap_m:6.2f}% | {ft_m:4.1f} ± {ft_s:3.1f} | {opt_m:6.2f} | {inf_m:7.2f}")

    print("\n--- WILCOXON SIGNED-RANK TEST (Elitist Q-BGWO + SVM vs Baselines) ---")
    for m in methods[:-1]:
        base_accs = [r[2] for r in records[m]]
        try:
            stat, p_val = wilcoxon(prop_accs, base_accs)
            print(f"vs. {m:<22}: W = {stat:.2f}, p-value = {p_val:.4f}")
        except Exception as e:
            print(f"vs. {m:<22}: {e}")

    print("\n--- AGGREGATED CONFUSION MATRICES (10 Splits x 1,200 Test = 12,000 Evaluated Flows) ---")
    for m in ["Full-Feature SVM", "Elitist Q-BGWO + SVM"]:
        cm = cms[m]
        fpr = (cm[0, 1] / (cm[0, 0] + cm[0, 1])) * 100.0
        fnr = (cm[1, 0] / (cm[1, 0] + cm[1, 1])) * 100.0
        print(f"{m:<22} -> TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]} | FPR={fpr:.2f}%, FNR={fnr:.2f}%")

    if len(imb_list) > 0:
        imb_arr = np.array(imb_list)
        print("\n--- IMBALANCED REALISTIC TRAFFIC EVALUATION (10% Attack / 90% Benign on CICIDS2017) ---")
        print(f"Acc: {imb_arr[:,0].mean():.2f}±{imb_arr[:,0].std():.2f}% | Prec: {imb_arr[:,1].mean():.2f}±{imb_arr[:,1].std():.2f}% | Rec: {imb_arr[:,2].mean():.2f}±{imb_arr[:,2].std():.2f}% | F1: {imb_arr[:,3].mean():.2f}±{imb_arr[:,3].std():.2f}% | AP: {imb_arr[:,4].mean():.2f}%")

if __name__ == "__main__":
    print(f"Environment: Python {platform.python_version()} | scikit-learn {sklearn.__version__} | OS: {platform.system()} {platform.release()} | CPU: {platform.processor()}")
    X_kdd, y_kdd = load_kdd99_deduplicated()
    run_benchmark_suite("KDD99 Deduplicated (DoS vs Normal)", X_kdd, y_kdd, is_kdd=True)

    X_cic, y_cic = load_cicids2017_clean()
    run_benchmark_suite("CICIDS2017 Clean (DoS/DDoS vs Benign)", X_cic, y_cic, is_kdd=False)

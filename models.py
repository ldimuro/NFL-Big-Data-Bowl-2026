import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import numpy as np

def run_logistic_regression(data: pd.DataFrame):
    data = data.dropna()
    print('DATA:', len(data))

    # --- 1. Split features and target ---
    X = data.drop(columns=['result'])
    y = data['result'].astype(int)

    accs = []
    aucs = []
    briers = []

    for seed in [1, 2, 3, 4, 5]:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed#, stratify=y
        )

        # --- 2. Logistic Regression pipeline (with scaling) ---
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('lr', LogisticRegression(max_iter=2000, solver='lbfgs'))
        ])

        # --- 3. Train and evaluate ---
        pipe.fit(X_train, y_train)
        print('finished LR seed', seed)

        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        accs.append(acc)
        aucs.append(auc)
        briers.append(brier)

    print(f"BASE | ROC-AUC: 0.5000, Accuracy: {y.sum()/len(y):.4f}, Brier: 0.2500")
    print(f'LR   | Avg ROC-AUC: {np.mean(aucs):.4f}, Avg Accuracy: {np.mean(accs):.4f}, Avg Brier: {np.mean(briers):.4f}')

    # --- 4. (Optional) Examine feature coefficients ---
    # lr = pipe.named_steps['lr']
    # coef = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    # print("\nTop coefficients (by magnitude):")
    # print(coef.round(4))


def run_random_forest(data: pd.DataFrame):
    X = data.drop(columns=['result'])
    y = data['result'].astype(int)

    raw_accs = []
    raw_aucs = []
    raw_briers = []
    cal_accs = []
    cal_aucs = []
    cal_briers = []
    all_importances = []

    for seed in [1, 2, 3, 4, 5]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

        model = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight='balanced')
        model.fit(X_train, y_train)
        print('finished RF-RAW seed', seed)

        probs_raw = model.predict_proba(X_test)[:, 1]

        raw_acc = accuracy_score(y_test, probs_raw >= 0.5)
        raw_accs.append(raw_acc)

        raw_auc = roc_auc_score(y_test, probs_raw)
        raw_aucs.append(raw_auc)

        raw_brier = brier_score_loss(y_test, probs_raw)
        raw_briers.append(raw_brier)

        # print(f"RF-RAW  | Accuracy: {raw_acc:.4f} ROC AUC: {raw_auc:.4f} Brier: {raw_brier:.4f}")

        # --- 3) Optional: fit an isotonic calibration wrapper (on train via CV) ---
        cal = CalibratedClassifierCV(model, method='isotonic', cv=5)
        cal.fit(X_train, y_train)
        print('finished RF-CAL seed', seed)

        probs_cal = cal.predict_proba(X_test)[:, 1]

        cal_acc = accuracy_score(y_test, probs_cal >= 0.5)
        cal_accs.append(cal_acc)

        cal_auc = roc_auc_score(y_test, probs_cal)
        cal_aucs.append(cal_auc)

        cal_brier = brier_score_loss(y_test, probs_cal)
        cal_briers.append(cal_brier)

        # print(f"RF-CAL  | Accuracy: {cal_acc:.4f} ROC AUC: {cal_auc:.4f} Brier: {cal_brier:.4f}")
        
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        importances = importances.sort_values(ascending=False)
        all_importances.append(importances)
        # print(importances)

    print(f"RF-RAW  | Avg ROC-AUC: {np.mean(raw_aucs):.4f}, Avg Accuracy: {np.mean(raw_accs):.4f}, Avg Brier: {np.mean(raw_briers):.4f}")
    print(f"RF-CAL  | Avg ROC-AUC: {np.mean(cal_aucs):.4f}, Avg Accuracy: {np.mean(cal_accs):.4f}, Avg Brier: {np.mean(cal_briers):.4f}")

    avg_importances = pd.concat(all_importances, axis=1).mean(axis=1)
    avg_importances = avg_importances.sort_values(ascending=False)
    print(f'\nAvg Feature Importance:\n{avg_importances}')

    # # --- 4) Calibration curves (reliability diagram) ---
    # # Use quantile bins so each bin has similar sample size
    # frac_pos_raw, mean_pred_raw = calibration_curve(y_test, probs_raw, n_bins=10, strategy='quantile')
    # frac_pos_cal, mean_pred_cal = calibration_curve(y_test, probs_cal, n_bins=10, strategy='quantile')

    # plt.figure(figsize=(6,6))
    # # Perfect calibration line
    # plt.plot([0,1], [0,1], linestyle='--', linewidth=1, label='Perfectly calibrated')

    # # Model curves
    # plt.plot(mean_pred_raw, frac_pos_raw, marker='o', linewidth=1.5, label=f'RandomForest (Brier={brier_raw:.3f})')
    # plt.plot(mean_pred_cal, frac_pos_cal, marker='o', linewidth=1.5, label=f'Isotonic Calibrated (Brier={brier_cal:.3f})')

    # plt.xlabel('Mean predicted probability')
    # plt.ylabel('Fraction of positives')
    # plt.title('Calibration Plot (Reliability Diagram)')
    # plt.legend(loc='best')
    # plt.grid(alpha=0.25)
    # plt.tight_layout()
    # plt.savefig('gi_calibration_plot.png')

    # # --- 5) (Optional) Probability histogram to inspect sharpness ---
    # plt.figure(figsize=(6,3))
    # plt.hist(probs_raw, bins=20, range=(0,1), alpha=0.6, label='Raw', density=True)
    # plt.hist(probs_cal, bins=20, range=(0,1), alpha=0.6, label='Calibrated', density=True)
    # plt.xlabel('Predicted probability')
    # plt.ylabel('Density')
    # plt.title('Predicted Probability Distribution')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('gi_calibration_distribution.png')


def run_mlp_model(data: pd.DataFrame):
    data = data.dropna()

    # Split data
    X = data.drop(columns=['result'])
    y = data['result'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Neural network model (basic)
    nn_model = Pipeline([
        ("scaler", StandardScaler()),   # NN really wants scaled inputs
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),  # two hidden layers: 64 → 32
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=42
        ))
    ])

    nn_model.fit(X_train, y_train)

    probs_raw = nn_model.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, probs_raw)

    print(f"NN   | Accuracy: {accuracy_score(y_test, probs_raw >= 0.5):.4f}  "
        f"ROC AUC: {roc_auc_score(y_test, probs_raw):.4f}  "
        f"Brier: {brier_raw:.4f}")


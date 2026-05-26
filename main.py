!pip install lightgbm --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.preprocessing import MinMaxScaler

print('Setup completo.')
print(f'LightGBM version: {lgb.__version__}')

from google.colab import files
import pandas as pd

print("Upload your file.")
uploaded = files.upload()

for fn in uploaded.keys():
  print(f'File "{fn}" caricato.')

df = pd.read_csv('SynthesizedStressData.csv')

print(f'Shape: {df.shape}')
print(f'Soggetti: {df["Subject"].nunique()}')
print(f'\nDistribuzione classi:')
print(df['metric'].value_counts().rename({0: 'No stress (0)', 1: 'Stress (1)'}))

FEATURES = ['hrrange','hrvar','hrstd','hrmin','hrmax','hrkurt',
            'edarange','edavar','edastd','edamin']

df.head(3)

np.random.seed(42)
subjects = df['Subject'].unique()
np.random.shuffle(subjects)

n_test = int(len(subjects) * 0.2)   # 40 soggetti in test
test_subjects  = subjects[:n_test]
train_subjects = subjects[n_test:]

train_df = df[df['Subject'].isin(train_subjects)]
test_df  = df[df['Subject'].isin(test_subjects)]

X_train = train_df[FEATURES]
y_train = train_df['metric']
X_test  = test_df[FEATURES]
y_test  = test_df['metric']

print(f'Train: {len(train_subjects)} soggetti, {len(X_train):,} righe')
print(f'Test:  {len(test_subjects)} soggetti, {len(X_test):,} righe')
print(f'\nClass balance train: {y_train.value_counts().to_dict()}')
print(f'Class balance test:  {y_test.value_counts().to_dict()}')





params = {
    'objective':       'binary',
    'metric':          'binary_logloss',
    'n_estimators':    500,
    'learning_rate':   0.05,
    'num_leaves':      63,
    'max_depth':       -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':    5,
    'verbose':         -1,
    'random_state':    42,
    'n_jobs':          -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)

print(f'\nBest iteration: {model.best_iteration_}')

y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]   # probabilità classe stress

acc       = accuracy_score(y_test, y_pred)
prec      = precision_score(y_test, y_pred)
rec       = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_prob)

print('=' * 45)
print('  METRICHE DI CORRETTEZZA — LightGBM')
print('=' * 45)
print(f'  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)')
print(f'  Precision : {prec:.4f}  → di quelli detti "stress", quanti lo sono?')
print(f'  Recall    : {rec:.4f}  → di quelli stress reali, quanti trovati?')
print(f'  F1 Score  : {f1:.4f}  → bilancia precision e recall')
print(f'  AUC-ROC   : {auc:.4f}  → separazione classi (1.0 = perfetto)')
print()
print('Classification report:')
print(classification_report(y_test, y_pred, target_names=['No stress','Stress']))


print('Cross-validation 5-fold (subject-level)...')

cv_scores = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_subjects_arr = np.array(train_subjects)

subject_labels = np.array([train_df[train_df['Subject']==s]['metric'].mode()[0]
                            for s in train_subjects_arr])

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_subjects_arr, subject_labels)):
    tr_subs = train_subjects_arr[tr_idx]
    va_subs = train_subjects_arr[va_idx]

    X_tr = train_df[train_df['Subject'].isin(tr_subs)][FEATURES]
    y_tr = train_df[train_df['Subject'].isin(tr_subs)]['metric']
    X_va = train_df[train_df['Subject'].isin(va_subs)][FEATURES]
    y_va = train_df[train_df['Subject'].isin(va_subs)]['metric']

    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])
    score = accuracy_score(y_va, m.predict(X_va))
    cv_scores.append(score)
    print(f'  Fold {fold+1}: {score:.4f}')

print(f'\nCV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Bekalo — LightGBM Stress Classifier', fontsize=13, fontweight='bold')

# matrix
ax = axes[0]
cm = confusion_matrix(y_test, y_pred)
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['No stress','Stress'])
ax.set_yticklabels(['No stress','Stress'])
ax.set_xlabel('Predetto'); ax.set_ylabel('Reale')
ax.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black',
                fontsize=13, fontweight='bold')

# ROC curve
ax = axes[1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, color='#2563eb', linewidth=2, label=f'AUC = {auc:.4f}')
ax.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.fill_between(fpr, tpr, alpha=0.1, color='#2563eb')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(); ax.grid(alpha=0.3)

#  Feature Importance
ax = axes[2]
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
colors = ['#f59e0b' if 'eda' in f else '#2563eb' for f in fi.index]
fi.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Feature Importance')
ax.set_xlabel('Importanza')
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#2563eb', label='HR'),
    Patch(color='#f59e0b', label='EDA')
], fontsize=9)

plt.tight_layout()
plt.savefig('bekalo_lgbm_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figura salvata: bekalo_lgbm_results.png')

# centroid of class 0 (meaning the one of no stress)
X_train_arr = X_train.values
y_train_arr = y_train.values

centroid_nostress = X_train_arr[y_train_arr == 0].mean(axis=0)

# normalize
scaler = MinMaxScaler()
scaler.fit(X_train_arr)

X_train_scaled   = scaler.transform(X_train_arr)
X_test_scaled    = scaler.transform(X_test.values)
centroid_scaled  = scaler.transform(centroid_nostress.reshape(1, -1))[0]

# euclidian distance from no stress
def compute_stress_index(X_scaled, centroid_scaled, max_dist=None):
    distances = np.linalg.norm(X_scaled - centroid_scaled, axis=1)
    if max_dist is None:
        max_dist = distances.max()
    stress_index = np.clip(distances / max_dist * 100, 0, 100)
    return stress_index, max_dist

train_distances = np.linalg.norm(X_train_scaled - centroid_scaled, axis=1)
MAX_DIST = np.percentile(train_distances, 99)  # 99° percentile, robusto agli outlier

stress_index_test, _ = compute_stress_index(X_test_scaled, centroid_scaled, MAX_DIST)

si_nostress = stress_index_test[y_test.values == 0]
si_stress   = stress_index_test[y_test.values == 1]

print('Stress Index — validazione:')
print(f'  No stress (classe 0): media = {si_nostress.mean():.1f}, mediana = {np.median(si_nostress):.1f}')
print(f'  Stress    (classe 1): media = {si_stress.mean():.1f}, mediana = {np.median(si_stress):.1f}')
print(f'  Separazione: {si_stress.mean() - si_nostress.mean():.1f} punti')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.hist(si_nostress, bins=50, alpha=0.7, color='#4ade80', label='No stress', density=True)
ax.hist(si_stress,   bins=50, alpha=0.7, color='#f87171', label='Stress',    density=True)
ax.axvline(50, color='gray', linestyle='--', linewidth=1, label='Threshold 50')
ax.set_xlabel('Stress Index (0–100)')
ax.set_ylabel('Densità')
ax.set_title('Distribuzione Stress Index per classe')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
sample_idx = np.random.choice(len(stress_index_test), 2000, replace=False)
sc = ax.scatter(
    y_prob[sample_idx],
    stress_index_test[sample_idx],
    c=y_test.values[sample_idx],
    cmap='RdYlGn_r', alpha=0.4, s=8
)
ax.set_xlabel('Probabilità stress (LightGBM)')
ax.set_ylabel('Stress Index — distanza centroide')
ax.set_title('Correlazione: probabilità vs stress index')
plt.colorbar(sc, ax=ax, label='Classe reale (0=no stress, 1=stress)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bekalo_stress_index.png', dpi=150, bbox_inches='tight')
plt.show()
plt.tight_layout()
plt.savefig('bekalo_normalization_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

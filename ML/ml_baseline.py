import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from mne.decoding import CSP

#====================================================
# 1. Data simulation

print("Initating sim 8 channel data...")

# expirment data set
n_trials = 100       # total rounds: 50 left hand, 50 right hand
n_channels = 8       # number of channel
sfreq = 250          # Cyton 
trial_duration = 3   # duration for movement imagination
n_times = sfreq * trial_duration 

# random 3D matrix, MNE standard：(Trials, Channels, Time_points)
X = np.random.randn(n_trials, n_channels, n_times)
y = np.array([0, 1] * (n_trials // 2)) # 0 left hand, 1 right hand

# Feature Injection
#X[y == 0, :4, :] *= 1.5 
#X[y == 1, 4:, :] *= 1.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("train shape:", X_train.shape)
print("test shape:", X_test.shape)

#====================================================
# 2. CSP+LDA pipeline

csp = CSP(n_components=4, reg=None, log=True, rank=None) # log for linear distribution, n_components=4: extract the four most discriminative spatial features
lda = LinearDiscriminantAnalysis()

# X_train -> CSP fetch 4 components -> LDA classification training
bci_pipeline = Pipeline([
    ('CSP_Extractor', csp),
    ('LDA_Classifier', lda)
])

print("Pipeline build complete, start model training...")

#====================================================
# 3. Training

print("Training CSP + LDA Baseline model...")
bci_pipeline.fit(X_train, y_train)
y_pred = bci_pipeline.predict(X_test)
print("Result:", y_pred[:5])

#====================================================
# 4. Plotting result

print("\n" + "="*40)
print("     ML Baseline report      ")
print("="*40)
print(classification_report(y_test, y_pred, target_names=['Left Hand', 'Right Hand']))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n5-Fold Cross Validation...")
cv_scores = cross_val_score(bci_pipeline, X, y, cv=5, n_jobs=1)
print(f"-> 5-Fold Cross-Validation Accuracy: {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores):.2%})")
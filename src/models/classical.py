"""Classical supervised baselines on hand-crafted features."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def build_svm(seed: int = 0) -> SVC:
    return SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)


def build_random_forest(seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1)

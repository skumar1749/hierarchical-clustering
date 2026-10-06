"""Select PCA components and the best Ward k for each date.
Run: Project/.venv/bin/python test.py
"""
from pathlib import Path
import json
import subprocess
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_samples
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/pca-toutes-dates"
THRESHOLDS = [0.5, 0.75, 1.0]  # Percentage points.
MAX_COMPONENTS = 20
KS = range(2, 16)


def choose_components(gains, threshold):
    # Keep at least 2 components.
    for n in range(3, MAX_COMPONENTS + 1):
        if gains[n - 1] < threshold:
            return n - 1, n
    return MAX_COMPONENTS, None


def test_clusters(X_pca):
    distances = pairwise_distances(X_pca, metric="euclidean")
    np.fill_diagonal(distances, 0)
    trials = []
    best = None
    for k in KS:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_pca)
        scores = silhouette_samples(distances, labels, metric="precomputed")
        sizes = np.bincount(labels)
        row = [k, float(scores.mean()), int(sizes.min()), int(sizes.max()),
               int((sizes == 1).sum()), float((scores < 0).mean())]
        trials.append(row)
        # Highest silhouette wins. Ties keep the smaller k.
        if best is None or row[1] > best[1]:
            best, best_labels = row, labels
    return trials, best, best_labels


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(ROOT / "Project/Datas/large_cap_x_data.pickle")
    results, details, models = [], [], {}
    clustering, best_clusters, saved_labels = [], [], []

    with threadpool_limits(limits=1):
        for date, df in sorted(data.items()):
            df = df.apply(pd.to_numeric, errors="raise").astype(float)
            if not np.isfinite(df.to_numpy()).all() or min(df.shape) < MAX_COMPONENTS:
                raise ValueError(f"Invalid data or fewer than 20 rows/features at {date}.")

            # Standardize each date separately.
            scaler = StandardScaler()
            X = scaler.fit_transform(df)
            pca = PCA(n_components=MAX_COMPONENTS, svd_solver="full").fit(X)
            gains = 100 * pca.explained_variance_ratio_
            variance = np.cumsum(gains)
            date_str = str(pd.Timestamp(date).date())

            for n in range(2, MAX_COMPONENTS + 1):
                details.append([date_str, n, float(variance[n - 1]), float(gains[n - 1])])

            for threshold in THRESHOLDS:
                n, rejected = choose_components(gains, threshold)
                results.append([
                    date_str, len(df), threshold, n,
                    float(variance[n - 1]), float(gains[n - 1]),
                    rejected, float(gains[rejected - 1]) if rejected else None,
                    "Seuil atteint" if rejected else "Limite de 20 atteinte",
                ])

                selected_pca = PCA(n_components=n, svd_solver="full").fit(X)
                X_pca = selected_pca.transform(X)
                trials, best, labels = test_clusters(X_pca)
                clustering.extend([[date_str, threshold, n, *row] for row in trials])
                best_clusters.append([date_str, threshold, n, *best])
                saved_labels.extend([
                    [date_str, threshold, str(entity), best[0], int(label)]
                    for entity, label in zip(df.index, labels)
                ])
                models[(date_str, threshold)] = {
                    "colonnes": df.columns.tolist(), "scaler": scaler, "pca": selected_pca,
                    "best_k": best[0], "silhouette": best[1],
                    "entity_ids": df.index.tolist(), "labels": labels
                }
            if len(models) // 3 % 50 == 0:
                print(f"{len(models) // 3}/{len(data)} dates processed", flush=True)

    # Save results, models and labels.
    (OUTPUT / "resultats.json").write_text(json.dumps({
        "resultats": results, "details": details,
        "clustering": clustering, "meilleurs_clusters": best_clusters
    }, allow_nan=False), encoding="utf-8")
    joblib.dump(models, OUTPUT / "modeles_pca.joblib", compress=3)
    pd.DataFrame(saved_labels, columns=[
        "date", "threshold_pp", "entity_id", "best_k", "cluster"
    ]).to_csv(OUTPUT / "best_cluster_labels.csv", index=False)
    # Format the Excel file.
    runtime = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    subprocess.run([str(runtime), str(OUTPUT / "export_excel.mjs")], check=True)
    print(f"{len(data)} dates, {len(results)} PCA selections.")
    print(f"Excel: {OUTPUT / 'resultats_pca.xlsx'}")
    print(f"Models: {OUTPUT / 'modeles_pca.joblib'}")


if __name__ == "__main__":
    main()

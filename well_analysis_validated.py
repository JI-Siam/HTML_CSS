import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import seaborn as sns

# NEW: Metrics for clustering validation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ----------------------
# 1. Load and Preprocess Data
# ----------------------

data_dir = "/content/RpSy Data/RpSy Data"  # Updated for Google Colab

well_data = {}

for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        well_id = filename[:-4]  # assuming file is named as "wellID.csv"
        df = pd.read_csv(os.path.join(data_dir, filename))
        df.columns = ["date", "RpSy", "RpSyu"]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        if len(df) >= 365:
            well_data[well_id] = df

print(f"Loaded {len(well_data)} wells with sufficient data.")

# ----------------------
# 2. Feature Extraction (Simple Stats)
# ----------------------

feature_list = []
anomaly_counts = []
well_ids = []

for well_id, df in well_data.items():
    ts = df["RpSy"].dropna()
    if len(ts) < 365:
        continue

    mean = ts.mean()
    std = ts.std()
    skew = ts.skew()
    zero_frac = (ts == 0).sum() / len(ts)
    autocorr = ts.autocorr(lag=1)

    ts_z = zscore(ts)
    anomalies = np.abs(ts_z) > 2.5
    anomaly_count = anomalies.sum()

    feature_list.append([mean, std, skew, zero_frac, autocorr])
    anomaly_counts.append(anomaly_count)
    well_ids.append(well_id)

features = pd.DataFrame(
    feature_list,
    columns=["mean", "std", "skew", "zero_frac", "autocorr"],
    index=well_ids,
)
features["anomaly_count"] = anomaly_counts

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.drop(columns=["anomaly_count"]))

# ----------------------
# 3. Clustering (KMeans)
# ----------------------

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
features["cluster"] = clusters

# NEW VALIDATION METRICS FOR KMEANS
silhouette_avg = silhouette_score(features_scaled, clusters)
ch_score = calinski_harabasz_score(features_scaled, clusters)
db_score = davies_bouldin_score(features_scaled, clusters)
print(
    f"KMeans Validation -> Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {ch_score:.1f}, Davies-Bouldin: {db_score:.3f}"
)

pca = PCA(n_components=2)
reduced = pca.fit_transform(features_scaled)

plt.figure(figsize=(8, 6))
for i in range(5):
    plt.scatter(reduced[clusters == i, 0], reduced[clusters == i, 1], label=f"Cluster {i}")
plt.legend()
plt.title("Clustering of Wells Based on Recharge Features")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="cluster", y="anomaly_count", data=features.reset_index())
plt.title("Anomaly Count per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Anomalies")
plt.grid()
plt.show()

# ----------------------
# 4. Anomaly Detection Visualization (Example Wells)
# ----------------------

example_wells = well_ids[:5]

for well_id in example_wells:
    df = well_data[well_id]
    ts = df["RpSy"].copy().dropna()
    ts_z = zscore(ts)
    anomalies = np.abs(ts_z) > 2.5

    plt.figure(figsize=(10, 4))
    plt.plot(ts.index, ts.values, label="RpSy")
    plt.scatter(ts.index[anomalies], ts[anomalies], color="red", label="Anomaly")
    plt.title(f"Anomaly Detection for Well {well_id}")
    plt.legend()
    plt.show()

# ----------------------
# 5. DTW-Based Clustering
# ----------------------

# Ensure tslearn is installed
try:
    from tslearn.metrics import dtw, cdist_dtw
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset
except ImportError:
    import subprocess, sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "tslearn", "--quiet"])
    from tslearn.metrics import dtw, cdist_dtw
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset

series_list = []
resampled_ids = []

target_len = 730

for well_id in well_ids:
    ts = well_data[well_id]["RpSy"].dropna()
    if len(ts) >= target_len:
        ts_resampled = np.interp(
            np.linspace(0, len(ts) - 1, target_len),
            np.arange(len(ts)),
            ts.values,
        )
        series_list.append(ts_resampled)
        resampled_ids.append(well_id)

X_dtw = to_time_series_dataset(series_list)

dtw_km = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=10, random_state=42)
dtw_labels = dtw_km.fit_predict(X_dtw)

# Store DTW cluster labels
for i, well_id in enumerate(resampled_ids):
    features.loc[well_id, "dtw_cluster"] = dtw_labels[i]

# NEW VALIDATION METRICS FOR DTW CLUSTERING
print(f"TimeSeriesKMeans Validation -> Inertia (sum of DTW distances): {dtw_km.inertia_:.2f}")

# Optional silhouette with precomputed DTW distances (can be slow on large datasets)
if len(resampled_ids) <= 200:  # heuristic to avoid heavy computation
    try:
        dist_matrix = cdist_dtw(X_dtw)
        silhouette_dtw = silhouette_score(dist_matrix, dtw_labels, metric="precomputed")
        print(f"TimeSeriesKMeans Silhouette (DTW distance): {silhouette_dtw:.3f}")
    except Exception as e:
        print(f"Silhouette computation for DTW clustering failed: {e}")
else:
    print("Skipping DTW silhouette due to large dataset size.")

plt.figure(figsize=(10, 6))
for i, center in enumerate(dtw_km.cluster_centers_):
    plt.plot(center.ravel(), label=f"Cluster {i}")
plt.legend()
plt.title("DTW Cluster Centers (Recharge Signature)")
plt.xlabel("Time Index (resampled)")
plt.ylabel("RpSy")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="dtw_cluster", y="anomaly_count", data=features.reset_index())
plt.title("Anomaly Count per DTW Cluster")
plt.xlabel("DTW Cluster")
plt.ylabel("Number of Anomalies")
plt.grid()
plt.show()

# ----------------------
# 6. Anomaly Seasonality & Yearly Trends
# ----------------------

anomaly_timing = []

for well_id, df in well_data.items():
    ts = df["RpSy"].dropna()
    if len(ts) < 365:
        continue

    ts_z = zscore(ts)
    anomalies = np.abs(ts_z) > 2.5
    anomaly_dates = ts.index[anomalies]

    for date in anomaly_dates:
        anomaly_timing.append({"well_id": well_id, "year": date.year, "month": date.month})

anomaly_df = pd.DataFrame(anomaly_timing)

plt.figure(figsize=(10, 5))
sns.countplot(data=anomaly_df, x="year", order=sorted(anomaly_df["year"].unique()))
plt.title("Anomalies per Year (All Wells)")
plt.xticks(rotation=45)
plt.ylabel("Anomaly Count")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=anomaly_df, x="month", order=range(1, 13))
plt.title("Anomalies per Month (All Wells)")
plt.xlabel("Month")
plt.ylabel("Anomaly Count")
plt.grid()
plt.show()

# ----------------------
# 7. Advanced Clustering (Hierarchical)
# ----------------------

agg = AgglomerativeClustering(n_clusters=5)
hier_labels = agg.fit_predict(features_scaled)
features["hierarchical_cluster"] = hier_labels

# NEW VALIDATION METRICS FOR HIERARCHICAL CLUSTERING
silhouette_avg_h = silhouette_score(features_scaled, hier_labels)
ch_score_h = calinski_harabasz_score(features_scaled, hier_labels)
db_score_h = davies_bouldin_score(features_scaled, hier_labels)
print(
    f"AgglomerativeClustering Validation -> Silhouette: {silhouette_avg_h:.3f}, Calinski-Harabasz: {ch_score_h:.1f}, Davies-Bouldin: {db_score_h:.3f}"
)

plt.figure(figsize=(8, 5))
sns.boxplot(x="hierarchical_cluster", y="anomaly_count", data=features.reset_index())
plt.title("Anomaly Count per Hierarchical Cluster")
plt.xlabel("Hierarchical Cluster")
plt.ylabel("Number of Anomalies")
plt.grid()
plt.show()

# ----------------------
# 8. Time-Series Feature Extraction (TSFresh)
# ----------------------

try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
except ImportError:
    import subprocess, sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "tsfresh", "--quiet"])
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

tsfresh_df = []

for well_id, df in well_data.items():
    ts = df["RpSy"].dropna().reset_index()
    ts = ts.rename(columns={"date": "time", "RpSy": "value"})
    ts["id"] = well_id
    tsfresh_df.append(ts)

tsfresh_df = pd.concat(tsfresh_df)

extracted_features = extract_features(
    tsfresh_df, column_id="id", column_sort="time", disable_progressbar=True
)
impute(extracted_features)

features = features.join(extracted_features, how="left")
features.fillna(0, inplace=True)

print("TSFresh features added.")
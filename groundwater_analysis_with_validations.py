import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import zscore, normaltest, shapiro
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# 1. Load and Preprocess Data with Validation
# ----------------------

def validate_data_loading(data_dir):
    """Validate data loading process"""
    print("=" * 50)
    print("DATA LOADING VALIDATION")
    print("=" * 50)
    
    validation_results = {
        'total_files': 0,
        'valid_csv_files': 0,
        'files_with_sufficient_data': 0,
        'data_quality_issues': []
    }
    
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Data directory '{data_dir}' does not exist!")
        return None, validation_results
    
    all_files = os.listdir(data_dir)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    validation_results['total_files'] = len(all_files)
    validation_results['valid_csv_files'] = len(csv_files)
    
    print(f"📁 Total files in directory: {len(all_files)}")
    print(f"📊 CSV files found: {len(csv_files)}")
    
    well_data = {}
    
    for filename in csv_files:
        try:
            well_id = filename[:-4]
            df = pd.read_csv(os.path.join(data_dir, filename))
            
            # Basic structure validation
            if df.shape[1] < 3:
                validation_results['data_quality_issues'].append(f"{filename}: Insufficient columns")
                continue
                
            df.columns = ["date", "RpSy", "RpSyu"]
            
            # Date validation
            try:
                df["date"] = pd.to_datetime(df["date"])
            except:
                validation_results['data_quality_issues'].append(f"{filename}: Invalid date format")
                continue
                
            df.set_index("date", inplace=True)
            
            # Data sufficiency validation
            if len(df) >= 365:
                well_data[well_id] = df
                validation_results['files_with_sufficient_data'] += 1
            else:
                validation_results['data_quality_issues'].append(f"{filename}: Insufficient data (<365 days)")
                
        except Exception as e:
            validation_results['data_quality_issues'].append(f"{filename}: {str(e)}")
    
    print(f"✅ Wells with sufficient data (≥365 days): {len(well_data)}")
    print(f"⚠️  Data quality issues: {len(validation_results['data_quality_issues'])}")
    
    if validation_results['data_quality_issues']:
        print("\nData Quality Issues:")
        for issue in validation_results['data_quality_issues'][:5]:  # Show first 5
            print(f"  - {issue}")
        if len(validation_results['data_quality_issues']) > 5:
            print(f"  ... and {len(validation_results['data_quality_issues']) - 5} more")
    
    return well_data, validation_results

# Update data directory path
data_dir = "/content/RpSy Data/RpSy Data"  # Updated for Google Colab

well_data, loading_validation = validate_data_loading(data_dir)

if well_data is None or len(well_data) == 0:
    print("❌ CRITICAL ERROR: No valid data loaded. Please check data directory and file formats.")
    exit()

print(f"\n✅ Successfully loaded {len(well_data)} wells with sufficient data.")

# ----------------------
# 2. Feature Extraction with Statistical Validation
# ----------------------

def validate_feature_extraction(well_data):
    """Validate feature extraction process with statistical tests"""
    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION VALIDATION")
    print("=" * 50)
    
    feature_list = []
    anomaly_counts = []
    well_ids = []
    feature_validation = {
        'wells_processed': 0,
        'wells_failed': 0,
        'feature_distributions': {},
        'statistical_tests': {}
    }
    
    for well_id, df in well_data.items():
        try:
            ts = df["RpSy"].dropna()
            if len(ts) < 365:
                feature_validation['wells_failed'] += 1
                continue

            # Calculate features
            mean = ts.mean()
            std = ts.std()
            skew = ts.skew()
            zero_frac = (ts == 0).sum() / len(ts)
            autocorr = ts.autocorr(lag=1)

            # Anomaly detection with z-score
            ts_z = zscore(ts)
            anomalies = np.abs(ts_z) > 2.5
            anomaly_count = anomalies.sum()

            # Validate features for NaN or infinite values
            features_valid = all(np.isfinite([mean, std, skew, zero_frac, autocorr]))
            if not features_valid:
                feature_validation['wells_failed'] += 1
                continue

            feature_list.append([mean, std, skew, zero_frac, autocorr])
            anomaly_counts.append(anomaly_count)
            well_ids.append(well_id)
            feature_validation['wells_processed'] += 1
            
        except Exception as e:
            feature_validation['wells_failed'] += 1
            print(f"⚠️  Failed to process well {well_id}: {str(e)}")

    # Create features dataframe
    features = pd.DataFrame(feature_list, columns=["mean", "std", "skew", "zero_frac", "autocorr"], index=well_ids)
    features["anomaly_count"] = anomaly_counts

    # Statistical validation of features
    print(f"✅ Wells processed successfully: {feature_validation['wells_processed']}")
    print(f"❌ Wells failed processing: {feature_validation['wells_failed']}")
    
    print("\nFeature Statistics:")
    print(features.describe())
    
    # Test for normality of features
    print("\nNormality Tests (p-value > 0.05 indicates normal distribution):")
    for col in features.columns:
        if col != 'anomaly_count':
            stat, p_value = normaltest(features[col].dropna())
            feature_validation['statistical_tests'][col] = p_value
            print(f"  {col}: p-value = {p_value:.4f} {'✅ Normal' if p_value > 0.05 else '❌ Not Normal'}")
    
    # Check for outliers in features
    print("\nOutlier Detection in Features (using IQR method):")
    for col in features.columns:
        Q1 = features[col].quantile(0.25)
        Q3 = features[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((features[col] < (Q1 - 1.5 * IQR)) | (features[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_pct = (outliers / len(features)) * 100
        print(f"  {col}: {outliers} outliers ({outlier_pct:.2f}%)")
    
    return features, feature_validation

features, feature_validation = validate_feature_extraction(well_data)

# Scale features for clustering
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.drop(columns=["anomaly_count"]))

# ----------------------
# 3. Clustering Validation (KMeans)
# ----------------------

def validate_clustering_performance(features_scaled, n_clusters_range=range(2, 8)):
    """Validate clustering performance using multiple metrics"""
    print("\n" + "=" * 50)
    print("CLUSTERING VALIDATION")
    print("=" * 50)
    
    clustering_metrics = {
        'n_clusters': [],
        'silhouette_score': [],
        'calinski_harabasz_score': [],
        'davies_bouldin_score': [],
        'inertia': []
    }
    
    print("Testing different numbers of clusters...")
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Calculate clustering metrics
        sil_score = silhouette_score(features_scaled, cluster_labels)
        ch_score = calinski_harabasz_score(features_scaled, cluster_labels)
        db_score = davies_bouldin_score(features_scaled, cluster_labels)
        inertia = kmeans.inertia_
        
        clustering_metrics['n_clusters'].append(n_clusters)
        clustering_metrics['silhouette_score'].append(sil_score)
        clustering_metrics['calinski_harabasz_score'].append(ch_score)
        clustering_metrics['davies_bouldin_score'].append(db_score)
        clustering_metrics['inertia'].append(inertia)
        
        print(f"k={n_clusters}: Silhouette={sil_score:.3f}, CH={ch_score:.1f}, DB={db_score:.3f}, Inertia={inertia:.1f}")
    
    # Find optimal number of clusters
    best_k_silhouette = clustering_metrics['n_clusters'][np.argmax(clustering_metrics['silhouette_score'])]
    best_k_ch = clustering_metrics['n_clusters'][np.argmax(clustering_metrics['calinski_harabasz_score'])]
    best_k_db = clustering_metrics['n_clusters'][np.argmin(clustering_metrics['davies_bouldin_score'])]
    
    print(f"\nOptimal k by different metrics:")
    print(f"  Silhouette Score: k={best_k_silhouette} (higher is better)")
    print(f"  Calinski-Harabasz: k={best_k_ch} (higher is better)")
    print(f"  Davies-Bouldin: k={best_k_db} (lower is better)")
    
    # Plot clustering metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].plot(clustering_metrics['n_clusters'], clustering_metrics['silhouette_score'], 'o-')
    axes[0,0].set_title('Silhouette Score vs Number of Clusters')
    axes[0,0].set_xlabel('Number of Clusters')
    axes[0,0].set_ylabel('Silhouette Score')
    axes[0,0].grid(True)
    
    axes[0,1].plot(clustering_metrics['n_clusters'], clustering_metrics['calinski_harabasz_score'], 'o-')
    axes[0,1].set_title('Calinski-Harabasz Score vs Number of Clusters')
    axes[0,1].set_xlabel('Number of Clusters')
    axes[0,1].set_ylabel('Calinski-Harabasz Score')
    axes[0,1].grid(True)
    
    axes[1,0].plot(clustering_metrics['n_clusters'], clustering_metrics['davies_bouldin_score'], 'o-')
    axes[1,0].set_title('Davies-Bouldin Score vs Number of Clusters')
    axes[1,0].set_xlabel('Number of Clusters')
    axes[1,0].set_ylabel('Davies-Bouldin Score')
    axes[1,0].grid(True)
    
    axes[1,1].plot(clustering_metrics['n_clusters'], clustering_metrics['inertia'], 'o-')
    axes[1,1].set_title('Elbow Method (Inertia)')
    axes[1,1].set_xlabel('Number of Clusters')
    axes[1,1].set_ylabel('Inertia')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return clustering_metrics, best_k_silhouette

clustering_metrics, optimal_k = validate_clustering_performance(features_scaled)

# Apply KMeans with optimal number of clusters (default to 5 if validation suggests otherwise)
n_clusters = 5  # You can change this to optimal_k if desired
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)
features["cluster"] = clusters

# Final clustering validation
final_silhouette = silhouette_score(features_scaled, clusters)
final_ch = calinski_harabasz_score(features_scaled, clusters)
final_db = davies_bouldin_score(features_scaled, clusters)

print(f"\nFinal Clustering Performance (k={n_clusters}):")
print(f"  Silhouette Score: {final_silhouette:.3f}")
print(f"  Calinski-Harabasz Score: {final_ch:.1f}")
print(f"  Davies-Bouldin Score: {final_db:.3f}")

# Cluster distribution validation
print(f"\nCluster Distribution:")
cluster_counts = pd.Series(clusters).value_counts().sort_index()
for i, count in cluster_counts.items():
    percentage = (count / len(clusters)) * 100
    print(f"  Cluster {i}: {count} wells ({percentage:.1f}%)")

# Check for empty or very small clusters
min_cluster_size = len(clusters) * 0.05  # 5% threshold
small_clusters = cluster_counts[cluster_counts < min_cluster_size]
if len(small_clusters) > 0:
    print(f"⚠️  Warning: {len(small_clusters)} clusters have less than 5% of data")

# PCA for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(features_scaled)

print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.3f}")

plt.figure(figsize=(10,8))
colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
for i in range(n_clusters):
    cluster_mask = clusters == i
    plt.scatter(reduced[cluster_mask, 0], reduced[cluster_mask, 1], 
               c=[colors[i]], label=f"Cluster {i} (n={sum(cluster_mask)})", alpha=0.7)

plt.legend()
plt.title(f"Clustering of Wells Based on Recharge Features\n(Silhouette Score: {final_silhouette:.3f})")
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.3f} variance)")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.3f} variance)")
plt.grid(True, alpha=0.3)
plt.show()

# Validate anomaly distribution across clusters
plt.figure(figsize=(10,6))
sns.boxplot(x="cluster", y="anomaly_count", data=features.reset_index())
plt.title("Anomaly Count Distribution per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Anomalies")
plt.grid(True, alpha=0.3)

# Add statistical annotations
for i in range(n_clusters):
    cluster_anomalies = features[features['cluster'] == i]['anomaly_count']
    median_anomalies = cluster_anomalies.median()
    plt.text(i, median_anomalies, f'Med: {median_anomalies:.0f}', 
             horizontalalignment='center', verticalalignment='bottom')

plt.show()

# Statistical test for anomaly differences between clusters
from scipy.stats import kruskal
cluster_anomaly_groups = [features[features['cluster'] == i]['anomaly_count'] for i in range(n_clusters)]
kruskal_stat, kruskal_p = kruskal(*cluster_anomaly_groups)
print(f"\nKruskal-Wallis test for anomaly differences between clusters:")
print(f"  H-statistic: {kruskal_stat:.3f}, p-value: {kruskal_p:.6f}")
print(f"  {'✅ Significant differences' if kruskal_p < 0.05 else '❌ No significant differences'} between clusters")

# ----------------------
# 4. Anomaly Detection Validation
# ----------------------

def validate_anomaly_detection(well_data, threshold=2.5):
    """Validate anomaly detection methods"""
    print("\n" + "=" * 50)
    print("ANOMALY DETECTION VALIDATION")
    print("=" * 50)
    
    anomaly_validation = {
        'total_wells': len(well_data),
        'wells_with_anomalies': 0,
        'total_anomalies': 0,
        'anomaly_rates': [],
        'method_comparison': {}
    }
    
    # Test different anomaly detection methods
    methods = {
        'Z-Score (2.5)': lambda x: np.abs(zscore(x)) > 2.5,
        'Z-Score (3.0)': lambda x: np.abs(zscore(x)) > 3.0,
        'IQR Method': lambda x: ((x < (x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)))) | 
                                 (x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))))
    }
    
    example_wells = list(well_data.keys())[:5]  # Use first 5 wells for visualization
    
    fig, axes = plt.subplots(len(example_wells), 1, figsize=(15, 4*len(example_wells)))
    if len(example_wells) == 1:
        axes = [axes]
    
    for idx, well_id in enumerate(example_wells):
        df = well_data[well_id]
        ts = df["RpSy"].copy().dropna()
        
        # Compare different anomaly detection methods
        method_results = {}
        for method_name, method_func in methods.items():
            try:
                if 'IQR' in method_name:
                    anomalies = method_func(ts)
                else:
                    anomalies = method_func(ts.values)
                method_results[method_name] = {
                    'anomalies': anomalies,
                    'count': anomalies.sum(),
                    'rate': anomalies.sum() / len(ts)
                }
            except:
                method_results[method_name] = {'anomalies': None, 'count': 0, 'rate': 0}
        
        # Plot with primary method (Z-Score 2.5)
        primary_anomalies = method_results['Z-Score (2.5)']['anomalies']
        
        axes[idx].plot(ts.index, ts.values, label="RpSy", alpha=0.7)
        if primary_anomalies is not None:
            axes[idx].scatter(ts.index[primary_anomalies], ts[primary_anomalies], 
                            color='red', label=f"Anomalies (n={primary_anomalies.sum()})", s=30)
        
        # Add statistical information
        mean_val = ts.mean()
        std_val = ts.std()
        axes[idx].axhline(y=mean_val, color='green', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
        axes[idx].axhline(y=mean_val + 2.5*std_val, color='orange', linestyle='--', alpha=0.5, label='±2.5σ')
        axes[idx].axhline(y=mean_val - 2.5*std_val, color='orange', linestyle='--', alpha=0.5)
        
        axes[idx].set_title(f"Anomaly Detection for Well {well_id}\n" + 
                           f"Methods comparison: " + 
                           ", ".join([f"{k}: {v['count']}" for k, v in method_results.items()]))
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        # Store validation metrics
        if primary_anomalies is not None and primary_anomalies.sum() > 0:
            anomaly_validation['wells_with_anomalies'] += 1
            anomaly_validation['total_anomalies'] += primary_anomalies.sum()
            anomaly_validation['anomaly_rates'].append(primary_anomalies.sum() / len(ts))
    
    plt.tight_layout()
    plt.show()
    
    # Method comparison statistics
    print("Anomaly Detection Method Comparison (across all wells):")
    print("=" * 60)
    
    method_stats = {method: {'total': 0, 'wells': 0} for method in methods.keys()}
    
    for well_id, df in well_data.items():
        ts = df["RpSy"].dropna()
        if len(ts) < 365:
            continue
            
        for method_name, method_func in methods.items():
            try:
                if 'IQR' in method_name:
                    anomalies = method_func(ts)
                else:
                    anomalies = method_func(ts.values)
                method_stats[method_name]['total'] += anomalies.sum()
                if anomalies.sum() > 0:
                    method_stats[method_name]['wells'] += 1
            except:
                pass
    
    for method_name, stats in method_stats.items():
        avg_rate = (stats['total'] / sum([len(df["RpSy"].dropna()) for df in well_data.values() 
                                          if len(df["RpSy"].dropna()) >= 365])) * 100
        print(f"{method_name:15} | Total: {stats['total']:5d} | Wells affected: {stats['wells']:3d} | Avg rate: {avg_rate:.2f}%")
    
    # Overall statistics
    if anomaly_validation['anomaly_rates']:
        avg_anomaly_rate = np.mean(anomaly_validation['anomaly_rates']) * 100
        print(f"\nOverall Anomaly Statistics:")
        print(f"  Wells with anomalies: {anomaly_validation['wells_with_anomalies']}/{anomaly_validation['total_wells']}")
        print(f"  Average anomaly rate: {avg_anomaly_rate:.2f}%")
        print(f"  Total anomalies detected: {anomaly_validation['total_anomalies']}")
    
    return anomaly_validation

anomaly_validation = validate_anomaly_detection(well_data)

print("\n" + "=" * 50)
print("INITIAL ANALYSIS VALIDATION COMPLETE")
print("=" * 50)
print("✅ Data Loading Validation: Complete")
print("✅ Feature Extraction Validation: Complete")  
print("✅ Clustering Validation: Complete")
print("✅ Anomaly Detection Validation: Complete")
print("\nProceed with DTW-based clustering and advanced analysis...")
# Complete OptiCrop Research Analytics Pipeline
# This script combines all phases from the Jupyter notebook into a single Python pipeline

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif, f_classif
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

OUTPUT_DIR = "outputs"
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

for d in [OUTPUT_DIR, CSV_DIR, JSON_DIR, CHART_DIR]:
    os.makedirs(d, exist_ok=True)

REQUIRED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ============================================================================
# LOAD DATASET
# ============================================================================

DATASET_PATH = "/kaggle/input/datasets/chitrakumari25/smart-agricultural-production-optimizing-engine/Crop_recommendation.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset Loaded Successfully")
print(df.head())

# ============================================================================
# PHASE 1: DATASET VALIDATION & DATA QUALITY ASSESSMENT
# ============================================================================

print("\n" + "="*60)
print("PHASE 1: Dataset Validation & Data Quality Assessment")
print("="*60)

# Dataset Validation
validation = []
validation.append({
    "Check": "Dataset Loaded",
    "Status": "PASS" if not df.empty else "FAIL",
    "Details": f"{len(df)} rows"
})

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
validation.append({
    "Check": "Required Columns",
    "Status": "PASS" if len(missing_cols) == 0 else "FAIL",
    "Details": "None" if len(missing_cols) == 0 else ",".join(missing_cols)
})

duplicate_cols = df.columns[df.columns.duplicated()].tolist()
validation.append({
    "Check": "Duplicate Columns",
    "Status": "PASS" if len(duplicate_cols) == 0 else "FAIL",
    "Details": "None" if len(duplicate_cols) == 0 else ",".join(duplicate_cols)
})

wrong_dtype = []
for c in REQUIRED_COLUMNS[:-1]:
    if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
        wrong_dtype.append(c)
validation.append({
    "Check": "Numeric Feature Types",
    "Status": "PASS" if len(wrong_dtype) == 0 else "FAIL",
    "Details": "All Numeric" if len(wrong_dtype) == 0 else ",".join(wrong_dtype)
})

validation_df = pd.DataFrame(validation)
print("\nValidation Report:")
print(validation_df)
validation_df.to_csv(f"{CSV_DIR}/validation_report.csv", index=False)
validation_df.to_json(f"{JSON_DIR}/validation_report.json", orient="records", indent=4)

# Data Quality Assessment
quality = {
    "Total Rows": int(len(df)),
    "Total Columns": int(len(df.columns)),
    "Missing Values": int(df.isnull().sum().sum()),
    "Duplicate Rows": int(df.duplicated().sum()),
    "Numeric Features": len(df.select_dtypes(include=np.number).columns),
    "Categorical Features": len(df.select_dtypes(include="object").columns)
}

quality_df = pd.DataFrame([quality])
print("\nData Quality Assessment:")
print(quality_df)
quality_df.to_csv(f"{CSV_DIR}/data_quality.csv", index=False)
with open(f"{JSON_DIR}/data_quality.json", "w") as f:
    json.dump(quality, f, indent=4)

# Descriptive Statistics
desc_stats = df[numeric_cols].describe()
print("\nDescriptive Statistics:")
print(desc_stats)
desc_stats.to_csv(f"{CSV_DIR}/descriptive_statistics.csv")
desc_stats.to_json(f"{JSON_DIR}/descriptive_statistics.json", orient="index", indent=4)

# Missing Value Analysis
missing_summary = df.isnull().sum().reset_index()
missing_summary.columns = ["Column", "MissingCount"]
missing_summary["MissingPercent"] = (missing_summary["MissingCount"] / len(df) * 100).round(2)
print("\nMissing Value Summary:")
print(missing_summary)
missing_summary.to_csv(f"{CSV_DIR}/missing_value_summary.csv", index=False)
missing_summary.to_json(f"{JSON_DIR}/missing_value_summary.json", orient="records", indent=4)

# Missing Value Charts
plt.figure(figsize=(10, 6))
msno.bar(df)
plt.title("Missing Value Bar Chart")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/missing_value_bar.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
msno.heatmap(df)
plt.title("Missing Value Heatmap")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/missing_value_heatmap.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
msno.matrix(df)
plt.title("Missingno Matrix")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/missingno_matrix.png", dpi=300)
plt.close()

# Dataset Summary
summary = {
    "rows": len(df),
    "columns": len(df.columns),
    "numeric_columns": list(df.select_dtypes(include=np.number).columns),
    "categorical_columns": list(df.select_dtypes(include="object").columns),
    "target": "label" if "label" in df.columns else None,
    "exports": {
        "csv": CSV_DIR,
        "json": JSON_DIR,
        "charts": CHART_DIR
    }
}
with open(f"{JSON_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=4)
print("\nSummary JSON created.")

print("\nPhase 1 Complete: validation_report, data_quality, descriptive_statistics, missing_value_summary, summary.json, Missing value charts")

# ============================================================================
# PHASE 2: DISTRIBUTION, CROP, SOIL, CLIMATE & CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("PHASE 2: Distribution, Crop, Soil, Climate & Correlation Analysis")
print("="*60)

DIST_DIR = f"{CHART_DIR}/distributions"
os.makedirs(DIST_DIR, exist_ok=True)

# Distribution Analysis
distribution_summary = []
for col in numeric_cols:
    distribution_summary.append({
        "Feature": col,
        "Mean": df[col].mean(),
        "Median": df[col].median(),
        "Std": df[col].std(),
        "Min": df[col].min(),
        "Max": df[col].max(),
        "Skewness": df[col].skew()
    })
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig(f"{DIST_DIR}/{col}_distribution.png", dpi=300)
    plt.close()

distribution_df = pd.DataFrame(distribution_summary).round(4)
print("\nDistribution Summary:")
print(distribution_df)
distribution_df.to_csv(f"{CSV_DIR}/distribution_summary.csv", index=False)
distribution_df.to_json(f"{JSON_DIR}/distribution_summary.json", orient="records", indent=4)

# Crop Distribution
crop_dist = df["label"].value_counts().reset_index()
crop_dist.columns = ["Crop", "Count"]
crop_dist["Percentage"] = (crop_dist["Count"] / len(df) * 100).round(2)
print("\nCrop Distribution:")
print(crop_dist)
crop_dist.to_csv(f"{CSV_DIR}/crop_distribution.csv", index=False)

plt.figure(figsize=(10, 7))
sns.countplot(data=df, y="label", order=df["label"].value_counts().index)
plt.title("Crop Distribution")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/crop_distribution.png", dpi=300)
plt.close()

# Soil Nutrient Analysis
soil_stats = df.groupby("label")[["N", "P", "K"]].agg(["mean", "median", "min", "max", "std"]).round(2)
print("\nSoil Nutrient Statistics (first 5):")
print(soil_stats.head())
soil_stats.to_csv(f"{CSV_DIR}/soil_crop_statistics.csv")

for nutrient in ["N", "P", "K"]:
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="label", y=nutrient)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/{nutrient}_boxplot.png", dpi=300)
    plt.close()

# Climate Analysis
climate_cols = ["temperature", "humidity", "ph", "rainfall"]
climate_stats = df.groupby("label")[climate_cols].mean().round(2)
print("\nClimate Statistics (first 5):")
print(climate_stats.head())
climate_stats.to_csv(f"{CSV_DIR}/climate_statistics.csv")

for feature in climate_cols:
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="label", y=feature)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/{feature}_boxplot.png", dpi=300)
    plt.close()

# Correlation Analysis
pearson = df[numeric_cols].corr(method="pearson")
spearman = df[numeric_cols].corr(method="spearman")
pearson.to_csv(f"{CSV_DIR}/pearson_correlation.csv")
spearman.to_csv(f"{CSV_DIR}/spearman_correlation.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(pearson, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Pearson Correlation")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/pearson_heatmap.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(spearman, annot=True, cmap="viridis", fmt=".2f")
plt.title("Spearman Correlation")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/spearman_heatmap.png", dpi=300)
plt.close()

# Research Insights (Basic)
insights = []
for col in numeric_cols:
    insights.append({
        "Feature": col,
        "Highest Mean Crop": df.groupby("label")[col].mean().idxmax(),
        "Lowest Mean Crop": df.groupby("label")[col].mean().idxmin(),
        "Overall Mean": round(df[col].mean(), 2)
    })
insights_df = pd.DataFrame(insights)
print("\nResearch Insights:")
print(insights_df)
insights_df.to_csv(f"{CSV_DIR}/research_insights.csv", index=False)
insights_df.to_json(f"{JSON_DIR}/research_insights.json", orient="records", indent=4)

print("\nPhase 2 Complete: distribution_summary, crop_distribution, soil_crop_statistics, climate_statistics, pearson_correlation, spearman_correlation, research_insights, Histograms, Crop distribution, Nutrient boxplots, Climate boxplots")

# ============================================================================
# PHASE 3: ADVANCED RESEARCH ANALYTICS
# ============================================================================

print("\n" + "="*60)
print("PHASE 3: Advanced Research Analytics")
print("="*60)

ADV_DIR = f"{CHART_DIR}/advanced"
os.makedirs(ADV_DIR, exist_ok=True)

X = df[numeric_cols]
X_scaled = StandardScaler().fit_transform(X)

# PCA Analysis
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
pca_df["label"] = df["label"]
pca_df.to_csv(f"{CSV_DIR}/pca_coordinates.csv", index=False)

variance = pd.DataFrame({
    "Component": ["PC1", "PC2"],
    "ExplainedVariance": pca.explained_variance_ratio_
})
variance.to_csv(f"{CSV_DIR}/pca_variance.csv", index=False)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="label", legend=False, s=35)
plt.title("PCA Projection")
plt.tight_layout()
plt.savefig(f"{ADV_DIR}/pca_scatter.png", dpi=300)
plt.close()

# Crop Requirement Heatmap
crop_profile = df.groupby("label")[numeric_cols].mean().round(2)
crop_profile.to_csv(f"{CSV_DIR}/crop_profiles.csv")

plt.figure(figsize=(12, 8))
sns.heatmap(crop_profile, cmap="YlGnBu")
plt.title("Crop Requirement Heatmap")
plt.tight_layout()
plt.savefig(f"{ADV_DIR}/crop_profile_heatmap.png", dpi=300)
plt.close()

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

cluster_df = pca_df.copy()
cluster_df["cluster"] = clusters
cluster_df.to_csv(f"{CSV_DIR}/cluster_results.csv", index=False)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=35)
plt.title("KMeans Clusters")
plt.tight_layout()
plt.savefig(f"{ADV_DIR}/kmeans_clusters.png", dpi=300)
plt.close()

# Hierarchical Clustering
plt.figure(figsize=(12, 5))
Z = linkage(X_scaled[:200], method="ward")
dendrogram(Z, no_labels=True)
plt.title("Hierarchical Clustering (First 200 Samples)")
plt.tight_layout()
plt.savefig(f"{ADV_DIR}/hierarchical_dendrogram.png", dpi=300)
plt.close()

# Outlier Detection
z = np.abs(zscore(X))
outlier_mask = (z > 3).any(axis=1)
outliers = df[outlier_mask].copy()
outliers.to_csv(f"{CSV_DIR}/outliers.csv", index=False)

out_summary = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    count = ((df[col] < low) | (df[col] > high)).sum()
    out_summary.append({"Feature": col, "IQR_Outliers": int(count)})
out_summary_df = pd.DataFrame(out_summary)
print("\nOutlier Summary:")
print(out_summary_df)
out_summary_df.to_csv(f"{CSV_DIR}/outlier_summary.csv", index=False)

# Feature Importance
y = df["label"]
mi = mutual_info_classif(X, y, random_state=42)
f_vals, p_vals = f_classif(X, y)

importance = pd.DataFrame({
    "Feature": numeric_cols,
    "MutualInformation": mi,
    "ANOVA_FScore": f_vals,
    "PValue": p_vals
}).sort_values("MutualInformation", ascending=False)
print("\nFeature Importance:")
print(importance)
importance.to_csv(f"{CSV_DIR}/feature_importance.csv", index=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=importance, x="MutualInformation", y="Feature")
plt.title("Feature Importance (Mutual Information)")
plt.tight_layout()
plt.savefig(f"{ADV_DIR}/feature_importance.png", dpi=300)
plt.close()

# Automated Research Observations
obs = []
for feat in numeric_cols:
    means = df.groupby("label")[feat].mean()
    obs.append({
        "Observation": f"Highest average {feat}",
        "Crop": means.idxmax(),
        "Value": round(means.max(), 2)
    })
    obs.append({
        "Observation": f"Lowest average {feat}",
        "Crop": means.idxmin(),
        "Value": round(means.min(), 2)
    })

obs_df = pd.DataFrame(obs)
print("\nResearch Observations (first 10):")
print(obs_df.head(10))
obs_df.to_csv(f"{CSV_DIR}/research_observations.csv", index=False)
obs_df.to_json(f"{JSON_DIR}/research_observations.json", orient="records", indent=4)

print("\nPhase 3 Complete: PCA, Crop profile heatmap, K-Means clustering, Hierarchical clustering, Z-score & IQR outlier detection, Mutual Information, ANOVA F-score, Automated research observations")

# ============================================================================
# PHASE 4: FINAL REPORT & DASHBOARD EXPORT
# ============================================================================

print("\n" + "="*60)
print("PHASE 4: Final Report & Dashboard Export")
print("="*60)

REPORT_DIR = f"{OUTPUT_DIR}/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Dashboard Summary JSON
dashboard_summary = {
    "dataset": {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target": "label"
    },
    "files": {
        "csv": sorted(os.listdir(CSV_DIR)),
        "json": sorted(os.listdir(JSON_DIR)),
        "charts": sorted(os.listdir(CHART_DIR))
    },
    "quality": quality,
    "validation": validation
}

with open(f"{JSON_DIR}/dashboard_summary.json", "w") as f:
    json.dump(dashboard_summary, f, indent=4)
print("Dashboard summary created.")

# Human-readable research summary
lines = []
lines.append("OptiCrop Research Analytics Report")
lines.append("=" * 40)
lines.append(f"Dataset Size : {len(df)} rows x {len(df.columns)} columns")
lines.append(f"Number of Crops : {len(df['label'].unique())}")
lines.append("")
lines.append("Highest Mean Values")
for feat in numeric_cols:
    grp = df.groupby("label")[feat].mean()
    lines.append(f"- {feat}: {grp.idxmax()} ({grp.max():.2f})")
summary_txt = "\n".join(lines)
with open(f"{REPORT_DIR}/research_summary.txt", "w") as f:
    f.write(summary_txt)
print("\n" + summary_txt)

# PDF Report
doc = SimpleDocTemplate(f"{REPORT_DIR}/OptiCrop_Research_Report.pdf")
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("<b>OptiCrop Research Analytics Report</b>", styles["Title"]))
story.append(Paragraph(f"Rows: {len(df)}", styles["BodyText"]))
story.append(Paragraph(f"Columns: {len(df.columns)}", styles["BodyText"]))
story.append(Paragraph(f"Crop Classes: {df['label'].unique()}", styles["BodyText"]))

story.append(Paragraph("<br/><b>Data Quality</b>", styles["Heading2"]))
for k, v in quality.items():
    story.append(Paragraph(f"{k}: {v}", styles["BodyText"]))

story.append(Paragraph("<br/><b>Key Research Observations</b>", styles["Heading2"]))
for feat in numeric_cols:
    grp = df.groupby("label")[feat].mean()
    story.append(Paragraph(
        f"{feat}: Highest average observed for <b>{grp.idxmax()}</b> ({grp.max():.2f})",
        styles["BodyText"]
    ))

doc.build(story)
print("PDF Report Generated")

# Final Export Manifest
manifest = []
for folder in [CSV_DIR, JSON_DIR, CHART_DIR, REPORT_DIR]:
    for file in sorted(os.listdir(folder)):
        manifest.append({
            "Folder": os.path.basename(folder),
            "File": file
        })
manifest_df = pd.DataFrame(manifest)
print("\nExport Manifest:")
print(manifest_df)
manifest_df.to_csv(f"{CSV_DIR}/export_manifest.csv", index=False)

print("\n" + "="*60)
print("OptiCrop Research Analytics Pipeline Complete")
print("="*60)
print("\nOutputs generated:")
print(f"  - CSV files: {CSV_DIR}")
print(f"  - JSON files: {JSON_DIR}")
print(f"  - Charts: {CHART_DIR}")
print(f"  - Reports: {REPORT_DIR}")
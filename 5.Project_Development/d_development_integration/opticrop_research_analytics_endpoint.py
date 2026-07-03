# Add these imports at the top of your file
import os
import json
import warnings

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from flask import Flask, request, jsonify
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif, f_classif
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import base64
import io

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")



# ============================================================================
# HELPER FUNCTION TO CONVERT PLOT TO BASE64
# ============================================================================

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def convert_to_serializable(obj):
    """Convert any object to JSON serializable format"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ============================================================================
# OPTICROP RESEARCH ANALYTICS ENDPOINT
# ============================================================================


def research_analytics(file, dataset_name):
    """
    Runs the complete OptiCrop research analytics pipeline on an uploaded file.
    """
    try:
        try:
            df = pd.read_csv(file)

        except Exception:
            file.seek(0)

            try:
                df = pd.read_excel(file)

            except Exception:
                return {
                    "success": False,
                    "error": "Unsupported or corrupted file. Please upload a valid CSV or XLSX file."
                }, 400
        
        # Validate required columns
        REQUIRED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            return {
                "success": False,
                "error": f"Missing required columns: {missing_cols}. Required columns: {REQUIRED_COLUMNS}"
            }, 400
        
        # Sanitize dataset name
        dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in (' ', '-', '_')).strip()
        dataset_name = dataset_name.replace(' ', '_')
        
        # Setup output directories with dataset name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = os.path.join("outputs", f"{dataset_name}_{timestamp}")
        CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
        JSON_DIR = os.path.join(OUTPUT_DIR, "json")
        CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
        
        for d in [OUTPUT_DIR, CSV_DIR, JSON_DIR, CHART_DIR]:
            os.makedirs(d, exist_ok=True)
        
        numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        
        # Initialize response data
        response_data = {
            "success": True,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "output_dir": OUTPUT_DIR,
            "files": {
                "csv": [],
                "json": [],
                "charts": []
            },
            "analytics": {},
            "charts": {},
            "key_findings": {}
        }
        
        # ====================================================================
        # PHASE 1: DATASET VALIDATION & DATA QUALITY ASSESSMENT
        # ====================================================================
        
        # Dataset Validation
        validation = []
        validation.append({
            "Check": "Dataset Loaded",
            "Status": "PASS" if not df.empty else "FAIL",
            "Details": f"{len(df)} rows"
        })
        
        missing_cols_check = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        validation.append({
            "Check": "Required Columns",
            "Status": "PASS" if len(missing_cols_check) == 0 else "FAIL",
            "Details": "None" if len(missing_cols_check) == 0 else ",".join(missing_cols_check)
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
        validation_df.to_csv(f"{CSV_DIR}/validation_report.csv", index=False)
        validation_df.to_json(f"{JSON_DIR}/validation_report.json", orient="records", indent=4)
        response_data["files"]["csv"].append("validation_report.csv")
        response_data["files"]["json"].append("validation_report.json")
        response_data["analytics"]["validation"] = validation
        
        # Data Quality Assessment
        quality = {
            "Total Rows": int(len(df)),
            "Total Columns": int(len(df.columns)),
            "Missing Values": int(df.isnull().sum().sum()),
            "Duplicate Rows": int(df.duplicated().sum()),
            "Numeric Features": int(len(df.select_dtypes(include=np.number).columns)),
            "Categorical Features": int(len(df.select_dtypes(include="object").columns))
        }
        
        quality_df = pd.DataFrame([quality])
        quality_df.to_csv(f"{CSV_DIR}/data_quality.csv", index=False)
        with open(f"{JSON_DIR}/data_quality.json", "w") as f:
            json.dump(quality, f, indent=4)
        response_data["files"]["csv"].append("data_quality.csv")
        response_data["files"]["json"].append("data_quality.json")
        response_data["analytics"]["quality"] = quality
        
        # Descriptive Statistics
        desc_stats = df[numeric_cols].describe()
        desc_stats.to_csv(f"{CSV_DIR}/descriptive_statistics.csv")
        desc_stats.to_json(f"{JSON_DIR}/descriptive_statistics.json", orient="index", indent=4)
        response_data["files"]["csv"].append("descriptive_statistics.csv")
        response_data["files"]["json"].append("descriptive_statistics.json")
        response_data["analytics"]["descriptive_statistics"] = convert_to_serializable(desc_stats.to_dict())
        
        # Missing Value Analysis
        missing_summary = df.isnull().sum().reset_index()
        missing_summary.columns = ["Column", "MissingCount"]
        missing_summary["MissingPercent"] = (missing_summary["MissingCount"] / len(df) * 100).round(2)
        missing_summary.to_csv(f"{CSV_DIR}/missing_value_summary.csv", index=False)
        missing_summary.to_json(f"{JSON_DIR}/missing_value_summary.json", orient="records", indent=4)
        response_data["files"]["csv"].append("missing_value_summary.csv")
        response_data["files"]["json"].append("missing_value_summary.json")
        response_data["analytics"]["missing_values"] = convert_to_serializable(missing_summary.to_dict(orient="records"))
        
        # Missing Value Charts
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.bar(df, ax=ax)
        plt.title("Missing Value Bar Chart")
        plt.tight_layout()
        plt.savefig(f"{CHART_DIR}/missing_value_bar.png", dpi=300)
        response_data["charts"]["missing_value_bar"] = plot_to_base64(fig)
        response_data["files"]["charts"].append("missing_value_bar.png")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.heatmap(df, ax=ax)
        plt.title("Missing Value Heatmap")
        plt.tight_layout()
        plt.savefig(f"{CHART_DIR}/missing_value_heatmap.png", dpi=300)
        response_data["charts"]["missing_value_heatmap"] = plot_to_base64(fig)
        response_data["files"]["charts"].append("missing_value_heatmap.png")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(df, ax=ax)
        plt.title("Missingno Matrix")
        plt.tight_layout()
        plt.savefig(f"{CHART_DIR}/missingno_matrix.png", dpi=300)
        response_data["charts"]["missingno_matrix"] = plot_to_base64(fig)
        response_data["files"]["charts"].append("missingno_matrix.png")
        
        # Dataset Summary
        summary = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
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
        response_data["files"]["json"].append("summary.json")
        
        # ====================================================================
        # PHASE 2: DISTRIBUTION, CROP, SOIL, CLIMATE & CORRELATION ANALYSIS
        # ====================================================================
        
        DIST_DIR = f"{CHART_DIR}/distributions"
        os.makedirs(DIST_DIR, exist_ok=True)
        response_data["charts"]["distributions"] = {}
        
        # Distribution Analysis
        distribution_summary = []
        for col in numeric_cols:
            dist_stats = {
                "Feature": col,
                "Mean": float(df[col].mean()),
                "Median": float(df[col].median()),
                "Std": float(df[col].std()),
                "Min": float(df[col].min()),
                "Max": float(df[col].max()),
                "Skewness": float(df[col].skew())
            }
            distribution_summary.append(dist_stats)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"{col} Distribution")
            plt.tight_layout()
            plt.savefig(f"{DIST_DIR}/{col}_distribution.png", dpi=300)
            response_data["charts"]["distributions"][col] = plot_to_base64(fig)
            response_data["files"]["charts"].append(f"distributions/{col}_distribution.png")
        
        distribution_df = pd.DataFrame(distribution_summary).round(4)
        distribution_df.to_csv(f"{CSV_DIR}/distribution_summary.csv", index=False)
        distribution_df.to_json(f"{JSON_DIR}/distribution_summary.json", orient="records", indent=4)
        response_data["files"]["csv"].append("distribution_summary.csv")
        response_data["files"]["json"].append("distribution_summary.json")
        response_data["analytics"]["distribution"] = distribution_summary
        
        # Crop Distribution
        response_data["charts"]["nutrient_boxplots"] = {}
        response_data["charts"]["climate_boxplots"] = {}
        
        if "label" in df.columns:
            crop_dist = df["label"].value_counts().reset_index()
            crop_dist.columns = ["Crop", "Count"]
            crop_dist["Percentage"] = (crop_dist["Count"] / len(df) * 100).round(2)
            crop_dist.to_csv(f"{CSV_DIR}/crop_distribution.csv", index=False)
            response_data["files"]["csv"].append("crop_distribution.csv")
            response_data["analytics"]["crop_distribution"] = convert_to_serializable(crop_dist.to_dict(orient="records"))
            
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.countplot(data=df, y="label", order=df["label"].value_counts().index, ax=ax)
            ax.set_title("Crop Distribution")
            plt.tight_layout()
            plt.savefig(f"{CHART_DIR}/crop_distribution.png", dpi=300)
            response_data["charts"]["crop_distribution"] = plot_to_base64(fig)
            response_data["files"]["charts"].append("crop_distribution.png")
            
            # Soil Nutrient Analysis
            soil_stats = df.groupby("label")[["N", "P", "K"]].agg(["mean", "median", "min", "max", "std"]).round(2)
            soil_stats.to_csv(f"{CSV_DIR}/soil_crop_statistics.csv")
            response_data["files"]["csv"].append("soil_crop_statistics.csv")
            response_data["analytics"]["soil_nutrients"] = convert_to_serializable(soil_stats.to_dict())
            
            for nutrient in ["N", "P", "K"]:
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.boxplot(data=df, x="label", y=nutrient, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                plt.tight_layout()
                plt.savefig(f"{CHART_DIR}/{nutrient}_boxplot.png", dpi=300)
                response_data["charts"]["nutrient_boxplots"][nutrient] = plot_to_base64(fig)
                response_data["files"]["charts"].append(f"{nutrient}_boxplot.png")
            
            # Climate Analysis
            climate_cols = ["temperature", "humidity", "ph", "rainfall"]
            climate_stats = df.groupby("label")[climate_cols].mean().round(2)
            climate_stats.to_csv(f"{CSV_DIR}/climate_statistics.csv")
            response_data["files"]["csv"].append("climate_statistics.csv")
            response_data["analytics"]["climate"] = convert_to_serializable(climate_stats.to_dict())
            
            for feature in climate_cols:
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.boxplot(data=df, x="label", y=feature, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                plt.tight_layout()
                plt.savefig(f"{CHART_DIR}/{feature}_boxplot.png", dpi=300)
                response_data["charts"]["climate_boxplots"][feature] = plot_to_base64(fig)
                response_data["files"]["charts"].append(f"{feature}_boxplot.png")
        
        # Correlation Analysis
        pearson = df[numeric_cols].corr(method="pearson")
        spearman = df[numeric_cols].corr(method="spearman")
        pearson.to_csv(f"{CSV_DIR}/pearson_correlation.csv")
        spearman.to_csv(f"{CSV_DIR}/spearman_correlation.csv")
        response_data["files"]["csv"].append("pearson_correlation.csv")
        response_data["files"]["csv"].append("spearman_correlation.csv")
        response_data["analytics"]["correlation"] = {
            "pearson": convert_to_serializable(pearson.to_dict()),
            "spearman": convert_to_serializable(spearman.to_dict())
        }
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pearson, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Pearson Correlation")
        plt.tight_layout()
        plt.savefig(f"{CHART_DIR}/pearson_heatmap.png", dpi=300)
        response_data["charts"]["pearson_heatmap"] = plot_to_base64(fig)
        response_data["files"]["charts"].append("pearson_heatmap.png")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(spearman, annot=True, cmap="viridis", fmt=".2f", ax=ax)
        ax.set_title("Spearman Correlation")
        plt.tight_layout()
        plt.savefig(f"{CHART_DIR}/spearman_heatmap.png", dpi=300)
        response_data["charts"]["spearman_heatmap"] = plot_to_base64(fig)
        response_data["files"]["charts"].append("spearman_heatmap.png")
        
        # Research Insights (Basic)
        insights = []
        for col in numeric_cols:
            if "label" in df.columns:
                highest_crop = df.groupby("label")[col].mean().idxmax()
                lowest_crop = df.groupby("label")[col].mean().idxmin()
            else:
                highest_crop = "N/A"
                lowest_crop = "N/A"
            insights.append({
                "Feature": col,
                "Highest Mean Crop": highest_crop,
                "Lowest Mean Crop": lowest_crop,
                "Overall Mean": round(float(df[col].mean()), 2)
            })
        insights_df = pd.DataFrame(insights)
        insights_df.to_csv(f"{CSV_DIR}/research_insights.csv", index=False)
        insights_df.to_json(f"{JSON_DIR}/research_insights.json", orient="records", indent=4)
        response_data["files"]["csv"].append("research_insights.csv")
        response_data["files"]["json"].append("research_insights.json")
        response_data["analytics"]["research_insights"] = insights
        
        # ====================================================================
        # PHASE 3: ADVANCED RESEARCH ANALYTICS
        # ====================================================================
        
        ADV_DIR = f"{CHART_DIR}/advanced"
        os.makedirs(ADV_DIR, exist_ok=True)
        
        X = df[numeric_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        
        # PCA Analysis
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        if "label" in df.columns:
            pca_df["label"] = df["label"].values
        pca_df.to_csv(f"{CSV_DIR}/pca_coordinates.csv", index=False)
        response_data["files"]["csv"].append("pca_coordinates.csv")
        
        variance = pd.DataFrame({
            "Component": ["PC1", "PC2"],
            "ExplainedVariance": pca.explained_variance_ratio_
        })
        variance.to_csv(f"{CSV_DIR}/pca_variance.csv", index=False)
        response_data["files"]["csv"].append("pca_variance.csv")
        response_data["analytics"]["pca"] = {
            "coordinates": convert_to_serializable(pca_df.to_dict(orient="records")),
            "variance": convert_to_serializable(variance.to_dict(orient="records")),
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]
        }
        
        fig, ax = plt.subplots(figsize=(8, 6))
        if "label" in df.columns:
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="label", legend=False, s=35, ax=ax)
        else:
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=35, ax=ax)
        ax.set_title("PCA Projection")
        plt.tight_layout()
        plt.savefig(f"{ADV_DIR}/pca_scatter.png", dpi=300)
        response_data["charts"]["pca_scatter"] = plot_to_base64(fig)
        response_data["files"]["charts"].append("advanced/pca_scatter.png")
        
        # Crop Requirement Heatmap
        if "label" in df.columns:
            crop_profile = df.groupby("label")[numeric_cols].mean().round(2)
            crop_profile.to_csv(f"{CSV_DIR}/crop_profiles.csv")
            response_data["files"]["csv"].append("crop_profiles.csv")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(crop_profile, cmap="YlGnBu", ax=ax)
            ax.set_title("Crop Requirement Heatmap")
            plt.tight_layout()
            plt.savefig(f"{ADV_DIR}/crop_profile_heatmap.png", dpi=300)
            response_data["charts"]["crop_profile_heatmap"] = plot_to_base64(fig)
            response_data["files"]["charts"].append("advanced/crop_profile_heatmap.png")
        
        # KMeans Clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        cluster_df = pca_df.copy()
        cluster_df["cluster"] = clusters
        cluster_df.to_csv(f"{CSV_DIR}/cluster_results.csv", index=False)
        response_data["files"]["csv"].append("cluster_results.csv")
        response_data["analytics"]["clustering"] = {
            "kmeans": {
                "cluster_centers": convert_to_serializable(kmeans.cluster_centers_.tolist()),
                "inertia": float(kmeans.inertia_),
                "n_iter": int(kmeans.n_iter_)
            }
        }
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=35, ax=ax)
        ax.set_title("KMeans Clusters")
        plt.tight_layout()
        plt.savefig(f"{ADV_DIR}/kmeans_clusters.png", dpi=300)
        response_data["charts"]["kmeans_clusters"] = plot_to_base64(fig)
        response_data["files"]["charts"].append("advanced/kmeans_clusters.png")
        
        # Hierarchical Clustering
        fig, ax = plt.subplots(figsize=(12, 5))
        Z = linkage(X_scaled[:min(200, len(X_scaled))], method="ward")
        dendrogram(Z, no_labels=True, ax=ax)
        ax.set_title("Hierarchical Clustering (First 200 Samples)")
        plt.tight_layout()
        plt.savefig(f"{ADV_DIR}/hierarchical_dendrogram.png", dpi=300)
        response_data["charts"]["hierarchical_dendrogram"] = plot_to_base64(fig)
        response_data["files"]["charts"].append("advanced/hierarchical_dendrogram.png")
        
        # Outlier Detection
        z = np.abs(zscore(X))
        outlier_mask = (z > 3).any(axis=1)
        outliers = df[outlier_mask].copy()
        outliers.to_csv(f"{CSV_DIR}/outliers.csv", index=False)
        response_data["files"]["csv"].append("outliers.csv")
        
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
        out_summary_df.to_csv(f"{CSV_DIR}/outlier_summary.csv", index=False)
        response_data["files"]["csv"].append("outlier_summary.csv")
        response_data["analytics"]["outliers"] = {
            "z_score_outliers": int(outlier_mask.sum()),
            "iqr_summary": out_summary,
            "outlier_rows": convert_to_serializable(outliers.to_dict(orient="records"))
        }
        
        # Feature Importance
        if "label" in df.columns:
            y = df["label"]
            mi = mutual_info_classif(X, y, random_state=42)
            f_vals, p_vals = f_classif(X, y)
            
            importance = pd.DataFrame({
                "Feature": numeric_cols,
                "MutualInformation": [float(x) for x in mi],
                "ANOVA_FScore": [float(x) for x in f_vals],
                "PValue": [float(x) for x in p_vals]
            }).sort_values("MutualInformation", ascending=False)
            importance.to_csv(f"{CSV_DIR}/feature_importance.csv", index=False)
            response_data["files"]["csv"].append("feature_importance.csv")
            response_data["analytics"]["feature_importance"] = convert_to_serializable(importance.to_dict(orient="records"))
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=importance, x="MutualInformation", y="Feature", ax=ax)
            ax.set_title("Feature Importance (Mutual Information)")
            plt.tight_layout()
            plt.savefig(f"{ADV_DIR}/feature_importance.png", dpi=300)
            response_data["charts"]["feature_importance"] = plot_to_base64(fig)
            response_data["files"]["charts"].append("advanced/feature_importance.png")
        
        # Automated Research Observations
        obs = []
        if "label" in df.columns:
            for feat in numeric_cols:
                means = df.groupby("label")[feat].mean()
                obs.append({
                    "Observation": f"Highest average {feat}",
                    "Crop": means.idxmax(),
                    "Value": float(round(means.max(), 2))
                })
                obs.append({
                    "Observation": f"Lowest average {feat}",
                    "Crop": means.idxmin(),
                    "Value": float(round(means.min(), 2))
                })
        
        obs_df = pd.DataFrame(obs)
        obs_df.to_csv(f"{CSV_DIR}/research_observations.csv", index=False)
        obs_df.to_json(f"{JSON_DIR}/research_observations.json", orient="records", indent=4)
        response_data["files"]["csv"].append("research_observations.csv")
        response_data["files"]["json"].append("research_observations.json")
        response_data["analytics"]["research_observations"] = obs
        
        # ====================================================================
        # PHASE 4: FINAL REPORT & DASHBOARD EXPORT
        # ====================================================================
        
        REPORT_DIR = f"{OUTPUT_DIR}/reports"
        os.makedirs(REPORT_DIR, exist_ok=True)
        
        # Dashboard Summary JSON
        dashboard_summary = {
            "dataset": {
                "name": dataset_name,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "target": "label" if "label" in df.columns else None
            },
            "files": {
                "csv": sorted(os.listdir(CSV_DIR)),
                "json": sorted(os.listdir(JSON_DIR)),
                "charts": sorted(os.listdir(CHART_DIR))
            },
            "quality": quality,
            "validation": validation,
            "timestamp": timestamp
        }
        
        with open(f"{JSON_DIR}/dashboard_summary.json", "w") as f:
            json.dump(dashboard_summary, f, indent=4)
        response_data["files"]["json"].append("dashboard_summary.json")
        
        # Human-readable research summary
        lines = []
        lines.append("OptiCrop Research Analytics Report")
        lines.append("=" * 40)
        lines.append(f"Dataset: {dataset_name}")
        lines.append(f"Dataset Size : {len(df)} rows x {len(df.columns)} columns")
        if "label" in df.columns:
            lines.append(f"Number of Crops : {len(df['label'].unique())}")
        lines.append("")
        lines.append("Highest Mean Values")
        if "label" in df.columns:
            for feat in numeric_cols:
                grp = df.groupby("label")[feat].mean()
                lines.append(f"- {feat}: {grp.idxmax()} ({grp.max():.2f})")
        summary_txt = "\n".join(lines)
        with open(f"{REPORT_DIR}/research_summary.txt", "w") as f:
            f.write(summary_txt)
        response_data["files"]["reports"] = ["research_summary.txt"]
        
        # PDF Report
        try:
            doc = SimpleDocTemplate(f"{REPORT_DIR}/OptiCrop_Research_Report.pdf")
            styles = getSampleStyleSheet()
            story = []
            
            story.append(Paragraph("<b>OptiCrop Research Analytics Report</b>", styles["Title"]))
            story.append(Paragraph(f"Dataset: {dataset_name}", styles["BodyText"]))
            story.append(Paragraph(f"Rows: {len(df)}", styles["BodyText"]))
            story.append(Paragraph(f"Columns: {len(df.columns)}", styles["BodyText"]))
            if "label" in df.columns:
                story.append(Paragraph(f"Crop Classes: {df['label'].unique()}", styles["BodyText"]))
            
            story.append(Paragraph("<br/><b>Data Quality</b>", styles["Heading2"]))
            for k, v in quality.items():
                story.append(Paragraph(f"{k}: {v}", styles["BodyText"]))
            
            if "label" in df.columns:
                story.append(Paragraph("<br/><b>Key Research Observations</b>", styles["Heading2"]))
                for feat in numeric_cols:
                    grp = df.groupby("label")[feat].mean()
                    story.append(Paragraph(
                        f"{feat}: Highest average observed for <b>{grp.idxmax()}</b> ({grp.max():.2f})",
                        styles["BodyText"]
                    ))
            
            doc.build(story)
            response_data["files"]["reports"].append("OptiCrop_Research_Report.pdf")
        except Exception as e:
            print(f"PDF generation warning: {e}")
        
        # Final Export Manifest
        manifest = []
        for folder in [CSV_DIR, JSON_DIR, CHART_DIR, REPORT_DIR]:
            for file in sorted(os.listdir(folder)):
                manifest.append({
                    "Folder": os.path.basename(folder),
                    "File": file
                })
        manifest_df = pd.DataFrame(manifest)
        manifest_df.to_csv(f"{CSV_DIR}/export_manifest.csv", index=False)
        response_data["files"]["csv"].append("export_manifest.csv")
        
        # Key findings summary
        key_findings = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "crop_count": int(len(df["label"].unique())) if "label" in df.columns else None,
            "quality_metrics": quality,
            "validation_status": [v for v in validation if v["Status"] == "FAIL"],
            "outlier_count": int(outlier_mask.sum()),
            "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_]
        }
        
        if "label" in df.columns:
            key_findings["highest_mean_values"] = {
                feat: {
                    "crop": str(df.groupby("label")[feat].mean().idxmax()),
                    "value": float(round(df.groupby("label")[feat].mean().max(), 2))
                }
                for feat in numeric_cols
            }
            
            # Get strongest correlations
            pearson_vals = pearson.unstack().sort_values(ascending=False)
            # Remove self-correlations
            pearson_vals = pearson_vals[pearson_vals < 0.999]
            if len(pearson_vals) > 0:
                key_findings["correlations"] = {
                    "strongest_positive": {
                        "pair": list(pearson_vals.index[0]) if len(pearson_vals.index) > 0 else None,
                        "value": float(pearson_vals.values[0]) if len(pearson_vals) > 0 else None
                    },
                    "strongest_negative": {
                        "pair": list(pearson_vals.index[-1]) if len(pearson_vals.index) > 0 else None,
                        "value": float(pearson_vals.values[-1]) if len(pearson_vals) > 0 else None
                    }
                }
        
        response_data["key_findings"] = key_findings
        
        # Convert entire response to JSON serializable
        response_data = convert_to_serializable(response_data)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500




# Usage 
# import requests

# url = "http://localhost:7000/research-analytics"
# files = {'file': open('your_dataset.csv', 'rb')}
# data = {'dataset_name': 'my_crop_data'}

# response = requests.post(url, files=files, data=data)
# result = response.json()

# # Access analytics
# print(result['analytics']['quality'])
# print(result['key_findings']['highest_mean_values'])

# # Display charts in frontend
# img_data = result['charts']['pearson_heatmap']
# # Use in HTML: <img src="data:image/png;base64,{img_data}" />
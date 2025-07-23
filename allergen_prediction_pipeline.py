# === Required Libraries ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from Bio import SeqIO
from tqdm import tqdm
from typing import Union, List, Dict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

# === Global Descriptor Calculation ===
def calculate_global_descriptors(fasta_file: str, output_csv: str):
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, 'fasta')]
    desc = GlobalDescriptor(sequences)
    desc.calculate_all(amide=True)
    df = pd.DataFrame(desc.descriptor, columns=desc.featurenames)
    df.insert(0, 'Identifier', [record.id for record in SeqIO.parse(fasta_file, 'fasta')])
    df.to_csv(output_csv, index=False)
    return df

# === Advanced Peptide Descriptor Class ===
class AdvancedPeptideAnalyzer:
    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self.descriptor_scales = [
            'AASI', 'ABHPRK', 'argos', 'bulkiness', 'charge_phys', 'charge_acid',
            'cougar', 'eisenberg', 'Ez', 'flexibility', 'grantham', 'gravy',
            'hopp-woods', 'ISAECI', 'janin', 'kytedoolittle', 'levitt_alpha',
            'MSS', 'MSW', 'pepArc', 'pepcats', 'polarity', 'PPCALI',
            'refractivity', 't_scale', 'TM_tend', 'z3', 'z5'
        ]
        self.special_scales = {
            'autocorr': ['PPCALI', 'pepcats'],
            'crosscorr': ['pepcats'],
            'moment': ['eisenberg', 'kytedoolittle'],
            'profile': ['eisenberg', 'kytedoolittle'],
            'arc': ['pepArc']
        }

    def safe_extract(self, descriptor):
        try:
            val = descriptor[0]
            return val if isinstance(val, (list, np.ndarray)) else float(val)
        except Exception:
            return None

    def calculate_standard_descriptor(self, sequence, scale):
        try:
            desc = PeptideDescriptor([sequence], scale)
            desc.calculate_global()
            val = self.safe_extract(desc.descriptor)
            return float(np.nanmean(val)) if isinstance(val, list) else val
        except:
            return None

    def process_sequences(self):
        records = list(SeqIO.parse(self.fasta_path, 'fasta'))
        results = []
        for record in tqdm(records):
            seq = str(record.seq)
            result_dict = {'Identifier': record.id, 'Length': len(seq)}
            for scale in self.descriptor_scales:
                result_dict[scale] = self.calculate_standard_descriptor(seq, scale)
            results.append(result_dict)
        return pd.DataFrame(results)

# === Merge Descriptors ===
def merge_descriptors(global_df: pd.DataFrame, advanced_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(global_df, advanced_df, on='Identifier', how='inner')

# === Prepare Allergen/Non-Allergen Data ===
def prepare_allergen_data(allergen_fasta, non_allergen_fasta):
    calculate_global_descriptors(allergen_fasta, 'allergen_global.csv')
    calculate_global_descriptors(non_allergen_fasta, 'non_allergen_global.csv')

    allergen_advanced = AdvancedPeptideAnalyzer(allergen_fasta).process_sequences()
    non_allergen_advanced = AdvancedPeptideAnalyzer(non_allergen_fasta).process_sequences()

    allergen_global = pd.read_csv('allergen_global.csv')
    non_allergen_global = pd.read_csv('non_allergen_global.csv')

    df_allergen = merge_descriptors(allergen_global, allergen_advanced)
    df_non_allergen = merge_descriptors(non_allergen_global, non_allergen_advanced)

    df_allergen.to_csv('merge_allergen_modlamp.csv', index=False)
    df_non_allergen.to_csv('merge_non-allergen_modlamp.csv', index=False)

# === ML Training and Evaluation ===
def run_ml_pipeline():
    df_allergen = pd.read_csv('merge_allergen_modlamp.csv')
    df_non_allergen = pd.read_csv('merge_non-allergen_modlamp.csv')
    df_allergen['label'] = 1
    df_non_allergen['label'] = 0
    df = pd.concat([df_allergen, df_non_allergen], ignore_index=True)

    X = df.drop(columns=['label'])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVC': SVC(probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    metrics_data = []
    plt.figure(figsize=(10, 8))
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for name, model in models.items():
        y_true_all, y_proba_all, y_pred_all = [], [], []
        for train_idx, test_idx in cv.split(X_scaled, y):
            model.fit(X_scaled[train_idx], y.iloc[train_idx])
            y_pred_proba = model.predict_proba(X_scaled[test_idx])[:, 1]
            y_pred = model.predict(X_scaled[test_idx])
            y_true_all.extend(y.iloc[test_idx])
            y_proba_all.extend(y_pred_proba)
            y_pred_all.extend(y_pred)

        fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        cm = confusion_matrix(y_true_all, y_pred_all)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y_true_all, y_pred_all),
            'Precision': precision_score(y_true_all, y_pred_all),
            'Recall (Sensitivity)': recall_score(y_true_all, y_pred_all),
            'Specificity': specificity,
            'F1-score': f1_score(y_true_all, y_pred_all),
            'AUC': roc_auc
        })

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves for All Models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ROC_Curve_All_Models.jpeg', dpi=300)
    plt.show()

    pd.DataFrame(metrics_data).to_csv('Model_Performance_Metrics.csv', index=False)

# === SHAP Analysis ===
def run_shap_analysis():
    df_allergen = pd.read_csv('merge_allergen_modlamp.csv')
    df_non_allergen = pd.read_csv('merge_non-allergen_modlamp.csv')
    df_allergen['label'] = 1
    df_non_allergen['label'] = 0
    df = pd.concat([df_allergen, df_non_allergen], ignore_index=True)

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LGBM': LGBMClassifier(),
        'RF': RandomForestClassifier(),
        'SVC': SVC(probability=True),
        'LogReg': LogisticRegression(max_iter=1000)
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        if name == 'SVC' or name == 'LogReg':
            explainer = shap.KernelExplainer(model.predict_proba, X_train_scaled[:100])
            shap_vals = explainer.shap_values(X_test_scaled[:100])[1]
        else:
            explainer = shap.Explainer(model, X_train_scaled)
            shap_vals = explainer(X_test_scaled)
            shap_vals = shap_vals.values if hasattr(shap_vals, 'values') else shap_vals

        shap.summary_plot(shap_vals, X_test.iloc[:100], plot_type="bar", show=False)
        plt.title(f"SHAP Bar Plot - {name}")
        plt.tight_layout()
        plt.savefig(f"shap_summary_bar_{name}.jpg", dpi=300)
        plt.close()

        shap.summary_plot(shap_vals, X_test.iloc[:100], show=False)
        plt.title(f"SHAP Dot Plot - {name}")
        plt.tight_layout()
        plt.savefig(f"shap_summary_dot_{name}.jpg", dpi=300)
        plt.close()

# === Main ===
if __name__ == '__main__':
    prepare_allergen_data('allergens_dataset.fasta', 'non-allergens_dataset.fasta')
    run_ml_pipeline()
    run_shap_analysis()
    print("\nPipeline completed successfully.")
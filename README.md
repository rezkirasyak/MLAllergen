# MLAllergen
Predicting Allergenicity Using Peptide Physicochemical Features

**MLAllergen** is a machine learning framework for predicting allergenicity using physicochemical and structural descriptors from protein sequences. The models used include XGBoost, LightGBM, Random Forest, SVM, and Logistic Regression.

## Highlights
- Curated dataset of 2,391 allergenic and 2,363 non-allergenic proteins
- 36 physicochemical features extracted using `modlamp`
- SHAP-based model interpretation
- XGBoost achieved AUC of 0.9826 and accuracy of 93.46%

## Open Data Declaration
All datasets used in this study are publicly available and were curated from trusted, open-access biological databases. Experimentally validated allergenic and non-allergenic protein sequences were obtained from AllergenOnline, AllerBase, UniProt, and the AllergenFP web server. Only proteins with evidence of existence at the protein level in UniProt were included. Redundant entries were removed and sequences were curated to ensure quality and balance. The data can be accessed from the respective original sources as listed in the references of the manuscript.

## Reproduction
```bash
pip install -r requirements.txt
python allergen_prediction_pipeline.py

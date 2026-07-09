import json

path = '/mnt/data/college/4_tri/ML-Lab-2547121/notebooks/Lab_4.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Imports
        if "from sklearn.model_selection import train_test_split" in source:
            source = source.replace(
                "from sklearn.model_selection import train_test_split, cross_val_score, KFold",
                "from sklearn.model_selection import train_test_split, cross_val_score, KFold\nfrom sklearn.pipeline import Pipeline"
            )
            
        # 2. Target Mapping
        if "df['target'] = df['y'].map({'M': 0, 'B': 1})" in source:
            source = source.replace("df['target'] = df['y'].map({'M': 0, 'B': 1})", "df['target'] = df['y'].map({'M': 1, 'B': 0})")
            
        # 3. Global Scaling Removal
        if "X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)" in source:
            source = source.replace("scaler = StandardScaler()\n", "")
            source = source.replace("X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)\n", "")
            source = source.replace("X_scaled.head()", "X.head()")
            
        # 4. Cell 7: Splits Loop
        if "X_train, X_test, y_train, y_test = train_test_split(X_scaled" in source:
            source = source.replace(
                "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)\n    scaler = StandardScaler()\n    X_train = scaler.fit_transform(X_train)\n    X_test = scaler.transform(X_test)"
            )
            
        # 5. Cell 10: Using 80:20 split as default
        if "X_train, X_test, y_train, y_test = train_test_split(X_scaled" in source and "Using 80:20 split" in source:
            source = source.replace(
                "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)"
            )
            
        # 6. Cell 14: PCA Visualization
        if "pca = PCA(n_components=2)" in source:
            source = source.replace(
                "X_pca = pca.fit_transform(X_scaled)\nX_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)",
                "X_train_pca = pca.fit_transform(X_train)\nX_test_pca = pca.transform(X_test)\ny_train_pca, y_test_pca = y_train, y_test"
            )
            source = source.replace("cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])", "cmap_light = ListedColormap(['#AAFFAA', '#FFAAAA'])")
            source = source.replace("cmap_bold = ListedColormap(['#FF0000', '#00FF00'])", "cmap_bold = ListedColormap(['#00FF00', '#FF0000'])")

        # 7. Cell 17: Cross-Validation
        if "scores = cross_val_score(knn, X_scaled, y, cv=kfold, scoring='accuracy')" in source:
            source = source.replace(
                "knn = KNeighborsClassifier(n_neighbors=k)\n    scores = cross_val_score(knn, X_scaled, y, cv=kfold, scoring='accuracy')",
                "pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])\n    scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')"
            )
            
        # 8. Confusion Matrix Labels
        if "xticklabels=['Malignant(0)', 'Benign(1)']" in source:
            source = source.replace("xticklabels=['Malignant(0)', 'Benign(1)'], yticklabels=['Malignant(0)', 'Benign(1)']", "xticklabels=['Benign(0)', 'Malignant(1)'], yticklabels=['Benign(0)', 'Malignant(1)']")
            
        # Write back line by line safely!
        cell['source'] = [line + '\n' for line in source.split('\n')]
        # Remove trailing newline from the last element if it shouldn't have one
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

    elif cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "- **Optimal K:**" in source and "\\sqrt{n}$" not in source:
            # First, fix the optimal K that we originally fixed
            source = source.replace(
                "- **Optimal K:** The heuristic $K = \\sqrt{n}$ gave a solid baseline, but rigorous K-Fold Cross-Validation helped pinpoint the true optimal K that maximizes generalization.",
                "- **Optimal K:** The heuristic $K = \\sqrt{n}$ gave a solid baseline (K=21), but rigorous K-Fold Cross-Validation helped pinpoint the true optimal K = 11 that maximizes generalization."
            )
            source = source.replace(
                "- **Splits:** Variations in train-test splits (80:20 vs 90:10) demonstrated that too small of a test set leads to unstable performance metrics, emphasizing the need for cross-validation.",
                "- **Splits:** Variations in train-test splits (70:30, 80:20, and 90:10) demonstrated that too small of a test set leads to unstable performance metrics, emphasizing the need for cross-validation to get robust results."
            )
            source = source.replace(
                "- **Model Performance:** The final KNN model achieved high recall and a strong ROC-AUC score, indicating it is highly capable of distinguishing between benign and malignant tumors, which is the primary goal in medical diagnostics.",
                "- **Model Performance:** The final KNN model achieved strong overall accuracy (~94.7%), but the specific metrics for Malignant detection (Precision: 100.0%, Recall: ~86.0%, F1 Score: ~92.5%) reveal that it missed 14% of the cancerous cases. This emphasizes that plain accuracy is misleading, and optimizing for Recall is critical in medical diagnostics to avoid life-threatening False Negatives."
            )
            source = source.replace(
                "- **Regression vs Classification:** Lab 3 showed how to measure numerical error magnitude (MSE, RMSE) for continuous targets. This lab highlighted that for categorical targets, we must evaluate decision boundaries and the specific types of errors made (False Positives vs False Negatives), using metrics like Recall and Confusion Matrices which are far more informative than plain accuracy, especially in healthcare.",
                "- **Insights from Lab 3 (Regression):** Lab 3 showed how to measure numerical error magnitude (MSE, RMSE) and variance explained (R²) for continuous targets in a linear space.\n- **Insights from Current Lab (Classification):** This lab highlighted that for categorical targets, we must evaluate decision boundaries and the specific types of errors made (False Positives vs False Negatives). Metrics like Recall, F1 Score, and Confusion Matrices are far more informative than plain accuracy or R², especially in critical domains like healthcare."
            )
        
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

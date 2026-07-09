import json

path = '/mnt/data/college/4_tri/ML-Lab-2547121/notebooks/Lab_4.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    
    # Cell 2: Imports
    if "from sklearn.model_selection import train_test_split" in source:
        if "Pipeline" not in source:
            source = source.replace(
                "from sklearn.model_selection import train_test_split, cross_val_score, KFold",
                "from sklearn.model_selection import train_test_split, cross_val_score, KFold\nfrom sklearn.pipeline import Pipeline"
            )
            cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]
            
    # Cell 4: Remove global scaling
    if "X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)" in source:
        source = source.replace(
            "scaler = StandardScaler()\nnp.random.seed(42)\nX_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)\nX_scaled.head()",
            "X.head()"
        ).replace(
            "scaler = StandardScaler()\nX_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)\nX_scaled.head()",
            "X.head()"
        )
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]

    # Cell 7: Splits Loop
    if "X_train, X_test, y_train, y_test = train_test_split(X_scaled" in source and "split_results = []" in source:
        source = source.replace(
            "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)\n    scaler = StandardScaler()\n    X_train = scaler.fit_transform(X_train)\n    X_test = scaler.transform(X_test)"
        )
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]

    # Cell 10: Using 80:20 split as default
    if "X_train, X_test, y_train, y_test = train_test_split(X_scaled" in source and "Using 80:20 split" in source:
        source = source.replace(
            "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)"
        )
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]

    # Cell 14: PCA Visualization
    if "pca = PCA(n_components=2)" in source and "plot_decision_boundary" in source:
        source = source.replace(
            "pca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_scaled)\nX_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)",
            "pca = PCA(n_components=2)\nX_train_pca = pca.fit_transform(X_train)\nX_test_pca = pca.transform(X_test)\ny_train_pca, y_test_pca = y_train, y_test"
        )
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]

    # Cell 17: Cross-Validation
    if "scores = cross_val_score(knn, X_scaled, y, cv=kfold, scoring='accuracy')" in source:
        source = source.replace(
            "knn = KNeighborsClassifier(n_neighbors=k)\n    scores = cross_val_score(knn, X_scaled, y, cv=kfold, scoring='accuracy')",
            "pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])\n    scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')"
        )
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

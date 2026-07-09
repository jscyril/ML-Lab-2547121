"""
Fix script for Lab_4.ipynb
Addresses:
1. Misleading target label comment (M=1, B=0 but comment said the opposite)
2. "Feature Scaling" comment on a cell that only separates X and y
3. K search range: ±10 step 2 → ±5 step 1
4. Pipeline consistency: final model now uses Pipeline (like cross-validation)
5. Preserve raw X_train/X_test for Pipeline-based final model
"""
import json

NOTEBOOK_PATH = "notebooks/Lab_4.ipynb"

with open(NOTEBOOK_PATH, "r") as f:
    nb = json.load(f)

cells = nb["cells"]

for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue

    source = cell["source"]
    ec = cell.get("execution_count")

    # --- Fix 1: Target mapping comment (execution_count 1) ---
    if ec == 1:
        for j, line in enumerate(source):
            if "# Map target to 0 (Malignant) and 1 (Benign)" in line:
                source[j] = line.replace(
                    "# Map target to 0 (Malignant) and 1 (Benign)",
                    "# Map target: 1 (Malignant) and 0 (Benign)"
                )
                print(f"Fix 1 applied: target comment (cell {i})")

    # --- Fix 2: "Feature Scaling" comment (execution_count 3) ---
    if ec == 3:
        for j, line in enumerate(source):
            if "# Feature Scaling" in line:
                source[j] = line.replace(
                    "# Feature Scaling",
                    "# Separate features and target"
                )
                print(f"Fix 2 applied: feature scaling comment (cell {i})")

    # --- Fix 3 & 5: K search range AND preserve raw data (execution_count 5) ---
    # This cell does train_test_split + scaling + heuristic K
    if ec == 5:
        new_source = []
        for j, line in enumerate(source):
            new_source.append(line)
        # Add raw data preservation after the split but before scaling
        # Find the split line and insert raw preservation after it
        rebuilt = []
        for line in new_source:
            rebuilt.append(line)
            if "X_train, X_test, y_train, y_test = train_test_split" in line:
                rebuilt.append("# Preserve raw (unscaled) copies for Pipeline-based evaluation later\n")
                rebuilt.append("X_train_raw, X_test_raw = X_train.copy(), X_test.copy()\n")
        source.clear()
        source.extend(rebuilt)
        print(f"Fix 5 applied: preserve raw train/test data (cell {i})")

    # --- Fix 3: K search range (execution_count 6) ---
    if ec == 6:
        for j, line in enumerate(source):
            if "k_values = range(max(1, heuristic_k - 10), heuristic_k + 11, 2)" in line:
                source[j] = line.replace(
                    "k_values = range(max(1, heuristic_k - 10), heuristic_k + 11, 2) # Checking odd Ks around heuristic",
                    "k_values = range(max(1, heuristic_k - 5), heuristic_k + 6, 1) # Checking Ks around heuristic"
                )
                print(f"Fix 3 applied: K search range ±5 step 1 (cell {i})")

    # --- Fix 4a: Add comment to CV Pipeline (execution_count 8) ---
    if ec == 8:
        for j, line in enumerate(source):
            if "for k in k_values:" in line and j > 0:
                source.insert(j, "# Using Pipeline to properly scale within each fold and avoid data leakage\n")
                print(f"Fix 4a applied: CV pipeline comment (cell {i})")
                break

    # --- Fix 4b: Final model uses Pipeline (execution_count 9) ---
    if ec == 9:
        new_source = [
            "# Using Pipeline for consistency with cross-validation (scales within the pipeline)\n",
            "final_k = optimal_k_cv\n",
            "final_pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=final_k))])\n",
            "final_pipeline.fit(X_train_raw, y_train)\n",
            "y_pred = final_pipeline.predict(X_test_raw)\n",
            "y_prob = final_pipeline.predict_proba(X_test_raw)[:, 1]\n",
        ]
        # Keep everything after the original y_prob line
        found_y_prob = False
        for line in source:
            if found_y_prob:
                new_source.append(line)
            if "y_prob = " in line:
                found_y_prob = True
        source.clear()
        source.extend(new_source)
        print(f"Fix 4b applied: final model uses Pipeline (cell {i})")

# Clear all outputs so the notebook needs to be re-run
for cell in cells:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("\nAll fixes applied. Outputs cleared — please re-run the notebook.")

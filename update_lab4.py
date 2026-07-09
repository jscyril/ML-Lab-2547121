import json

path = '/mnt/data/college/4_tri/ML-Lab-2547121/notebooks/Lab_4.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Update Mapping
        for i, line in enumerate(cell['source']):
            if "df['target'] = df['y'].map({'M': 0, 'B': 1})" in line:
                cell['source'][i] = "df['target'] = df['y'].map({'M': 1, 'B': 0})\n"
            
            # Update Colormap
            if "cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])" in line:
                cell['source'][i] = "    cmap_light = ListedColormap(['#AAFFAA', '#FFAAAA'])\n"
            if "cmap_bold = ListedColormap(['#FF0000', '#00FF00'])" in line:
                cell['source'][i] = "    cmap_bold = ListedColormap(['#00FF00', '#FF0000'])\n"
            
            # Update Confusion Matrix labels
            if "xticklabels=['Malignant(0)', 'Benign(1)'], yticklabels=['Malignant(0)', 'Benign(1)']" in line:
                cell['source'][i] = line.replace("xticklabels=['Malignant(0)', 'Benign(1)'], yticklabels=['Malignant(0)', 'Benign(1)']", "xticklabels=['Benign(0)', 'Malignant(1)'], yticklabels=['Benign(0)', 'Malignant(1)']")

    # Update Conclusion Markdown
    if cell['cell_type'] == 'markdown':
        for i, line in enumerate(cell['source']):
            if "- **Model Performance:**" in line:
                cell['source'][i] = "- **Model Performance:** The final KNN model achieved strong overall accuracy (~94.7%), but the specific metrics for Malignant detection (Precision: 100.0%, Recall: ~86.0%, F1 Score: ~92.5%) reveal that it missed 14% of the cancerous cases. This emphasizes that plain accuracy is misleading, and optimizing for Recall is critical in medical diagnostics to avoid life-threatening False Negatives.\n"

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

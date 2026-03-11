# Isotope Classifier - K-Nearest Neighbors Machine Learning

A machine learning project using KNN to classify gamma-ray spectra and identify isotopes.

## About

This project uses the K-Nearest Neighbors (KNN) algorithm to classify radioactive isotopes based on their gamma-ray spectra. The model is trained on spectra from 5 different isotopes with 10 samples each.

## Files

- `isotope_classifier_knn_only.py` - Main Python script
- `spectra/` - Folder containing CSV files with gamma-ray spectra

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## How to Use

1. Place all CSV spectrum files in the `spectra/` folder
2. Run the script:

```bash
python3 isotope_classifier_knn_only.py
```

3. The script will:
   - Load all spectra from the `spectra/` folder
   - Train a KNN model
   - Evaluate accuracy
   - Generate confusion matrix
   - Find optimal k value
   - Save the trained model

## Output

- `knn_confusion_matrix.png` - Confusion matrix visualization
- `knn_k_optimization.png` - K value optimization chart
- `knn_isotope_model.pkl` - Trained model file

## Author

petrabrown

## License

MIT

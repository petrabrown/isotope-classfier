import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import glob
import os

# ============================================================
# LOAD MULTIPLE CSV FILES (gamma-ray spectra)
# ============================================================

def load_spectrum_data(csv_directory="./spectra/"):
    """
    Load all gamma-ray spectrum CSV files from a directory.
    Handles the metadata header and extracts only the Channel Data section.
    """
    X = []
    y = []
    
    # Find all CSV files in directory
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in '{csv_directory}'")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Please create a 'spectra' folder and add CSV files there.")
        return None, None
    
    print(f"✓ Found {len(csv_files)} spectrum files\n")
    
    for filepath in csv_files:
        try:
            # Read entire file
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Find where "Channel Data:" starts
            data_start_idx = None
            for i, line in enumerate(lines):
                if "Channel Data:" in line:
                    data_start_idx = i + 2  # Skip "Channel Data:" and header row
                    break
            
            if data_start_idx is None:
                print(f"⚠ Skipped {os.path.basename(filepath)}: No 'Channel Data:' section found")
                continue
            
            # Read only the data section
            data = pd.read_csv(filepath, skiprows=data_start_idx)
            
            # Extract counts as feature vector
            if 'Counts' in data.columns:
                spectrum = data['Counts'].values.astype(float)
                
                # Extract isotope label from filename
                filename = os.path.basename(filepath)
                isotope_label = filename.replace('.csv', '').strip()
                
                X.append(spectrum)
                y.append(isotope_label)
                
                print(f"✓ Loaded: {isotope_label} ({len(spectrum)} channels, {spectrum.sum():.0f} total counts)")
            else:
                print(f"⚠ Skipped {os.path.basename(filepath)}: No 'Counts' column found")
                
        except Exception as e:
            print(f"❌ Error loading {os.path.basename(filepath)}: {str(e)}")
    
    if not X:
        print("\n❌ No valid spectra loaded!")
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✓ Successfully loaded {len(X)} spectra")
    print(f"  Isotope labels: {np.unique(y)}")
    print(f"  Spectrum length: {X.shape[1]} channels")
    print(f"  Data shape: {X.shape}\n")
    
    return X, y


# ============================================================
# MAIN KNN CLASSIFIER PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("  K-NEAREST NEIGHBORS ISOTOPE CLASSIFIER")
    print("=" * 70)
    print()
    
    # Load spectra from CSV files
    print("Loading spectra from ./spectra/ directory...\n")
    X, y = load_spectrum_data("./spectra/")
    
    if X is None:
        print("Exiting: No data loaded")
        return
    
    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)
    print()
    
    # Split data into training and testing sets
    # stratify=y ensures each class is represented in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} spectra")
    print(f"Test set: {len(X_test)} spectra")
    print()
    
    # ============================================================
    # K-NEAREST NEIGHBORS MODEL
    # ============================================================
    print("=" * 70)
    print("K-NEAREST NEIGHBORS (KNN) MODEL")
    print("=" * 70)
    print()
    
    # Create pipeline: scale features, then apply KNN
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform'))
    ])
    
    print("Training KNN model (k=5)...")
    knn_pipeline.fit(X_train, y_train)
    print("✓ Model trained successfully\n")
    
    # Make predictions on test set
    y_pred = knn_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print()
    print(f"Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    
    # ============================================================
    # CROSS-VALIDATION
    # ============================================================
    print("=" * 70)
    print("CROSS-VALIDATION (5-fold)")
    print("=" * 70)
    print()
    
    cv_scores = cross_val_score(knn_pipeline, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print()
    
    # ============================================================
    # CONFUSION MATRIX
    # ============================================================
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print()
    
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print()
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=knn_pipeline.named_steps['knn'].classes_,
                yticklabels=knn_pipeline.named_steps['knn'].classes_)
    plt.title(f'KNN Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.ylabel('True Isotope')
    plt.xlabel('Predicted Isotope')
    plt.tight_layout()
    plt.savefig('knn_confusion_matrix.png', dpi=300)
    print("✓ Saved: knn_confusion_matrix.png")
    print()
    
    # ============================================================
    # FIND OPTIMAL K VALUE
    # ============================================================
    print("=" * 70)
    print("FINDING OPTIMAL K VALUE")
    print("=" * 70)
    print()
    
    best_score = 0
    best_k = 1
    scores = []
    
    for k in range(1, 16):
        knn_test = KNeighborsClassifier(n_neighbors=k)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        knn_test.fit(X_train_scaled, y_train)
        score = knn_test.score(X_test_scaled, y_test)
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_k = k
        
        print(f"k={k:2d}: {score:.4f}")
    
    print(f"\n✓ Best k value: {best_k} with accuracy {best_score:.4f}")
    print()
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16), scores, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('KNN: Accuracy vs k Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('knn_k_optimization.png', dpi=300)
    print("✓ Saved: knn_k_optimization.png")
    print()
    
    # ============================================================
    # PREDICT ON NEW SPECTRUM
    # ============================================================
    print("=" * 70)
    print("PREDICT ON NEW SPECTRUM")
    print("=" * 70)
    print()
    
    # Load a test spectrum file
    try:
        test_file = "./spectra/Eu-152 1.csv"
        
        if not os.path.exists(test_file):
            print(f"⚠ File not found: {test_file}")
            print("Skipping prediction example.")
        else:
            with open(test_file, 'r') as f:
                lines = f.readlines()
            
            # Find where "Channel Data:" starts
            data_start_idx = None
            for i, line in enumerate(lines):
                if "Channel Data:" in line:
                    data_start_idx = i + 2
                    break
            
            if data_start_idx:
                new_data = pd.read_csv(test_file, skiprows=data_start_idx)
                
                if 'Counts' in new_data.columns:
                    new_spectrum = new_data['Counts'].values.astype(float)
                    new_features = new_spectrum.reshape(1, -1)
                    
                    # Predict
                    prediction = knn_pipeline.predict(new_features)[0]
                    
                    # Get distances and neighbors
                    distances, indices = knn_pipeline.named_steps['knn'].kneighbors(new_features)
                    neighbors = y_train[indices[0]]
                    
                    print(f"File: Eu-152 1.csv")
                    print(f"Predicted Isotope: {prediction}")
                    print()
                    print(f"5 Nearest Neighbors (with distances):")
                    for i, (neighbor, distance) in enumerate(zip(neighbors, distances[0])):
                        print(f"  {i+1}. {neighbor} (distance: {distance:.4f})")
                    print()
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
    
    print()
    print("=" * 70)
    print("DONE!")
    print("=" * 70)
    
    # ============================================================
    # SAVE MODEL
    # ============================================================
    import joblib
    joblib.dump(knn_pipeline, 'knn_isotope_model.pkl')
    print("\n✓ Model saved: knn_isotope_model.pkl")
    print()
    print("To load and use the model later:")
    print("  import joblib")
    print("  model = joblib.load('knn_isotope_model.pkl')")
    print("  prediction = model.predict(new_spectrum)")


if __name__ == "__main__":
    main()

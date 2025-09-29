# Titanic TensorFlow.js — Fixes and Improvements

This application predicts Titanic passenger survival using a TensorFlow.js neural network. Several fixes were made to ensure stable and correct functionality from data loading to exporting predictions.

---

## Key Changes and Fixes

### 1. Data Loading and Preprocessing

- **Issue:**  
  The original CSV parser was simplistic, failing to handle quotes, commas, and BOM. Numeric and categorical features were not normalized.

- **Fix:**  
  - Implemented a robust CSV parser in `app.js` (`parseCSV` and `loadData`) handling CSV formatting correctly.  
  - Added normalization for features like `Survived`, `Pclass`, `Sex`, and `Embarked`.

- **Impact:**  
  Fixed empty charts issue during data inspection.

---

### 2. Evaluation Metrics

- **Issue:**  
  ROC points were unsorted, causing incorrect AUC calculations, sometimes negative (e.g., -0.90).

- **Fix:**  
  - Sorted ROC points before plotting.  
  - Ensured AUC is always positive.

- **Impact:**  
  Accurate ROC curve and meaningful AUC scores.

---

### 3. Model Predictions

- **Issue:**  
  Predictions were nested arrays (`[[0.8],[0.3], ...]`), causing errors like “toFixed is not a function” during table rendering.

- **Fix:**  
  - Flattened predictions to simple numbers.  
  - Updated table rendering accordingly.

- **Impact:**  
  Correct display of survival probabilities without errors.

---

### 4. Export Functionality

- **Issue:**  
  CSV export could be incorrect; no option to save the model.

- **Fix:**  
  - Flattened and formatted predictions before CSV export.  
  - Added model saving via `model.save`.

- **Impact:**  
  Proper CSV exports and downloadable model files.

---

## Summary

After these fixes, the app works correctly at all stages:

- Reliable data loading and normalization  
- Correct evaluation metric calculation  
- Accurate predictions display  
- Full export capability with CSV and model download

These improvements enhance the app's stability and user experience.

---

Ready to provide code examples or detailed explanations on request.

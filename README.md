# Premier League Match Prediction Using Machine Learning

This project explores the effects of different preprocessing techniques on machine learning model performance in predicting the outcomes of Premier League soccer matches. By analyzing historical data and team performance statistics, the project employs various machine learning models to forecast match results and identify key predictors of match outcomes.

---

## Features
- **Data Collection**: Combines datasets from `football-data.co.uk` and `STATHEAD - FBREF` for the 2017/18 to 2023/24 seasons.
- **Preprocessing Techniques**: Includes one-hot encoding, correlation filtering, PCA, and LDA for dimensionality reduction.
- **Model Implementations**:
  - Support Vector Machine (SVM)
  - Random Forest (RF)
  - Decision Tree
  - Deep Neural Networks (e.g., VGG-like, ResNet, LSTM, GRU)
  - Baseline Random Guess Model
- **Evaluation Metrics**: Accuracy, F1-score, precision, recall, and confusion matrices.

---

## How It Works
### Data Preprocessing
1. **Combining Datasets**: Merges multiple CSV files to create a unified dataset.
2. **Cleaning**: Removes missing or NaN values and irrelevant columns.
3. **Transformation**: Applies one-hot encoding to categorical variables and filters features using correlation analysis.
4. **Dimensionality Reduction**: Employs PCA and LDA to reduce the dataset’s dimensionality while retaining key features.

### Machine Learning Models
- **Baseline Model**: Randomly guesses match outcomes to establish a performance baseline.
- **Traditional Models**: Includes SVM, Random Forest, and Decision Tree, optimized with grid search.
- **Deep Neural Networks**:
  - **VGG-like Networks**: Fully connected layers with dropout for regularization.
  - **ResNet**: Dense layers and residual connections to improve performance on complex data.
  - **Sequential Models**: LSTM and GRU for time-series analysis, with bidirectional variants for enhanced context understanding.

---

## Performance Comparison
| Model                  | Regular Dataset | Filtered Dataset | PCA Dataset | LDA Dataset |
|------------------------|-----------------|------------------|-------------|-------------|
| **SVM**               | 0.848          | 0.998            | 0.788       | 0.561       |
| **Random Forest**     | 0.910          | 0.940            | 0.750       | 0.753       |
| **Decision Tree**     | 0.759          | 0.993            | 0.570       | 0.545       |
| **ResNet**            | 0.952          | 0.869            | 0.910       | 0.807       |
| **VGG-like Network**  | 0.811          | -                | 0.829       | 0.545       |
| **LSTM**              | 0.785          | -                | -           | -           |

### Key Observations
- Filtered datasets significantly improved traditional models but reduced the accuracy of neural networks.
- PCA and LDA preprocessing did not enhance accuracy and, in some cases, worsened it.
- ResNet performed best overall, achieving the highest F1 score of 0.952 on the regular dataset.

---

## Requirements
- **Environment**: Python 3.x
- **Libraries**: `numpy`, `pandas`, `sklearn`, `tensorflow`, `keras`

---

## Future Enhancements
- **Feature Engineering**: Incorporate advanced metrics and external data sources.
- **Hybrid Models**: Combine traditional and deep learning models for improved predictions.
- **Explainability**: Use SHAP or LIME to interpret model predictions.

---

For questions or feedback, please open an issue or contact the authors directly.

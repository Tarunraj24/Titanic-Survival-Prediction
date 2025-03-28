# Titanic Survival Prediction

## Overview
This project implements a machine learning model to predict the survival of passengers on the Titanic using the Titanic dataset. The model is built using Python and scikit-learn, and employs a Random Forest Classifier for prediction.

## Dataset
The dataset used for this project is assumed to be `titanic.csv`, which contains information about passengers, including their age, fare, class, and whether they survived the disaster.

## Preprocessing Steps
1. **Handling Missing Values**:
   - `Age` was imputed using the median.
   - `Fare` was imputed using the median.
   - `Embarked` was imputed using the mode.
   - Columns `Cabin`, `Ticket`, `Name`, and `PassengerId` were dropped as they were deemed irrelevant.

2. **Encoding Categorical Variables**:
   - `Sex` and `Embarked` columns were label-encoded.

3. **Feature Selection**:
   - The features selected for training were `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.

4. **Data Splitting**:
   - The dataset was split into training and testing sets (80% train, 20% test) using stratified sampling.

5. **Feature Scaling**:
   - The numerical features `Age` and `Fare` were standardized using `StandardScaler`.

## Model Selection
- The chosen model is **Random Forest Classifier** with the following hyperparameters:
  - `n_estimators=200`
  - `max_depth=7`
  - `min_samples_split=5`
  - `min_samples_leaf=3`
  - `random_state=42`
- The model was trained using the training data and evaluated on the test data.

## Performance Analysis
- **Cross-validation Accuracy**: The model achieved an average accuracy score from 5-fold cross-validation.
- **Test Accuracy**: The accuracy of the model on the test dataset was computed.
- **Classification Report**: Precision, recall, and F1-score were analyzed for both survival classes.
- **Confusion Matrix**: A confusion matrix was plotted to visualize model performance.
- **Feature Importance**: The importance of each feature was visualized using a bar chart.

## Results
- The model performed well in predicting survival, showing a balance between precision and recall.
- Feature importance analysis indicated that `Sex`, `Fare`, and `Pclass` were the most significant factors influencing survival.

## Usage
1. Ensure `titanic.csv` is present in the project directory.
2. Install required dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
3. Run the script:
   ```bash
   python titanic_survival.py
   ```

## Visualizations
- Confusion Matrix
- Feature Importance Plot


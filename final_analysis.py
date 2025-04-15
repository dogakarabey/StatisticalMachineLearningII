import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the youth drug survey dataset and display basic information"""
    print("\n[INFO] Loading Youth Drug Survey Dataset...")
    df = pd.read_csv(file_path, delimiter=',')
    print(f"[SUCCESS] Data loaded successfully!")
    print(f"[DATA] Dataset contains {df.shape[0]} participants and {df.shape[1]} features")
    print("\n[FEATURES] Key Features:")
    print("- Alcohol use frequency (IRALCFY)")
    print("- Demographic information (NEWRACE2, INCOME, HEALTH2, IRSEX)")
    print("- Family factors (PARHLPHW, PRLMTTV2, PARCHKHW, PRCHORE2)")
    print("- School-related factors (AVGGRADE, SCHFELT, TCHGJOB)")
    print("- Behavioral indicators (YOFIGHT2, YOGRPFT2, YOSTOLE2, YOATTAK2)")
    print("- Peer influence (FRDMEVR2, FRDMJMON, FRDADLY2, FRDPCIG2)")
    print("- Risk perception (STNDALC, STNDSMJ, STNDSCIG, STNDDNK)")
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and special codes"""
    print("\n[INFO] Cleaning Dataset...")
    
    # Replace special codes with NaN (these represent missing/unknown values)
    special_codes = [91, 93, 94, 97, 98, 99]
    df = df.replace(special_codes, np.nan)
    print("[SUCCESS] Replaced special codes with missing values")
    
    # Drop columns with excessive missing values
    missing_threshold = 0.3
    cols_to_drop = df.columns[df.isnull().mean() > missing_threshold].tolist()
    if cols_to_drop:
        print(f"[WARNING] Dropping {len(cols_to_drop)} features with >{missing_threshold*100}% missing values")
        print("Dropped features:", cols_to_drop)
    else:
        print("[SUCCESS] No features exceeded missing value threshold")
    df = df.drop(columns=cols_to_drop)
    
    return df

def prepare_data(df):
    """Prepare the data for different types of analysis"""
    print("\n[INFO] Preparing Data for Analysis...")
    
    # Create binary target (use vs. no use)
    df['alcohol_use_binary'] = df['IRALCFY'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
    print("[SUCCESS] Created binary target: Use (1) vs. No Use (0)")
    
    # Create multi-class target (levels of use)
    df['alcohol_use_multi'] = df['IRALCFY'].apply(lambda x: {
        0: 0,  # No use
        1: 1,  # Light use
        2: 2,  # Moderate use
        3: 3   # Heavy use
    }.get(x, 0))
    print("[SUCCESS] Created multi-class target: No Use (0), Light (1), Moderate (2), Heavy (3)")
    
    # Keep original values for regression
    df['alcohol_use_regression'] = df['IRALCFY']
    print("[SUCCESS] Prepared original values for regression analysis")
    
    # Define feature groups for better organization
    feature_groups = {
        'demographic': ['NEWRACE2', 'INCOME', 'HEALTH2', 'IRSEX'],
        'family': ['PARHLPHW', 'PRLMTTV2', 'PARCHKHW', 'PRCHORE2'],
        'school': ['AVGGRADE', 'SCHFELT', 'TCHGJOB'],
        'behavioral': ['YOFIGHT2', 'YOGRPFT2', 'YOSTOLE2', 'YOATTAK2'],
        'peer': ['FRDMEVR2', 'FRDMJMON', 'FRDADLY2', 'FRDPCIG2'],
        'risk_perception': ['STNDALC', 'STNDSMJ', 'STNDSCIG', 'STNDDNK']
    }
    
    # Combine all features
    all_features = []
    for group in feature_groups.values():
        all_features.extend(group)
    
    print("\n[FEATURES] Feature Groups:")
    for group_name, features in feature_groups.items():
        print(f"- {group_name.capitalize()}: {len(features)} features")
    
    # Create feature matrix and targets
    X = df[all_features]
    y_binary = df['alcohol_use_binary']
    y_multi = df['alcohol_use_multi']
    y_regression = df['alcohol_use_regression']
    
    return X, y_binary, y_multi, y_regression, all_features

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """Create a preprocessing pipeline for numerical and categorical features"""
    print("\n[INFO] Creating Preprocessing Pipeline...")
    
    # Handle numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    print(f"[SUCCESS] Added numerical preprocessing for {len(numerical_features)} features")
    
    # Handle categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    print(f"[SUCCESS] Added categorical preprocessing for {len(categorical_features)} features")
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def plot_decision_tree(model, feature_names, class_names, title):
    """Create a visualization of the decision tree"""
    print("\n[INFO] Creating Decision Tree Visualization...")
    plt.figure(figsize=(20,10))
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('decision_tree.png')
    plt.close()
    print("[SUCCESS] Decision tree saved as 'decision_tree.png'")

def plot_confusion_matrix(y_true, y_pred, title):
    """Create a confusion matrix visualization"""
    print(f"\n[INFO] Creating Confusion Matrix for {title}...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
    plt.close()
    print(f"[SUCCESS] Confusion matrix saved as 'confusion_matrix_{title.lower().replace(' ', '_')}.png'")

def train_and_evaluate_models(X, y, task_type, preprocessor):
    """Train and evaluate models for the specified task"""
    print(f"\n[INFO] Training Models for {task_type} Task...")
    
    # Split data with stratification for classification tasks
    if task_type != 'regression':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    print(f"[SUCCESS] Split data into training ({len(X_train)} samples) and test ({len(X_test)} samples) sets")
    
    # Define models based on task type
    if task_type == 'binary':
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=3),
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        print("\n[MODELS] Binary Classification Models:")
        print("- Decision Tree (max_depth=3 for interpretability)")
        print("- Random Forest (ensemble of decision trees)")
        print("- Gradient Boosting (sequential improvement)")
    elif task_type == 'multi':
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced')
        }
        print("\n[MODELS] Multi-class Classification Models:")
        print("- Decision Tree")
        print("- Random Forest")
    else:  # regression
        models = {
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        print("\n[MODELS] Regression Models:")
        print("- Decision Tree")
        print("- Gradient Boosting")
    
    # Define parameter grids for hyperparameter tuning
    param_grids = {
        'Decision Tree': {
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        },
        'Random Forest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        },
        'Gradient Boosting': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Grid search with balanced accuracy scoring for classification
        scoring = 'balanced_accuracy' if task_type != 'regression' else 'r2'
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[name], 
            cv=5,
            scoring=scoring,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        print(f"[SUCCESS] {name} training completed")
        print(f"[PARAMS] Best parameters: {grid_search.best_params_}")
        
        # Get predictions
        y_pred = grid_search.predict(X_test)
        
        # For binary classification, plot the decision tree
        if task_type == 'binary' and name == 'Decision Tree':
            fitted_model = grid_search.best_estimator_.named_steps['model']
            feature_names = X.columns.tolist()
            class_names = ['No Use', 'Use']
            plot_decision_tree(fitted_model, feature_names, class_names, "Decision Tree for Alcohol Use Prediction")
        
        # Calculate metrics
        if task_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {
                'mse': mse,
                'r2': r2,
                'params': grid_search.best_params_
            }
            print(f"[METRICS] Regression Metrics:")
            print(f"- MSE: {mse:.3f}")
            print(f"- R²: {r2:.3f}")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred, f"{name} - {task_type}")
            
            results[name] = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'classification_report': classification_report(y_test, y_pred),
                'params': grid_search.best_params_
            }
            print(f"[METRICS] Classification Metrics:")
            print(f"- Accuracy: {accuracy:.3f}")
            print(f"- Balanced Accuracy: {balanced_accuracy:.3f}")
            print(f"- Precision: {precision:.3f}")
            print(f"- Recall: {recall:.3f}")
            print(f"- F1 Score: {f1:.3f}")
    
    return results

def main():
    """Main function to run the complete analysis"""
    print("\n" + "="*50)
    print("Youth Drug Use Analysis")
    print("="*50)
    
    # Load and prepare data
    df = load_data('youth_drug_analysis/youth_data.csv')
    df = clean_data(df)
    X, y_binary, y_multi, y_regression, all_features = prepare_data(df)
    
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Train and evaluate models for each task
    print("\n" + "="*50)
    print("Binary Classification Analysis")
    print("="*50)
    binary_results = train_and_evaluate_models(X, y_binary, 'binary', preprocessor)
    
    print("\n" + "="*50)
    print("Multi-class Classification Analysis")
    print("="*50)
    multi_results = train_and_evaluate_models(X, y_multi, 'multi', preprocessor)
    
    print("\n" + "="*50)
    print("Regression Analysis")
    print("="*50)
    regression_results = train_and_evaluate_models(X, y_regression, 'regression', preprocessor)
    
    # Print final summary
    print("\n" + "="*50)
    print("Final Results Summary")
    print("="*50)
    
    print("\nBinary Classification Results:")
    for name, result in binary_results.items():
        print(f"\n{name}:")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"Balanced Accuracy: {result['balanced_accuracy']:.3f}")
        print(f"Precision: {result['precision']:.3f}")
        print(f"Recall: {result['recall']:.3f}")
        print(f"F1 Score: {result['f1']:.3f}")
        print("Best Parameters:", result['params'])
    
    print("\nMulti-class Classification Results:")
    for name, result in multi_results.items():
        print(f"\n{name}:")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"Balanced Accuracy: {result['balanced_accuracy']:.3f}")
        print(f"Precision: {result['precision']:.3f}")
        print(f"Recall: {result['recall']:.3f}")
        print(f"F1 Score: {result['f1']:.3f}")
        print("Classification Report:")
        print(result['classification_report'])
        print("Best Parameters:", result['params'])
    
    print("\nRegression Results:")
    for name, result in regression_results.items():
        print(f"\n{name}:")
        print(f"MSE: {result['mse']:.3f}")
        print(f"R² Score: {result['r2']:.3f}")
        print("Best Parameters:", result['params'])
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("="*50)

if __name__ == "__main__":
    main() 
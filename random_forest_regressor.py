import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import random
from feature_engineering import create_features,select_features
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support

def evaluate_classification(y_test, y_pred ,title):
    class_labels = ['Low mood', 'High mood']
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    # Compute F1 score specifically for Class 1
    _, _, f1_scores, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    f1_class1 = f1_scores[1]  # F1 score for Class 1
    
    f1 = f1_score(y_test, y_pred, average='weighted') 
    
    print(f'Accuracy score: {accuracy:.2f}')
    print(f'F1 score: {f1:.2f}')
    print("Classification Report:")
    class_report = classification_report(y_test, y_pred, target_names=class_labels)
    print(class_report)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))  # Set the figure size
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    ax = disp.plot(cmap=plt.cm.Blues).ax_  # Using a blue color map for the confusion matrix
    ax.set_title(title)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    
    # Adjust the text annotation below the plot
    plt.gcf().text(0.5, 0.02, f'Accuracy score: {accuracy:.2f} | F1 score: {f1:.2f} | F1 score for Class 1: {f1_class1:.2f}', ha='center', fontsize=12)
                  
    plt.show()

    return class_report



def evaluate_regression(y_test, y_pred, model, title):
    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error

    # Plotting predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue', label='Predictions')
    
    # Fit line to the data
    # Fit a linear polynomial (degree 1) to the data
    z = np.polyfit(y_test.flatten(), y_pred, 1)
    print(type(z))
    print(z.shape)
    p = np.poly1d(z)
    
    # Plot the fitted line over the range of y_test values
    plt.plot(y_test, p(y_test), "r--", label=f'Fitted Line: y={z[0]:.2f}x+{z[1]:.2f}')
    
    plt.title(title)
    plt.xlabel('Actual Mood')
    plt.ylabel('Predicted Mood')
    plt.legend()

    # Add text for MSE and MAE inside the plot
    plt.text(0.05, 0.05, f'MSE: {mse:.2f}\nMAE: {mae:.2f}', transform=plt.gca().transAxes, fontsize=12,
              bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.9))

    plt.show()



def evaluate_regression_two_preds(y_test, y_pred1, y_pred2, title):
    # Compute metrics for the first set of predictions
    mse1 = mean_squared_error(y_test, y_pred1)
    mae1 = mean_absolute_error(y_test, y_pred1)
    rmse1 = np.sqrt(mse1)  # Root Mean Squared Error

    # Compute metrics for the second set of predictions
    mse2 = mean_squared_error(y_test, y_pred2)
    mae2 = mean_absolute_error(y_test, y_pred2)
    rmse2 = np.sqrt(mse2)  # Root Mean Squared Error

    # Plotting predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred1, alpha=0.3, color='blue', label='Predictions 1')
    plt.scatter(y_test, y_pred2, alpha=0.3, color='green', label='Predictions 2')
    
    # Fit lines to the data for both predictions
    z1 = np.polyfit(y_test.flatten(), y_pred1, 1)
    p1 = np.poly1d(z1)
    plt.plot(y_test, p1(y_test), "r--", label=f'Fitted Line 1: y={z1[0]:.2f}x+{z1[1]:.2f}')

    z2 = np.polyfit(y_test.flatten(), y_pred2, 1)
    p2 = np.poly1d(z2)
    plt.plot(y_test, p2(y_test), "m--", label=f'Fitted Line 2: y={z2[0]:.2f}x+{z2[1]:.2f}')
    
    plt.title(title)
    plt.xlabel('Actual Mood')
    plt.ylabel('Predicted Mood')
    plt.legend(loc='upper left')

    #Add text for MSE and MAE inside the plot for both predictions
    plt.text(0.05, 0.10, f'MSE1: {mse1:.2f}, MAE1: {mae1:.2f}\nMSE2: {mse2:.2f}, MAE2: {mae2:.2f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.9),
             verticalalignment='top')

    plt.show()



def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    max_depth = trial.suggest_int('max_depth', 2, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])  # Choice of criterion
    
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=60
    )

    #select first for MSE second for MAE
    #score = cross_val_score(model, X_train, y_train.flatten(), cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    score = cross_val_score(model, X_train, y_train.flatten(), cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

    return -score 


#SEED = np.random.randint(100)
SEED = 60
np.random.seed(SEED), random.seed(SEED)   
print(f"SEED: {SEED}") 
#pd.set_option('future.no_silent_downcasting', True)
dataset = pd.read_csv('./cleaned_dataset.csv')
# Creating features
X, y = create_features(dataset,window_size=3)
# Splitting in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
# Selecting optimal features
features = select_features(X_train, y_train, SEED)
print(features)
X_train, X_test = X_train[features], X_test[features]

y_train, y_test = np.array(y_train), np.array(y_test)



# Setup and train RandomForestRegressor
model = RandomForestRegressor(random_state=42,
                              criterion = 'absolute_error')
model.fit(X_train, y_train.flatten())

# Predict and evaluate
y_pred = model.predict(X_test)
evaluate_regression(np.array(y_test),np.array( y_pred), model, 'Regression Base Model Performance')



# Optuna for hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best parameters found
print("Best trial:")
best_params = study.best_trial.params
print(f"Best MSE: {study.best_trial.value:.4f}")
print("Best Params:", best_params)




# Initialize and fit the first model with squared_error
model_optimized = RandomForestRegressor(**best_params, random_state=60)
model_optimized.fit(X_train, y_train.flatten())
y_pred_optimized = model_optimized.predict(X_test)

# Initialize and fit the second model with absolute_error
absolute_error_params = best_params.copy()
absolute_error_params['criterion'] = 'absolute_error'

new_model = RandomForestRegressor(**absolute_error_params, random_state=60)
new_model.fit(X_train, y_train.flatten())
y_pred_new_model = new_model.predict(X_test)

title1 = 'MSE vs MAE Random Forest Regression Model Results'
evaluate_regression_two_preds(y_test, y_pred_optimized, y_pred_new_model, title1)

# Plotting the actual vs predicted mood for both models
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_optimized, alpha=0.5, color='red', label='Predicted (Squared Error)')
plt.scatter(y_test, y_pred_new_model, alpha=0.5, color='blue', label='Predicted (Absolute Error)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Fitted line
plt.xlabel('Actual Mood')
plt.ylabel('Predicted Mood')
plt.title('Comparison of Prediction Accuracies')
plt.legend()
plt.show()

evaluate_regression(np.array(y_test), np.array(y_pred_optimized), model_optimized, 'Optimized Regression Model Performance')


#classification  of the regression result
y_pred_binary = (y_pred > 7.5).astype(int)
y_test_binary = (y_test > 7.5).astype(int)
title = 'Random Forest Regressor Confusion Matrix'
evaluate_classification(y_test_binary, y_pred_binary, title)










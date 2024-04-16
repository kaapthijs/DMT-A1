import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from feature_engineering import create_features,select_features
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score,make_scorer      
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff


def evaluate(y_test, y_pred, model,title):
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='weighted')  
    # recall = recall_score(y_test, y_pred, average='weighted')        
    f1 = f1_score(y_test, y_pred, average='weighted') 
    
    print(f'Accuracy score: {accuracy:.2f}')
    # print(f'Precision score: {precision:.2f}')
    # print(f'Recall score: {recall:.2f}')
    print(f'F1 score: {f1:.2f}')
    print("Classification Report:")
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    #plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    ax = disp.plot().ax_
    ax.set_title(title)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.gcf().text(0.5, 0.02, f'Accuracy score: {accuracy:.2f}  |  F1 score: {f1:.2f}', ha='center', fontsize=12)
    plt.show()

#define optuna trial object for hyperparamater tuning
def objective(trial):
    #define the range of parameters for hyperparameter tuning
    n_estimators = trial.suggest_int('n_estimators',10,1000)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    
    model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    class_weight= class_weight,
    random_state= 42
    )
    
    # F1 Score as the scoring metric
    f1_scorer = make_scorer(f1_score, average='weighted')  # 'weighted' to handle class imbalance
    f1 = cross_val_score(model, X_train, y_train, cv=5, scoring=f1_scorer, n_jobs=-1).mean()
    
    return f1
    
   
    
    
#pd.set_option('future.no_silent_downcasting', True)
dataset = pd.read_csv('./cleaned_dataset.csv')
# Creating features
X, y = create_features(dataset,window_size=3)
# Splitting in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Selecting optimal features
features = select_features(X_train, y_train)
X_train, X_test = X_train[features], X_test[features]




#KBinsDiscritizer to convert labels to classes
est1 = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
y_train = est1.fit_transform(y_train)
y_test = est1.transform(y_test)

# Convert float to int
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print(f'treshold for splitting target as high and low = {est1.bin_edges_[0]}')


#Random Forest set up
model = RandomForestClassifier(n_estimators=100,
                                       random_state=42,
                                       max_depth=5,
                                       class_weight= 'balanced',
                                       )
model.fit(X_train, y_train.flatten())

#Predict
y_pred = model.predict(X_test)
#evaluate and plot the confusion matrix 
evaluate(y_test,y_pred,model,'Confusion Matrix for Base Model')


##### Use optuna for hyperparamater search
study = optuna.create_study(direction = 'maximize', sampler = optuna.samplers.RandomSampler(seed=42))
study.optimize(objective, n_trials=200)

# Print the best parameters found 
print("Best trial:")
trial = study.best_trial


print("Value: {:.4f}".format(trial.value))

print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# Initialize the model with the best hyperparameters
best_params = study.best_trial.params
model = RandomForestClassifier(**best_params, random_state=42)

#train the model with tuned params
model.fit(X_train, y_train.flatten())
y_pred = model.predict(X_test)
evaluate(y_test,y_pred,model,'Confusion Matrix for Tuned Model')




































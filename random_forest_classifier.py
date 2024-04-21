import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from feature_engineering import create_features,select_features
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score,make_scorer,precision_recall_fscore_support   
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import random

import numpy as np
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar



def evaluate(y_test, y_pred, model,title):
    
    class_labels = ['Low mood', 'High mood']
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    # Compute F1 score specifically for Class 1
    _, _, f1_scores, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    f1_class1 = f1_scores[1]  # F1 score for Class 1
    
    f1 = f1_score(y_test, y_pred, average='weighted') 
    
    print(f'Accuracy score: {accuracy:.2f}')
    # print(f'Precision score: {precision:.2f}')
    # print(f'Recall score: {recall:.2f}')
    print(f'F1 score: {f1:.2f}')
    print("Classification Report:")
    class_report = classification_report(y_test, y_pred, target_names=class_labels)
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
    plt.gcf().text(0.5, 0.02, f'Accuracy score: {accuracy:.2f} | F1 score: {f1:.2f} | F1 score for Class 1: {f1_class1:.2f}', ha='center', fontsize=12)
    plt.show()

    return class_report
    
def custom_f1_scorer(y_true, y_pred):
    _, _, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    return f1_scores[1]  # F1 score for Class 1


#define optuna trial object for hyperparamater tuning
def objective(trial,scoring_metric= 'f1'):
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
    random_state= 60
    )
    
    
    # Set the scorer based on user input
    if scoring_metric == 'f1':
        #scorer = make_scorer(custom_f1_scorer, greater_is_better=True)
        scorer = make_scorer(f1_score, average='weighted')  # 'weighted' to handle class imbalance
    elif scoring_metric == 'accuracy':
        scorer = 'accuracy'  # Default scorer for accuracy

    
    
    score = cross_val_score(model, X_train, y_train.flatten(), cv=5, scoring=scorer, n_jobs=-1).mean()
    
    return score
    
   

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
X_train['mood_target'] = y_train
X_test['mood_target'] = y_test


#KBinsDiscritizer to convert labels to classes
est1 = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
y_train = est1.fit_transform(y_train)
y_test= est1.transform(y_test)


# Convert float to int
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print(f'treshold for splitting target as high and low = {est1.bin_edges_[0]}')


#Random Forest set up
model = RandomForestClassifier(n_estimators=100,
                                       random_state=42,
                                       max_depth=5,
                                       class_weight= None,
                                       )
model.fit(X_train, y_train.flatten())

#Predict
y_pred = model.predict(X_test)
#evaluate and plot the confusion matrix 
class_report_base = evaluate(y_test,y_pred,model,'Confusion Matrix for Base Model')


##### Use optuna for hyperparamater search
study = optuna.create_study(direction = 'maximize', sampler = optuna.samplers.RandomSampler(seed=42))
study.optimize(objective, n_trials=100)

# Print the best parameters found 
print("Best trial:")
trial = study.best_trial


print("Value: {:.4f}".format(trial.value))

print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# Initialize the model with the best hyperparameters
best_params = study.best_trial.params
model1 = RandomForestClassifier(**best_params, random_state=60)

#train the model with tuned params
model1.fit(X_train, y_train.flatten())
y_pred = model1.predict(X_test)
class_report_tuned = evaluate(y_test,y_pred,model1,'Confusion Matrix for Tuned Model')

print(class_report_base)
print(class_report_tuned)




cm1 = [[168,15],[47,20]]
cm2 = [[140,  11],[ 41 , 57]]
# Create confusion matrices for each model against the true labels

# Extracting counts for McNemar's test
b = cm1[0][1] + cm1[1][0]  # False positives + false negatives for Model 1
c = cm2[0][1] + cm2[1][0]  # False positives + false negatives for Model 2

# Build the contingency table
table = np.array([[0, b], 
                  [c, 0]])

# Apply McNemar's test
result = mcnemar(table, exact=False, correction=True)  # correction=True applies continuity correction

print('statistic=%.3f' % (result.statistic))
























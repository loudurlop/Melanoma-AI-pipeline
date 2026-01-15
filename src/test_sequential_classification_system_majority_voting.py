from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import pandas as pd
from keras.models import load_model
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import itertools    
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pylab
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score, f1_score

# Parse input arguments
parser = argparse.ArgumentParser(description='Test the hierarchical model system.')
parser.add_argument('--exp1', type=int, required=True, help='Best experiment for task 1: classifying Dysplastic Nevus vs Melanoma')
parser.add_argument('--exp2', type=int, required=True, help='Best experiment for task 2: classifying Mis vs Miv')
parser.add_argument('--exp3', type=int, required=True, help='Best experiment for task 3: classifying BT')
parser.add_argument('--threshold', type=float, default=0, help='Custom threshold for classifying Dysplastic Nevus vs Melanoma')
parser.add_argument('--external_test', type=int, default=0, help='Test: 0; external test: 1')
args = parser.parse_args()

# Configuration
output_dir = os.path.join('..', 'results', 'hierarchical_classification_system', 'global_system_results')
results_dir = os.path.join('..', 'results', 'hierarchical_classification_system')
csv_dir = os.path.join('..', 'csv')

def load_csvs(sources, csv_dir='csv', tipo='train', fold=None):
    dfs = []
    for src in sources:
        if fold is not None:
            path = f"{csv_dir}/{src}_fold_{fold}_{tipo}.csv"
        else:
            path = f"{csv_dir}/{src}_test.csv"

        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f"Not found file: {path}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def reclassify(df, opcion):
    """
    Reclassifies classes according to the selected task:

    Tasks:
    1: Class 0 and Class 1 (merges classes 1, 2, and 3 into Class 1)
    2: Class 1 and Class 2 (merges classes 2 and 3 into Class 2)
    3: Class 2 and Class 3 (only includes these two classes)
    4: Class 0, 1, 2 and 3
    """
    df = df.copy()
    if opcion == 1:
        # Keep 0 as 0; merge 1–3 into 1
        df = df[df["label"].isin([0,1,2,3])]
        df.loc[df["label"] != 0, "label"] = 1        
    elif opcion == 2:
        # Merge classes 2 & 3 into 1; remap class 1 → 0, class 2/3 → 1
        df = df[df["label"].isin([1,2,3])]
        df.loc[df["label"] == 3, "label"] = 2
        df['label'] = df['label'].map({1: 0, 2: 1})  
    elif opcion == 3:
        # Remap class 2 → 0, class 3 → 1
        df = df[df["label"].isin([2,3])]
        df['label'] = df['label'].map({2: 0, 3: 1})        
    elif opcion == 4:
        return df
    else:
        raise ValueError("Not valid task (valid: 1, 2, 3)")
    
    return df

def plot_confusion_matrix(cm, target_names, cmap=None, normalize=True, path_to_save = None):
    
    dpi = 600
    fontsize = 36
    
    Fi = pylab.gcf()
    DefaultSize = Fi.get_size_inches()

    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1800.0/float(DPI),1800.0/float(DPI))    

    if len(target_names) == 4:
        fontsize = 26    

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
        
    cb = plt.colorbar()
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0,fontsize=fontsize, ha='center')
        plt.yticks(tick_marks, target_names,rotation=90, fontsize=fontsize, va='center')

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black",fontsize=fontsize)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=fontsize)

    plt.ylabel('Ground truth', fontsize=fontsize)
    plt.xlabel('Predicted', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(path_to_save, format='svg',dpi=dpi)
    plt.close()

def load_model_group(subfolder, exp_num):
    base_path = os.path.join(results_dir, subfolder, f'N_EXP_{exp_num}', 'densenet121')
    return [
        load_model(os.path.join(base_path, f'model_fold_{fold}.h5'))
        for fold in range(1, 6)
    ]

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Data sources
if args.external_test == 0:
    sources = ["argenciano", "isic", "polesie", "vruh"] # Test
else:    
    sources = ["vruh"] # External test

# Load the models for majority voting
model1 = load_model_group('DysNevus_vs_Melanoma', args.exp1)
model2 = load_model_group('Mis_vs_Miv', args.exp2)
model3 = load_model_group('BT', args.exp3)

if args.external_test == 0:
    test_df = load_csvs(sources, csv_dir=csv_dir, tipo='test', fold=None)
elif args.external_test == 1:
    test_df = df = pd.read_csv(os.path.join('..', 'csv', 'vruh_external_test_50_cases.csv'))

X_test = test_df['path'].values
final_preds = []  # Final labels for each sample (0 to 3)
y_pred_proba = []

for i, x in enumerate(X_test):
    try:
        img = load_img(x, target_size=(224, 224))
    except Exception as e:
        print(f"Error loading image {x}: {e}")
        continue
    
    img_array = img_to_array(img)              
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0              
    
    # Step 1: Dysplastic Nevus (0) vs Melanoma (1) majority voting for the 5 folds
    preds1 = [model.predict(img_array, verbose=0) for model in model1]
    pred1 = np.mean(preds1, axis=0)    
    
    # CUSTOM THREHOLD
    if args.threshold > 0:
        predicted_class1 = 0 if pred1[0][0] > args.threshold else 1
    else:
        predicted_class1 = np.argmax(pred1, axis=1)[0]

    # Step 2: In situ (0) vs Invasive (1) majority voting for the 5 folds
    preds2 = [model.predict(img_array, verbose=0) for model in model2]
    pred2 = np.mean(preds2, axis=0)
    # Prediction of the 5 folds
    predicted_class2 = np.argmax(pred2, axis=1)[0]

    # Step 3: Breslow <1mm (0) vs ≥1mm (1) majority voting for the 5 folds
    preds3 = [model.predict(img_array, verbose=0) for model in model3]
    pred3 = np.mean(preds3, axis=0)
    # Prediction of the 5 folds
    predicted_class3 = np.argmax(pred3, axis=1)[0]
    
    if predicted_class1 == 0:
        final_preds.append(0)  # Dysplastic Nevus
        y_pred_proba.append([pred1[0][0], pred1[0][1], 0, 0]) 
    else:
        if predicted_class2 == 0:
            final_preds.append(1)  # Melanoma in situ
            y_pred_proba.append([pred1[0][0], pred1[0][1]*pred2[0][0], pred1[0][1]*pred2[0][1], 0])
        else:
            if predicted_class3 == 0:
                final_preds.append(2)  # Melanoma invasive <1mm
                y_pred_proba.append([pred1[0][0], pred1[0][1]*pred2[0][0], pred1[0][1]*pred2[0][1]*pred3[0][0], pred1[0][1]*pred2[0][1]*pred3[0][1]])
            else:
                final_preds.append(3)  # Melanoma invasive ≥1mm
                y_pred_proba.append([pred1[0][0], pred1[0][1]*pred2[0][0], pred1[0][1]*pred2[0][1]*pred3[0][0], pred1[0][1]*pred2[0][1]*pred3[0][1]])

y_test = test_df['label'].values.astype(str)
y_true = test_df['label'].astype(int).values
y_pred = np.array(final_preds).astype(int)
y_pred_proba = np.array(y_pred_proba)

# Metrics
# Confusion matrix, balanced accuracy, AUC, Kappa, Recall, F1

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

report = classification_report(y_true, y_pred, output_dict=True, digits=3)
print("\nClassification Report:\n", report)

balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print('Balanced accuracy:', balanced_accuracy)

auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
print('AUC:', auc)

kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print('Kappa:', kappa)

recall = recall_score(y_true, y_pred, average='weighted')
print('Recall:', recall)

f1 = f1_score(y_true, y_pred, average='weighted')
print('F1-score:', f1)

# Specificity
n_classes = cm.shape[0]
specificities = []

for i in range(n_classes):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - (tp + fp + fn)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(spec)

# weighted average:
class_support = cm.sum(axis=1) 
weighted_specificity = np.average(specificities, weights=class_support)
print('Specificity:', weighted_specificity)

# Save confusion matrix
target_names = ['Dysp. nevus', 'Mis', 'BT\n< 1 mm', 'BT\n≥ 1 mm']

# Normalized
plot_confusion_matrix(cm = np.array(cm), 
						normalize = True,
						target_names = target_names,						
						path_to_save = os.path.join(output_dir, f'normalized_confusion_matrix_exp_{args.exp1}_{args.exp2}_{args.exp3}_thr_{args.threshold}_test_{args.external_test}_majority_voting.svg'))

# Not normalized
plot_confusion_matrix(cm = np.array(cm), 
						normalize = False,
						target_names = target_names,
						path_to_save = os.path.join(output_dir, f'confusion_matrix_exp_{args.exp1}_{args.exp2}_{args.exp3}_thr_{args.threshold}_test_{args.external_test}_majority_voting.svg'))


# Generate CSV with classification report
rows = []

for label, metrics in report.items():
    if isinstance(metrics, dict):
        row = {'class': label}
        row.update(metrics)
        rows.append(row)

rows.extend([
    {'class': 'balanced_accuracy', 'precision': balanced_accuracy},
    {'class': 'AUC', 'precision': auc},
    {'class': 'kappa', 'precision': kappa},
    {'class': 'recall', 'precision': recall},
    {'class': 'F1-score', 'precision': f1},
    {'class': 'specificity_weighted', 'precision': weighted_specificity}
])

df_results = pd.DataFrame(rows)

if args.external_test == 0:
    filename = f'classification_results_exp_{args.exp1}_{args.exp2}_{args.exp3}_thr_{args.threshold}_test_0_majority_voting.csv'
elif args.external_test == 1:
    filename = f'classification_results_exp_{args.exp1}_{args.exp2}_{args.exp3}_thr_{args.threshold}_test_1_majority_voting.csv'

with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
    df_results.to_csv(f, index=False)
    f.write('\nConfusion Matrix\n')
    f.write(','.join([''] + [f'pred_{i}' for i in range(cm.shape[1])]) + '\n')
    for i, row in enumerate(cm):
        f.write(f'true_{i},' + ','.join(map(str, row)) + '\n')

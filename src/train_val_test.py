import pandas as pd
import glob
import os
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import argparse
import pylab
import itertools  
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, roc_curve, auc, confusion_matrix, cohen_kappa_score, recall_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Parse input arguments
parser = argparse.ArgumentParser(description='Train a model on the dataset.')
parser.add_argument('--backbone', type=str, choices=['densenet121', 'vgg16', 'resnet50'], required=True, help='Backbone model to use.')
parser.add_argument('--index', type=int, required=True, help='Index for creating a folder with all the files generated in this run.')
parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation.')
parser.add_argument('--task', type=int, choices=[1, 2, 3, 4], required=True, default=1, help='Task to solve: 1: NevusDisp vs Melanoma, 2: Miv vs Mis, 3: BT, 4: Muliclass model')
args = parser.parse_args()

# Configuration
csv_dir = os.path.join('..', 'csv')
EPOCHS = 100 
n_folds = args.n_folds
index = args.index
task_to_perform = args.task 
n_classes = {1: 2, 2: 2, 3: 2, 4: 4}[task_to_perform]
backbone = args.backbone

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
    elif opcion ==4:
        return df
    else:
        raise ValueError("Not valid task (valid: 1, 2, 3)")
    
    return df

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

def plot_roc_curve(fpr, tpr, path_to_save, roc_auc):
    
    fontsize = 50
    dpi = 600
    lw = 5

    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(2400.0/float(DPI), 1800.0/float(DPI))

    plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.legend(loc='lower right', fontsize=fontsize, fancybox=True, shadow=True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, f'roc_curve_fold_{fold}.svg'), dpi=dpi, format='svg')
    plt.close()

def evaluate_model(model, generator, task_to_perform, output_dir, fold_name, target_names):
    y_score = model.predict(generator)
    y_pred = np.argmax(y_score, axis=1)
    y_true = generator.classes

    # Metrics
    # Confusion matrix, balanced accuracy, AUC, Kappa, Recall, F1
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, digits=3)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)    

    auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted')
    if task_to_perform in [1, 2, 3]:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

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
    class_support = cm.sum(axis=1)
    weighted_specificity = np.average(specificities, weights=class_support)

    # Confusion matrix plots
    plot_confusion_matrix(cm, normalize=True, target_names=target_names,
                          path_to_save=os.path.join(output_dir, f'{fold_name}_normalized_confusion_matrix.svg'))
    plot_confusion_matrix(cm, normalize=False, target_names=target_names,
                          path_to_save=os.path.join(output_dir, f'{fold_name}_confusion_matrix.svg'))

    # ROC curve
    if task_to_perform in [1, 2, 3]:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1] if y_score.shape[1] == 2 else y_pred) 
        plot_roc_curve(fpr, tpr,
                       path_to_save=os.path.join(output_dir, f'{fold_name}_roc_curve.svg'),
                       roc_auc=auc)
    
    # Save metrics
    rows = [{'class': label, **metrics} for label, metrics in report.items() if isinstance(metrics, dict)]
    rows.extend([
        {'class': 'balanced_accuracy', 'precision': balanced_accuracy},
        {'class': 'AUC', 'precision': auc},
        {'class': 'kappa', 'precision': kappa},
        {'class': 'recall', 'precision': recall},
        {'class': 'F1-score', 'precision': f1},
        {'class': 'specificity_weighted', 'precision': weighted_specificity}
    ])

    df_results = pd.DataFrame(rows)
    df_results.to_csv(os.path.join(output_dir, f'{fold_name}_classification_results.csv'), index=False)


if task_to_perform == 1:
    target_names = ['Dysplastic nevus', 'Melanoma']
    TASK = 'DysNevus_vs_Melanoma'    
elif task_to_perform == 2:
    target_names = ['Mis', 'Miv']
    TASK = 'Mis_vs_Miv'    
elif task_to_perform == 3:
    target_names = ['BT < 1 mm', 'BT\n≥ 1 mm']
    TASK = 'BT'    
elif task_to_perform == 4:
    target_names = ['Dysp. nevus', 'Mis', 'BT\n< 1 mm', 'BT\n≥ 1 mm']
    

# Create output directory
if task_to_perform in [1, 2, 3]:
    output_dir = os.path.join('..', 'results', 'hierarchical_classification_system')
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, 'TASK')
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f'N_EXP_{index}')
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, backbone)
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = os.path.join('..', 'results', 'multiclass_model')
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f'N_EXP_{index}')
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, backbone)
    os.makedirs(output_dir, exist_ok=True)

# Data sources
sources = ["argenciano", "isic", "polesie", "vruh"]

for fold in range(1, n_folds + 1):

    print(f"Fold {fold}/{n_folds}")
   
    # Load CSVs
    train_df = load_csvs(sources, csv_dir=csv_dir, tipo='train', fold=fold)
    val_df = load_csvs(sources, csv_dir=csv_dir, tipo='val', fold=fold)

    # Reclasify
    train_df = reclassify(train_df, task_to_perform)
    val_df = reclassify(val_df, task_to_perform)

    # Create dataframe for flow_from_dataframe
    train_df_keras = pd.DataFrame({
        'filename': train_df['path'],
        'class': train_df['label'].astype(str)  
    })
    val_df_keras = pd.DataFrame({
        'filename': val_df['path'],
        'class': val_df['label'].astype(str)
    })

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df_keras,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df_keras,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Load the pre-trained model
    if args.backbone == 'densenet121':
        base_model = DenseNet121(weights='imagenet', include_top=False)
    elif args.backbone == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif args.backbone == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(val_df['label']),
        y=val_df['label']
    )
    class_weights = dict(enumerate(class_weights))
    
    # Callbacks
    checkpoint = ModelCheckpoint(os.path.join(output_dir, f'model_fold_{fold}.h5'), save_best_only=True)
    early_stopping = EarlyStopping(patience=10)
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // 32,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.n // 32,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping])
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, f'training_history_fold_{fold}.csv'), index=False)    



    ###### Validation ######

    model = load_model(os.path.join(output_dir, f'model_fold_{fold}.h5'))
    evaluate_model(model, val_generator, task_to_perform, output_dir, f'validation_fold_{fold}', target_names)


    ###### Test ######

    test_df = load_csvs(sources, csv_dir=csv_dir, tipo='test')
    test_df = reclassify(test_df, task_to_perform)

    X_test = test_df['path'].values
    y_test = test_df['label'].values.astype(str)
    test_df = pd.DataFrame({'filename': X_test, 'class': y_test})

    test_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        test_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    evaluate_model(model, test_generator, task_to_perform, output_dir, f'test_fold_{fold}', target_names)
import treeple.tree._honest_tree
from treeple.ensemble._supervised_forest import ObliqueRandomForestClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Feature normalization
df_human = pd.read_excel('data/Human.parcellated_thickness.xlsx')
df_human.head()

df_human_normalize= {}
features = df_human.columns[2:]  # features are from the 2nd column to the last

# Z-score normalization
for feature in features:
    mean = df_human[feature].mean()
    std = df_human[feature].std()
    df_human_normalize[feature] = (df_human[feature] - mean) / std

# Save the Human normalized data
df_human_normalize = pd.DataFrame(df_human_normalize)
label_human = df_human.iloc[:, :2]
df_human_normalize = pd.concat([label_human, df_human_normalize], axis=1)
df_human_normalize.to_excel('Human_normalized_parcellated_thickness.xlsx', index=False)

df_human_normalize_markov = df_human_normalize.loc[:, ~df_human_normalize.columns.str.startswith('Schaefer')]

# Read y
df_sex = pd.read_excel('data/subjects_age_sex_data_MRI.xlsx')

## set up training data
X1 = []
X2 = []
y_human = []
IDs = set(df_human_normalize_markov['sid'])
ref_IDs = set(df_sex['ID'])

for subject in tqdm(IDs):
    if subject in ref_IDs:
        features = np.array(df_human_normalize_markov[df_human_normalize_markov['sid']==subject]).reshape(-1)[2:]
        gender = list(df_sex[df_sex['ID']==subject]['Sex'])
        sex = int(gender[0]=='FEMALE')

        X1.append(list(features[:182]))
        X2.append(list(features[182:]))
        y_human.append(sex)

X1_human = np.array(X1)
X2_human = np.array(X2)



### SPORF ###
reps = 5
sporf_accuracy = []
n_estimator = 50000
accuracies = []
for ii in tqdm(range(reps)):
    x_train, x_test, y_train, y_test = train_test_split(
                    X1_human, y_human, train_size=0.8, random_state=ii, stratify=y_human)
    clf = ObliqueRandomForestClassifier(n_estimators=n_estimator, n_jobs=-1, feature_combinations=2.3)
    clf.fit(x_train, y_train)
    accuracy = np.mean(clf.predict(x_test)==y_test)
    accuracies.append(accuracy)
sporf_accuracy = np.concatenate((sporf_accuracy, accuracies))
print('Accuracy for n_estimator = ', n_estimator,' is ', accuracies)

sporf_accuracy = sporf_accuracy.reshape(5, 5)
print(sporf_accuracy)
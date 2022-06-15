import pandas as pd

# Reading dataset cleane
penguins = pd.read_csv('penguins_cleaned.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'species'  # What we are looking, What we are going to predict
encode = ['sex', 'island']  # Input parameters

for col in encode:  # Enconding the Sex and Island column
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}  # Converting into number the species


def target_encode(val):
    return target_mapper[val]


df['species'] = df['species'].apply(target_encode)  # Applying the function target_encode

# Separating the dataset into X and y matrices
X = df.drop('species', axis=1)  # Input features
Y = df['species']  # species

# Build random forest model
from sklearn.ensemble import RandomForestClassifier  # Applying Classifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle

pickle.dump(clf, open('penguins_clf.pkl', 'wb'))

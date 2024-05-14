import numpy as np 
import pandas as pd 
import plotly.express as px
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max.columns", None) 
pd.set_option("display.precision", 2)

df = pd.read_csv('drug_consumption.csv')
df

df.info()

df.head()

df.describe()

shape = df.shape
print("Number of features: {} ".format(shape[1]))
print("Number of records: {}".format(shape[0]))

print(df.duplicated().value_counts())

df.drop_duplicates()
df = df.drop_duplicates()

print(df.duplicated().value_counts())

df.isnull().sum()

df.drop("ID", axis=1, inplace=True)
psych_score_cols = ['Neuroticism', 'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness', 'Impulsiveness', 'Sensationness']
demographic_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
drug_usage_cols = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

age_col = {
        -0.95197: '18-24',
        -0.07854: '25 - 34',
        0.49788: '35 - 44',
        1.09449: '45 - 54',
        1.82213: '55 - 64',
        2.59171: '65+'
        }
df['Age'] = df['Age'].replace(age_col)

gender_col = {
            0.48246: 'Female',
            -0.48246: 'Male'
            }
df['Gender'] = df['Gender'].replace(gender_col)

education_col = {
            -2.43591: 'Left School Before 16 years',
            -1.73790: 'Left School at 16 years',
            -1.43719: 'Left School at 17 years',
            -1.22751: 'Left School at 18 years',
            -0.61113: 'Some College,No Certificate Or Degree',
            -0.05921: 'Professional Certificate/ Diploma',
            0.45468: 'University Degree',
            1.16365: 'Masters Degree',
            1.98437: 'Doctorate Degree',
            }
df['Education'] = df['Education'].replace(education_col)
df['Education'] = pd.Categorical(df['Education'], ['Left School Before 16 years',
                            'Left School at 16 years',
                            'Left School at 17 years',
                            'Left School at 18 years',
                            'Some College,No Certificate Or Degree',
                            'Professional Certificate/ Diploma',
                            'University Degree',
                            'Masters Degree',
                            'Doctorate Degree'
                            ])

country_col = {
            -0.09765: 'Australia',
            0.24923: 'Canada',
            -0.46841: 'New Zealand',
            -0.28519: 'Other',
            0.21128: 'Republic of Ireland',
            0.96082: 'UK',
            -0.57009: 'USA'
            }
df['Country'] = df['Country'].replace(country_col)

ethnicity_col = {
            -0.50212: 'Asian',
            -1.10702: 'Black',
            1.90725: 'Mixed-Black/Asian',
            0.12600: 'Mixed-White/Asian',
            -0.22166: 'Mixed-White/Black',
            0.11440: 'Other',
            -0.31685: 'White'
            }
df['Ethnicity'] = df['Ethnicity'].replace(ethnicity_col)

for d in drug_usage_cols:
    df[d].replace({
        "CL0": "Never", 
        "CL1": "Over 10Y", 
        "CL2": "Last 10Y", 
        "CL3": "Last 1Y", 
        "CL4": "Last 1M", 
        "CL5": "Last 1W", 
        "CL6": "Last 1D"
    }, inplace=True)

df.rename(columns={
    'Nscore': 'Neuroticism', 
    'Escore': 'Extraversion', 
    'Oscore': 'Openness',
    'Ascore': 'Agreeableness',
    'Cscore':'Conscientiousness',
    'Impulsive':'Impulsiveness',
    'SS': 'Sensationness'
}, inplace=True)

df.head()

#GENDER DISTRIBUTION PIE-CHART
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.show()
plt.clf()

#HISTOGRAM OF AGES
df.sort_values(['Age'], inplace=True)
sns.histplot(x='Age', data=df)
plt.title('Distribution of Ages')
plt.show()
plt.clf()

#AGE PERCENTAGES
age_distribution = df['Age'].value_counts(normalize=True) * 100
print("Percentage distribution of each category in the 'Age' column:")
print(age_distribution)

#HISTOGRAM OF EDUCATION
df.sort_values(['Education'], inplace=True)
sns.countplot(y='Education', data=df)
plt.title('Distribution of Education')
plt.show()
plt.clf()

#EDUCATION PERCENTAGES
age_distribution = df['Education'].value_counts(normalize=True) * 100
print("Percentage distribution of each category in the 'Education' column:")
print(age_distribution)

#COUNTRY DISTRIBUTION 
df['Country'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Country Distribution')
plt.show()
plt.clf()

# ETHINCITY DISTRIBUTION 
df['Ethnicity'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Ethnicity Distribution')
plt.show()
plt.clf()

psych_score_cols=['Neuroticism', 'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness', 'Impulsiveness', 'Sensationness']

for col in psych_score_cols:
        plt.figure()
        plt.title(col)
        plt.boxplot(df[col])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd

df = pd.DataFrame()

df = pd.read_csv("drug_consumption.csv")

df.head()

features = df[['Age', 'Gender', 'Education', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']]
target = df['Heroin']  #Predicting alcohol use pattern

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = BaggingClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
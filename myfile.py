import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as ns
import seaborn as sns
clear
del ns
sns.set_style("whitegrid")
train = pd.read_csv(r'C:/Users/Apu/Documents/Datset/Titanic/train.csv', dtype={"Age" : np.float64})
test = pd.read_csv(r'C:/Users/Apu/Documents/Datset/Titanic/test.csv',dtype={ "Age" : np.float64})
train.head()
test.head()
train.info()
test.info()
np.unique(train['Sex'])
np.unique(train['Embarked'])
train['Embarked'].value_counts()
from sklearn.preprocessing import LabelBinarizer
LabelBinarizer lb = LabelBinarizer()
lb = LabelBinarizer()
lb.fit_transform(train['Embarked'])
df changeOfFeatureType(df):
def changeOfFeatureType(df):
    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1
    df.loc[df['Embarked'] == 'S','Embarked'] = 0
    df.loc[df['Embarked'] == 'C','Embarked'] = 1
    df.loc[df['Embarked'] == 'Q','Embarked'] = 2
    return df
df RemoveUnwantedColumns(df):
def RemoveUnwantedColumns(df):
    drop_columns = ['Ticket','Name','Cabin']
    df = df.drop(drop_columns,1)
    return df
train_set = changeOfFeatureType(train)
train_set = RemoveUnwantedColumns(train_set)
test_set = changeOfFeatureType(test)
test_set = RemoveUnwantedColumns(test)
train.head()
train_set.head()
train_set.isnull().sum()
test_set.isnull().sum()
test_set(test_set.loc[test_set['Fare'].isnull(),['Embarked','Pclass']])
test_set.loc[test_set['Fare'].isnull(),['Embarked','Pclass']]
fare_dist = train_set.loc[(train_set.Embarked == 0) && (train_set.Pclass == 3),['Fare']]
fare_dist = train_set.loc[(train_set.Embarked == 0) and (train_set.Pclass == 3),['Fare']]
fare_dist = train_set.loc[(train_set.Embarked == 0) & (train_set.Pclass == 3),['Fare']]
fare_dist.value_counts().head()
fare_dist['Fare'].value_counts().head()
fare_dist = fare_dist['Fare'].value_counts().head()
fare_dist.head()
fare_dist = fare_dist.reset_index()
fare_dist.columns = ['Fare', 'Counts']
fare_dist
g = sns.lmplot('Fare','Counts',data=fare_dist,hue='Fare', size=10)
plt.xlabel('Fare')
plt.ylabel('Counts')
plt.show()
g = sns.lmplot('Fare','Counts',data=fare_dist,hue='Fare', size=5,scatter_kws={"s" : 100})
plt.xlabel('Fare')
plt.ylabel('Counts')
plt.show()
g = sns.lmplot('Fare','Counts',data=fare_dist,hue='Fare', size=5,scatter_kws={"s" : 100},x_jitter=5.0,y_jitter=5.0)
plt.show()
g = sns.lmplot('Fare', 'Counts',data=fare_dist,fit_reg=False,hue='Fare',x_jitter=5.0,y_jitter=5.0,size=8,scatter_kws={"s": 100})
g.set(xlim=(0, None))
g.set(ylim=(0, None))
plt.title('Embarked = S and Pclass == 3')
plt.xlabel('Fare')
plt.ylabel('Counts')
plt.show()
test_set.Fare = test_set['Fare'].fillna(8.050)
test_set.isnull().sum()
train_set.isnull().sum()
train_set['Embarked'] = train_set['Embarked'].fillna(1)
train_set.isnull().sum()
titanic_df =  train_set.append(pd.DataFrame(data=test_set),ignore_index=True)
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
train_set['Age'] = train_set['Age'].fillna(train_set['Age'].median())
test_set['Age'] = test_set['Age'].fillna(test_set['Age'].median())

# for the sake of showing the plot for each and every age we will drop all the null values 
# remove the outlier age values from the Age feature
titanic_df['Age1'] = titanic_df.Age
titanic_df['Age1'] = titanic_df[titanic_df['Age1'] < 60]

#Impact visualization of Age on Survival through graph
fig = plt.figure(figsize=(13, 5))
average_age = titanic_df[["Age1", "Survived"]].groupby(['Age1'],as_index=False).mean()
average_age['Age1'] = average_age['Age1'].astype(int)
sns.barplot("Age1", "Survived",data=average_age)
plt.show()
fig = plt.figure(figsize=(13, 5))
alpha = 0.3

titanic_df[titanic_df.Survived==0].Age.value_counts().plot(kind='density', color='#6ACC65', label='Not Survived', alpha=alpha)
titanic_df[titanic_df.Survived==1].Age.value_counts().plot(kind='density', color='#FA2379', label='Survived', alpha=alpha)

plt.xlim(0,80)
plt.xlabel('Age')
plt.ylabel('Survival Count')
plt.title('Age Distribution')
plt.legend(loc ='best')
plt.grid()
plt.show()
sex_survived = pd.crosstab(train_set["Sex"],train_set["Survived"])
parch_survived = pd.crosstab(train_set["Parch"],train_set["Survived"])
pclass_survived = pd.crosstab(train_set["Pclass"],train_set["Survived"])

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,5))    
sns.barplot(train_set["Sex"], train_set["Survived"], palette="Set3" ,ax=axis1)
sns.barplot(train_set["Parch"], train_set["Survived"], palette="Set3", ax=axis2)

fig, (axis3,axis4) = plt.subplots(1,2,figsize=(12,5))  
sns.barplot(train_set["Parch"], train_set["Survived"], palette="Set3", ax=axis3)
sns.barplot(train_set["Embarked"], train_set["Survived"], palette="Set3", ax=axis4)

plt.xticks(rotation=90)
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

imp_features = ["Pclass", "Sex", "Age", "Fare", "Embarked","SibSp", "Parch"]

model = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(
    model,
    train_df[imp_features],
    train_df["Survived"],
    cv=3
)
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

imp_features = ["Pclass", "Sex", "Age", "Fare", "Embarked","SibSp", "Parch"]

model = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(
    model,
    train_set[imp_features],
    train_set["Survived"],
    cv=3
)
scores.mean()
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

imp_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

model = RandomForestClassifier(
    random_state=1,
    n_estimators=150,
    min_samples_split=4,
    min_samples_leaf=2
)

scores = cross_validation.cross_val_score(
    model,
    train_set[imp_features],
    train_set["Survived"],
    cv=3
)

print(scores.mean())
def submission_result(model, train_df, test_df, predictors, filename):

    model.fit(train_set[predictors], train_set["Survived"])
    predictions = model.predict(test_set[predictors])

    submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": predictions
    })
model
submission_result(model, train_set, test_df, imp_features, C:\Users\Apu\Documents\Datset\Titanic\sub.csv)
submission_result(model, train_set, test_df, imp_features, r'C:\Users\Apu\Documents\Datset\Titanic\sub.csv')
submission_result(model, train_set, test_set, imp_features, r'C:\Users\Apu\Documents\Datset\Titanic\sub.csv')
def submission_result(model, train_df, test_df, predictors, filename):

    model.fit(train_set[predictors], train_set["Survived"])
    predictions = model.predict(test_set[predictors])

    submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": predictions
    })
    print(submission.head())
submission_result(model, train_set, test_set, imp_features, r'C:\Users\Apu\Documents\Datset\Titanic\sub.csv')
def submission_result(model, train_df, test_df, predictors, filename):

    model.fit(train_set[predictors], train_set["Survived"])
    predictions = model.predict(test_set[predictors])

    submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": predictions
    })
    print(submission.loc[submission['Survived'] == 1])
submission_result(model, train_set, test_set, imp_features, r'C:\Users\Apu\Documents\Datset\Titanic\sub.csv')
%notebook
%notebook C:\Users\Apu\Documents\Datset\Titanic\myfile
%notebook -f r'C:\Users\Apu\Documents\Datset\Titanic\myfile'
%notebook  r'C:\Users\Apu\Documents\Datset\Titanic\myfile'
%history -f r'C:\Users\Apu\Documents\Datset\Titanic\myfile.py'
%history -f C:\Users\Apu\Documents\Datset\Titanic\myfile.py

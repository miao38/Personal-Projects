#Predict who survives on the Titanic
import pandas as pd
pd.set_option("display.expand_frame_repr", False) #this makes it print out all the rows
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#reading in the train and test file
train = pd.read_csv("C:\\Users\\roymi\Documents\Projects\\Titanic\\train.csv")
test = pd.read_csv("C:\\Users\\roymi\Documents\Projects\Titanic\\test.csv")
combine = [train, test] #makes it easier to change both at the same time
#print(train.describe()) #Age is the only one missing data

#Cleaning up the data
#print(train.head())
#handling missing data in Embarked, filled with most frequent occurence
freq_port = train.Embarked.dropna().mode()[0] #most common is "S"
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)
#handling missing data in age, using the median age
median_age = train["Age"].median()
train["Age"] = train["Age"].fillna(median_age)
test["Age"] = train["Age"].fillna(median_age)
train["AgeRange"] = pd.cut(train["Age"], 5)
test["AgeRange"] = pd.cut(test["Age"], 5)
combine = [train, test]

#mapping age
for dataset in combine:
    dataset.loc[dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
    dataset.loc[dataset["Age"] > 64, "Age"] = 4;

#change Sex to numerical
for sex in combine:
    sex["Sex"] = sex["Sex"].map({"female": 0, "male": 1}).astype(int)
#change Embarked to numerical
for embarked in combine:
    embarked["Embarked"] = embarked["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

#Family Stuff
for dataset in combine:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1 #getting family size
for dataset in combine:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

#Creating a interval for fare
for dataset in combine:
    dataset["Fare"] = dataset["Fare"].fillna(train["Fare"].median())
    train["CategoricalFare"] = pd.qcut(train["Fare"], 4)
#mapping fare
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#looking at the titles in the names
for dataset in combine:
    dataset["Title"] = dataset.Name.str.extract(" ([A-Za-z]+)\.", expand = False)
#print(pd.crosstab(train["Title"], train["Sex"]))

#replacing the titles with more common names, and seeing the survival rate
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#converting categorical titles to ordinal and adding a column called Title
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0) #if there isn't a title, it gives it a 0
#print(train.head())

#dropping variables
train = train.drop(["SibSp", "CategoricalFare", "AgeRange", "Parch", "Name", "Ticket", "Cabin"], axis = 1)
test = test.drop(["SibSp", "AgeRange", "Parch", "Name", "Ticket", "Cabin"], axis = 1)
combine = [train, test]
#selecting features using Univariate Selection, uses chi-squared test to select best 10 features
X = train.iloc[:, 2:9]
y = train.iloc[:, 1]
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k="all")
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
'''print(featureScores.nlargest(10,'Score'))  #print 10 best features'''


#Using XGBoost as a model
no_survived = train.drop("Survived", axis = 1)
survived = train["Survived"]
xgb = XGBRegressor()
xgb.fit(no_survived, survived, verbose = False)

#making the prediction
X_test = test.copy()
XGBPredict = np.round(xgb.predict(X_test),0 )
print("XGBoost")
print(round(xgb.score(no_survived, survived) *100, 2)) #got a score of 60.82

#Random Forest 74.162%, 77.033% with titles
RF = RandomForestClassifier(n_estimators = 100)
RF.fit(no_survived, survived)
RFPredict = RF.predict(X_test)
RF.score(no_survived, survived)
RFScore = round(RF.score(no_survived, survived) * 100, 2)
print("Random Forest")
print(RFScore)

#Decision Tree 75.598%, 75.119% with titles
decisionTree = DecisionTreeClassifier()
decisionTree.fit(no_survived, survived)
decisionTreePredict = decisionTree.predict(X_test)
decisionTree.score(no_survived, survived)
decisionTreeScore = round(decisionTree.score(no_survived, survived) * 100, 2)
print("Decision Tree")
print(decisionTreeScore)


#Submission
#Random Forest
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": RFPredict})
submission.to_csv("C:\\Users\\roymi\Documents\Projects\Titanic\SubmissionRandomForest.csv", index=False)
#Decision Tree
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": decisionTreePredict})
submission.to_csv("C:\\Users\\roymi\Documents\Projects\Titanic\SubmissionDecisionTree.csv", index=False)
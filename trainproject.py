# -*- coding: utf-8 -*-
"""trainproject.ipynb

Importing required libaries
"""

import numpy as np
import pandas as pd

"""we use panda read_csv function toReading the data"""

df = pd.DataFrame(pd.read_csv('/content/train.csv'))
df.head()

df.shape

"""returns null value of each column in series """

df.isnull().sum()

"""seprating out of columns which have more than 35% of the values missing in the dataset"""

drop_col = df.isnull().sum()[df.isnull().sum()>(35/100*df.shape[0])]
drop_col

drop_col.index

df.drop(drop_col.index, axis=1, inplace=True)
df.isnull().sum()

df.fillna(df.mean(), inplace = True)
df.isnull().sum()

df['Embarked'].describe()

"""for embarked attribute,we fill null values eith most frequent value in the column"""

df['Embarked'].fillna('S',inplace=True)
df.isnull().sum()

df.corr()

"""sibsp:Number of sibling/spouses abroad
parch:Number of parents/children abroad

so we can make a new column family_size by combinning these two columns
"""

df['FamilySize'] = df['SibSp']+df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
df.corr()

"""Lets check if we weather the person was alone or not can affect the survival rate."""

df['Alone'] = [0 if df['FamilySize'][i]>0 else 1 for i in df.index]
df.head()

df.groupby(['Alone'])['Survived'].mean()

df[['Alone', 'Fare']].corr()

df['Sex'] = [0 if  df['Sex'][i]=='male' else 1 for i in df.index]
df.groupby(['Sex'])['Survived'].mean()

"""It show female passenger  have more chance of surviving than male ones.
It shows women were prioritized over men.
"""

df.groupby(['Embarked'])['Survived'].mean()

""" **CONCLUSION**

*   Female passenger were prioritized over men.
*   People with high class or rich people have higher survival rate than others. The hierarichy might have been followed while saving passangers.
*   passanger traveling with their family have high survival rate.
*   Passenger who borded the ship at cherbourg survived more in proportion then the others.




"""
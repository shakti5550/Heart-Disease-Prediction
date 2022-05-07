import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

df.shape

df.head(10)

df.info()

df.describe()

import seaborn as sns
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

ig = plt.figure(figsize = (15,15))
x = fig.gca()
df.hist(ax = x)

sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')
df.isnull().sum()

X = df.drop('target',axis='columns');
X.head(3)

y = df.target
y.head(3)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators = 100)
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)

from sklearn.metrics import accuracy_score
print("Decision Tree:",accuracy_score(y_test,y_pred))
print("Random Forest:",accuracy_score(y_test,y_pred1))

prediction = model.predict([[58,0,0,100,248,0,0,122,0,1.0,1,0,2]])
print(prediction[0])

# decision tree model into pickle
import pickle
with open('heart_model.pickle','wb') as f:
    pickle.dump(model,f)




import numpy 
import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Global model
with open('heart_model.pickle','rb') as f:
    __model = pickle.load(f)

app  = FastAPI()

class Heart(BaseModel):
    Age: int
    Sex: bool
    Cp:int
    Trestbps:int
    Chol:int
    Fbs:bool
    Restecg:int
    Thalach:int
    Exang:bool
    Oldpeak:float
    Slope:int
    Ca:int
    Thal:int

@app.get('/')
async def home():
    return "Welcome"

@app.post('/predict')
async def model(data:Heart):
    data = data.dict()

    age = data['Age']
    sex = data['Sex']
    cp = data['Cp']
    trestbps = data['Trestbps']
    chol = data['Chol']
    fbs = data['Fbs']
    restecg = data['Restecg']
    thalach = data['Thalach']
    exang = data['Exang']
    oldpeak = data['Oldpeak']
    slope = data['Slope']
    ca = data['Ca']
    thal = data['Thal']

    result = __model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])[0]
    
    if(result == 1):
        return "Pearson has Heart disease"
    return "Pearson is healthy"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

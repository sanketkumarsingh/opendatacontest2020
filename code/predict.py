from pandas import read_csv
import pandas as pd
from sklearn.linear_model import LinearRegression
# COPD
df = read_csv('/Users/sanket/Desktop/Covid/COVID_Training.csv');
#print(df)
X = pd.DataFrame(df[['Geogname', 'Year', 'Datagroup', 'Diseases of the Respiratory System',
                     'Infectious and Parasitic Diseases',	'Respiratory System: asthma',
                     'Respiratory System: Pneumonia',	   'Males',	'Females',
                     'Both Genders',	'min temperature',	'max temperature']])

y = pd.DataFrame(df[['Respiratory System: COPD']])
#print(X[0])
#print (y)
model = LinearRegression().fit(X, y)

df_test = read_csv('/Users/sanket/Desktop/Covid/COVID_Test.csv');
X_Test = pd.DataFrame(df_test[['Geogname', 'Year', 'Datagroup', 'Diseases of the Respiratory System',
                     'Infectious and Parasitic Diseases',	'Respiratory System: asthma',
                     'Respiratory System: Pneumonia',	   'Males',	'Females',
                     'Both Genders',	'min temperature',	'max temperature']])
Y_real_Test = pd.DataFrame(df_test[['Respiratory System: COPD']])
Y_predict_Test = model.predict(X_Test);
print(Y_predict_Test)

# print(Y_real_Test)
# print (Y_predict_Test)
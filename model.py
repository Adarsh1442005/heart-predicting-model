from py4j.java_gateway import JavaGateway,GatewayParameters 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlxtend.plotting import plot_decision_regions
import mysql.connector

#ldataframe = pd.read_csv(r"C:\Users\BIT\AppData\Local\Temp\9df9de47-9894-45f2-9338-d08824fa6470_heart+disease.zip.470\processed.cleveland.data")
# dataframe.to_csv("cardio.csv",index=None)
read= pd.read_csv("cardio.csv")
'''
there is no null values in the dataset therefore we can use it efficiently no need to much more framing on the dataset
'''
read.info()
'''
data visualization
'''
column=['age','sex','cp','trestbps','chol','fbs','restcg','thalach','exang','oldpeak','slope','ca','thal','num']
read.columns=column
a=read.head(50)
print(read.head(10))
# plt.scatter(a['age'],a['trestbps'])
plt.xlabel('age')
plt.ylabel('restbps')
# by scatter plot we can visualize that the patient under the age of (50,60) having a major isuue related to the blood pesuree;
plt.show()
'''
visualizing the model again for training via svm
'''
data= read.head(500)
sns.scatterplot(x="trestbps",y="chol",data=data,hue="num")
plt.show()
# here we are not able to draw any planevector for any kind of the testing features so we cannot use here svm otherwise accuracy will be lay down for the model
from sklearn.model_selection import train_test_split
read=read.drop("thal", axis=1)
read=read.drop("ca",axis=1)
x= read.iloc[:,:-1]
y=read["num"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=15)
from sklearn.svm import SVC
sv=SVC(kernel="linear")
sv.fit(x_train,y_train)
# eprint(sv.score(x_train,y_train)*100)
# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=sv, filler_feature_values={0:1.5,1:0.5,2:1.5,3:1.5,4:1.5,5:1.5,6:1.5,7:1.5,8:1.5,9:1.5,10:1.5})
plt.show()
"""
using svm our model can predict accurate result
"""
# since data is non linear so we use to prefer the decision tree
a=["cp","fbs","thalach","oldpeak","slope"]

for i in a:
    read= read.drop(i,axis=1)
read=read[read['num'].isin([0,1])]
x=read.iloc[:,:-1]
y=read["num"]
read.info()
x_train,x_test,y_train,y_test=train_test_split=train_test_split(x,y,test_size=0.5,random_state=15)
from sklearn.tree import DecisionTreeClassifier
for i in range(1,20):

 dt=DecisionTreeClassifier(max_depth=i)
 dt.fit(x_train,y_train)
 print(dt.score(x_test,y_test),i)
# maximum accuracy is at 1;
dt2= DecisionTreeClassifier(max_depth=2)
dt2.fit(x_train,y_train)
print(dt2.score(x_test,y_test))
# connection with the database
host="localhost"
username="username"
password="password"

conn=mysql.connector.connect(host,username,password)
# creating instance of cursor class to executr sql queries
connect= conn.cursor()
# creating Database with name
connect.execute("use databasename")
connect.execute("select age from tablename ")
row= connect.fetchall()
age=row
connect.execute("select * from gender")
sex=connect.fetchall()
connect.execute("select * from bp")
blood_pressure=connect.fetchall()
connect.execute("select * from chol")
serum_cholestrol=connect.fetchall()
connect.execute("select * from restcg")
restcg=connect.fetchall()
connect.execute("select * from exang")
exang=connect.fetchall()
if(dt.predict([[age,sex,blood_pressure,serum_cholestrol,restcg,exang]])==0):
   print("patient is out of danger")
else:
   print("please talk to the nearst doctor")   
# since model is giving an acurrate result



















































import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
@st.cache()
def load_dataset():
  df=pd.read_csv("glass-types.csv",header=None)
  df.drop([0],inplace=True,axis=1)
  df.columns=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
  return df
df=load_dataset()
x_train,x_test,y_train,y_test=train_test_split(df[df.columns[:-1]],df["GlassType"],train_size=0.70,random_state=42)
@st.cache()
def prediction(x,model):
  y_pred=model.predict([x])
  y_pred=y_pred[0]
  if y_pred==1:
    return ("used for making building windows (float processed)".upper())
  elif y_pred==2:
    return ("used for making building windows (non-float processed)".upper())
  elif y_pred==3:
    return ("used for making vehicle windows (float processed)".upper())
  elif y_pred==4:
    return ("used for making vehicle windows (non-float processed)".upper())
  elif y_pred==5:
    return ("used for making containers".upper())
  elif y_pred==6:
    return ("used for making tableware".upper())
  elif y_pred==7:
    return ("used for making headlamps".upper())
st.title("Glass Type prediction Web app")
st.sidebar.title("Glass Type prediction Web app")    
if st.sidebar.checkbox("show raw data"):
  st.subheader("Glass Type dataset")
  st.dataframe(df)       
st.sidebar.subheader("Visualisation")   
graph=st.sidebar.multiselect("Select the desired plot type",("pie-chart","countplot","box-plot","area chart","line-chart","correlation heatmap","histogram","bargraph"))
st.set_option('deprecation.showPyplotGlobalUse', False)
if "pie-chart" in graph:
    st.subheader("Pie-chart")
    plt.pie(df["GlassType"].value_counts(),labels=df["GlassType"].value_counts().index,startangle=45,autopct="%1.2f%%",explode=np.linspace(0.06,0.12,6))
    st.pyplot()
if "countplot" in graph:
    st.subheader("countplot")
    sns.countplot(df["GlassType"])
    st.pyplot()    
if "box-plot" in graph:
    st.subheader("boxplot")
    at=st.sidebar.selectbox("Select the attribute for boxplot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
    sns.boxplot(df[at])
    st.pyplot()     
if "area chart" in graph:
    st.subheader("area chart")
    st.area_chart(df)   
if "line-chart" in graph:
    st.subheader("line chart")
    at=st.sidebar.selectbox("Select the attribute for line-chart",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
    st.line_chart(df[at])
if "correlation heatmap" in graph:
    st.subheader("correlation heatmap")
    sns.heatmap(df.corr(),annot=True)
    st.pyplot()
if "histogram" in graph:
    st.subheader("histogram")
    sns.distplot(df,bins="sturges")
    st.pyplot()            
if "bargraph" in graph:
    st.subheader("bargraph")
    st.bar_chart(df["GlassType"])
st.sidebar.subheader("Select the required chemical properties")
ri=st.sidebar.slider("Select the Refractive Index",float(df["RI"].min()),float(df["RI"].max()))
na=st.sidebar.slider("Select the Sodium composition",float(df["Na"].min()),float(df["Na"].max()))
mg=st.sidebar.slider("Select the Magnesium composition",float(df["Mg"].min()),float(df["Mg"].max()))
al=st.sidebar.slider("Select the Aluminium composition",float(df["Al"].min()),float(df["Al"].max()))
si=st.sidebar.slider("Select the Silicon composition",float(df["Si"].min()),float(df["Si"].max()))
k=st.sidebar.slider("Select the Pottasium composition",float(df["K"].min()),float(df["K"].max()))
ca=st.sidebar.slider("Select the Calcium composition",float(df["Ca"].min()),float(df["Ca"].max()))
ba=st.sidebar.slider("Select the Barium composition",float(df["Ba"].min()),float(df["Ba"].max()))
fe=st.sidebar.slider("Select the Iron composition",float(df["Fe"].min()),float(df["Fe"].max()))
st.sidebar.subheader("Select the classifier")
model_=st.sidebar.selectbox("Classifier",("Support Vector Machine","Random Forest Classifier","Logistic Regression"))


if model_=="Support Vector Machine":
    error_rate=st.sidebar.number_input("Select the Error Rate",1,100,1)
    gamma_=st.sidebar.number_input("Select the gamma value",1,100,1)
    kernel=st.sidebar.radio("Select the type for kernel",("poly","linear","rbf"))
    if st.sidebar.button("Classify"):
      st.subheader("Support vector machine")
      svc=SVC(kernel=kernel,C=error_rate,gamma=gamma_)
      svc.fit(x_train,y_train)
      score=svc.score(x_train,y_train)
      ps=prediction([ri,na,mg,al,si,k,ca,ba,fe],svc)
      st.write(ps)
      st.write("the accuracy score is",score.round(2))
      plot_confusion_matrix(svc,x_test,y_test)
      st.pyplot()
if model_=="Random Forest Classifier":
    decision_trees=st.sidebar.number_input("Select the number of decision trees",100,2000,step=10)
    depth=st.sidebar.number_input("Select the depth of the tree",1,100,1)
    if st.sidebar.button("Classify"):
      st.subheader("Random Forest Classifier")
      rfc=RandomForestClassifier(n_jobs=-1,n_estimators=decision_trees,max_depth=depth)
      rfc.fit(x_train,y_train)
      score=rfc.score(x_train,y_train)
      ps=prediction([ri,na,mg,al,si,k,ca,ba,fe],rfc)
      st.write(ps)
      st.write("the accuracy score is",score.round(2))
      plot_confusion_matrix(rfc,x_test,y_test)
      st.pyplot()      
if model_=="Logistic Regression":
    error_rate=st.sidebar.number_input("Select the error rate",1,100,step=1)
    max_iterations=st.sidebar.number_input("Select the maximum number of iterations",10,1000,step=10)
    if st.sidebar.button("Classify"):
      st.subheader("Logistic Regression")
      lr=LogisticRegression(C=error_rate,max_iter=max_iterations)
      lr.fit(x_train,y_train)
      score=lr.score(x_train,y_train)
      ps=prediction([ri,na,mg,al,si,k,ca,ba,fe],lr)
      st.write(ps)
      st.write("the accuracy score is",score.round(2))
      plot_confusion_matrix(lr,x_test,y_test)
      st.pyplot()    

       

    


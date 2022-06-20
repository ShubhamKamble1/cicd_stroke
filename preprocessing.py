import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


df = pd.read_csv("healthcare-dataset-stroke-data.csv")
# df.shape

# df.head()

# df.info()

# Missing values
plt.style.use('seaborn')
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), yticklabels = False, cmap = 'plasma')
plt.title('Null Values in Data Frame')

# get the number of missing data points per column
missing_value_count = (df.isnull().sum())
print(missing_value_count[missing_value_count > 0])
# percent of data that is missing
total_cells = np.product(df.shape)
total_missing_value = missing_value_count.sum()
print('Percentage of missing value in Data Frame is:', total_missing_value / total_cells*100)
print('Total number of our cells is:', total_cells)
print('Total number of our missing value is:', total_missing_value)

# Handle missing data
df['bmi'].fillna(df['bmi'].mean(),inplace=True)
df['bmi'].isnull().sum()

df.drop(['id'],axis=1,inplace=True)

# Data preparation
# Labeling data fields to Text value for easy interpretation of Visualization
data_eda = df.copy()
#hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
data_eda["hypertension"]     = df["hypertension"]    .map({1: "Yes",           0: "No"})
#1 if the patient had a stroke or 0 if not
data_eda["stroke"]     = df["stroke"]    .map({1: "Yes",           0: "No"})
#0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
data_eda["heart_disease"]  = df["heart_disease"] .map({1: "Yes" ,           0: "No"})

# Exploratory data analysis
def cnditioning_linear_plot(x,y,hue,df):
    sns.lmplot(x=x, y=y, hue=hue, data=df,
               markers=["o", "x"], palette="Set1")

def count_bar_plot(df,x,hue,title):
    fig = sns.countplot(x=x, hue=hue, data=df)
    fig.set_title(title)

def pie_graph(df,title,values):   
    labels = df[values].value_counts().index
    values = df[values].value_counts()

    fig = go.Figure(data = [
        go.Pie(
        labels = labels,
        values = values,
        hole = .5)
    ])

    fig.update_layout(title_text = title)
    fig.show()

def distplot(x):
    ax = sns.distplot(data_eda[x], rug=True, rug_kws={"color": "g"},
                  kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})

def horizontal_bar_chart(df,x,y,color,title):    
    fig = px.bar(df, x=x, y=y, color=color,                  
                 height=600,
                 title=title)
    fig.show()

df.describe()

distplot('age')

## cohort analysis of age with output
def age_cohort(age):
    if   age >= 0 and age <= 20:
        return "0-20"
    elif age > 20 and age <= 40:
        return "20-40"
    elif age > 40 and age <= 50:
        return "40-50"
    elif age > 50 and age <= 60:
        return "50-60"
    elif age > 60:
        return "60+"
    
data_eda['age group'] = data_eda['age'].apply(age_cohort)
data_eda.sort_values('age group', inplace = True)

# Analysis Viz
# pie_graph(data_eda,"Age Group Distribution",'age group')

# pie_graph(data_eda, 'Gender Distribution','gender')

# pie_graph(data_eda, 'Hypertension Distribution','hypertension')

# pie_graph(data_eda, ' Heart disease Distribution','heart_disease')

# pie_graph(data_eda, 'Ever married  Distribution','ever_married')

# distplot('bmi')

# pie_graph(data_eda, 'Work type Distribution','work_type')

# distplot('avg_glucose_level')

# pie_graph(data_eda, 'Residence type Distribution','Residence_type')

# pie_graph(data_eda,'Smoking Status Distribution','smoking_status')

# pie_graph(data_eda, 'Stroke Distribution', 'stroke')

# count_bar_plot(data_eda,'gender','stroke','Distribution by Gender')

# count_bar_plot(data_eda,'hypertension','stroke','Distribution by hypertension')

# count_bar_plot(data_eda,'heart_disease','stroke','Distribution by heart_disease')

# count_bar_plot(data_eda,'ever_married','stroke','Distribution by ever married')

group = data_eda.groupby(['stroke','work_type'],as_index = False).size().sort_values(by='size')
horizontal_bar_chart(df = group,x = 'stroke',y = 'size',color = 'work_type',title = 'Distribution of stroke by work type')

# Analysis Viz
# count_bar_plot(data_eda,'Residence_type','stroke','Distribution by Residence type')

group = data_eda.groupby(['stroke','smoking_status'],as_index = False).size().sort_values(by='size')
horizontal_bar_chart(df = group,x = 'stroke',y = 'size',color = 'smoking_status',title = 'Distribution of stroke by smoking status')

# Analysis Viz
# cnditioning_linear_plot('age','avg_glucose_level','stroke',data_eda)

# cnditioning_linear_plot('bmi','avg_glucose_level','stroke',data_eda)

# cnditioning_linear_plot('bmi','age','stroke',data_eda)

plt.figure(1, figsize=(15,7))
n = 0
for x in ['age','avg_glucose_level','bmi']:
    for y in ['age','avg_glucose_level','bmi']:
        n += 1
        plt.subplot(3,3,n)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        sns.regplot(x = x, y = y, data = df)
        plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y)

plt.show()

f, ax = plt.subplots(figsize = (14,14))
sns.heatmap(df.corr(),
            annot = True,
            linewidths = .5,
            fmt = '.1f',
            ax = ax)

df.corr()['stroke'].sort_values(ascending = False)

# df.dtypes

object_col = ["gender", "ever_married" ,"Residence_type"]
label_encoder = preprocessing.LabelEncoder()
for col in object_col:
    df[col]=  label_encoder.fit_transform(df[col])

# df.head(2)

df = pd.get_dummies(df)
# df.head(2)

# df.shape

X = df.drop(columns = ['stroke'])
y = df['stroke']

# SMOTE
sm = SMOTE(random_state=123)
X_sm , y_sm = sm.fit_resample(X,y)

# print(f'''Shape of X before SMOTE:{X.shape}
# Shape of X after SMOTE:{X_sm.shape}''',"\n\n")

# print(f'''Target Class distributuion before SMOTE:\n{y.value_counts(normalize=True)}
# Target Class distributuion after SMOTE :\n{y_sm.value_counts(normalize=True)}''')

df1 = pd.concat([X_sm, y_sm], axis=1)

df1.to_csv("data_processed.csv")

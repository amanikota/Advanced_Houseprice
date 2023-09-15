#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv("C:/Users/amani/Downloads/train.csv")
df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.duplicated().sum()


# In[9]:


df.isnull().sum()


# In[10]:


null_counts=df.isnull().sum()


# In[11]:


features_with_null=null_counts[null_counts>0].index
print(features_with_null)


# In[12]:


null_counts=df[['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
       'MiscFeature']].isnull().sum()
print(null_counts)


# In[13]:


df=df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'],axis=1)


# In[14]:


numeric_features=['LotFrontage','MasVnrArea','GarageYrBlt']
for feature in numeric_features:
    df[feature].fillna(df[feature].mean(),inplace=True)
categorical_features=['BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','FireplaceQu',
         'GarageType','Electrical','GarageFinish', 'GarageQual', 'GarageCond']
for feature in categorical_features:
    df[feature].fillna(df[feature].mode(),inplace=True)


# In[15]:


df.isnull().sum()


# In[16]:


df.info()


# In[17]:


df.describe()


# In[18]:


df=df.drop(['Id'],axis=1)


# In[19]:


#dividing object and numerical columns and making a list
object_columns=df.select_dtypes(include='object').columns.tolist()
numerical_columns=df.select_dtypes(include=['int','float']).columns.tolist()
print("object columns:",object_columns)

print("Numerical columns:",numerical_columns)


# In[20]:


df.nunique()  #to print unique  values present in each and every columns


# In[21]:


#to show the unique values in object columns
for i in object_columns:
    print(i)
    print(df[i].unique())
    print('\n')


# In[22]:


#to display how many unique values are present in object columns
for i in object_columns:
    print(i)
    print(df[i].value_counts())
    print('\n')


# In[23]:


for i in object_columns:
    print('Countplot for:',i)
    plt.figure(figsize=(15,6))
    sns.countplot(df[i],data=df,palette='hls')
    plt.xticks(rotation=-45)
    plt.show()
    print('\n')


# In[24]:


for i in object_columns:
    print("pie plot for:",i)
    plt.figure(figsize=(20,10))
    df[i].value_counts().plot(kind='pie',autopct='%1.1f%%')
    plt.title('Distribution of '+i)
    plt.ylabel('')
    plt.show()


# In[25]:


#graphical object
#plotly is used for better visualization that matplot or seaborn
for i  in object_columns:
    fig = go.Figure(data=[go.Bar(x=df[i].value_counts().index,y=df[i].value_counts())])
    fig.update_layout(
    title=i,
    xaxis_title=i,
    yaxis_title="count")
    fig.show()


# In[26]:


# px is used or high level interface
for i in object_columns:
    print('Pie plot for:',i)
    fig=px.pie(df,names=i,title='Distribution of'+i)
    fig.show()
("   print('\n')")


# In[27]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.histplot(df[i],kde=True,bins=20,palette='hls')
    plt.xticks(rotation=0)
    plt.show()


# In[28]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.distplot(df[i],kde=True,bins=20)
    plt.xticks(rotation=0)
    plt.show()


# In[29]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.boxplot(df[i],data=df,palette='hls')
    plt.xticks(rotation=0)
    plt.show()


# In[30]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.violinplot(df[i],data=df,palette='hls')
    plt.xticks(rotation=0)
    plt.show()


# In[31]:


for i in numerical_columns:
    fig =go.Figure(data=[go.Histogram(x=df[i],nbinsx=20)])
    fig.update_layout(
    title=i,
    xaxis_title=i,
    yaxis_title="count")
    fig.show()
    


# In[32]:


for i in numerical_columns:
    if i!='SalePrice':
        plt.figure(figsize=(15,6))
        sns.barplot(x=df[i],y=df['SalePrice'],data=df,ci=None,palette='hls')
        plt.show()


# In[33]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.scatterplot(x=df[i],y=df['SalePrice'],data=df,palette='hls')
    plt.show()


# In[34]:


for i in numerical_columns:
    for j in object_columns:
        plt.figure(figsize=(15,6))
        sns.barplot(x=df[j],y=df[i],data=df,ci=None,palette='hls')
        plt.show()


# In[35]:


corr = df.corr()


# In[36]:


corr


# In[37]:


plt.figure(figsize=(30,20))
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt=" .2f")
plt.title("correlation plot")
plt.show()


# In[38]:


df1 =df.copy()


# In[39]:


plt.figure(figsize=(15,6))
sns.histplot(df1['SalePrice'],kde=True,bins=20,palette='hls')
plt.xticks(rotation=0)
plt.show()
#target feature and other features should be normally distributed then only we can apply linear regression


# In[40]:


#if features are not normally distributed make it normal distribution
df1['SalepPrice'] =np.log(df1['SalePrice'])


# In[41]:


plt.figure(figsize=(15,6))
sns.distplot(df1['SalePrice'],kde=True)
plt.xticks(rotation=0)
plt.show()


# In[42]:


numerical_columns


# In[43]:


numerical_columns=['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd', 'MasVnrArea','BsmtFinSF1','BsmtFinSF2',
 'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
 'Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold',
 'YrSold',]


# In[44]:


## numerical_columns


# In[ ]:





# In[45]:


#prints all skewed distributed features
#0: The data is perfectly symmetric.
#Greater than 0: The data is right-skewed (positively skewed), meaning it has a long tail on the right side.
#Less than 0: The data is left-skewed (negatively skewed), meaning it has a long tail on the left side.
skewness =df1[numerical_columns].skew()
skewed_columns =skewness[(skewness >1)|(skewness < -1)]
print(skewed_columns)


# In[46]:


skew_features = skewed_columns.index.tolist()


# In[47]:


skew_features


# In[48]:


for feature in skew_features:
    df1[feature] = np.log1p(df1[feature])

transformed_features=['LotFrontage',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'TotalBsmtSF',
 '1stFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'BsmtHalfBath',
 'KitchenAbvGr',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal']
# In[ ]:





# In[49]:


for feature in skew_features:
    plt.figure(figsize=(15,6))
    sns.histplot(df1[feature],kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()


# In[50]:


df1=pd.get_dummies(df1,columns=object_columns,drop_first=True)


# In[51]:


df1


# In[52]:


corr1 = df1.corr()


# In[53]:


corr1


# In[54]:


correlation_threshold = 0.5


# In[55]:


good_features=corr1[corr1['SalePrice'].abs() > correlation_threshold]['SalePrice'].index.tolist()


# In[56]:


good_features.append('SalePrice')


# In[57]:


df2 = corr1[good_features]


# In[58]:


df2


# In[59]:


X =df2.drop(['SalePrice'],axis=1)
y=df2['SalePrice']


# In[60]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[61]:


from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()


# In[62]:


regression_model.fit(X_train,y_train)


# In[63]:


y_pred =regression_model.predict(X_test)


# In[64]:


from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2_linear = r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("R-squared Score:",r2_linear)


# In[65]:


df3 = df1[good_features]
df3


# In[66]:


X =df3.drop(['SalePrice'],axis=1)
y=df3['SalePrice']


# In[67]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[68]:


from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()


# In[69]:


regression_model.fit(X_train,y_train)


# In[70]:


y_pred =regression_model.predict(X_test)


# In[71]:


from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2_linear = r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("R-squared Score:",r2_linear)


# In[ ]:





# In[ ]:





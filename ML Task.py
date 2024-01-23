#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[175]:


import numpy as np
import pandas as pd
df_1= pd.read_csv('train.csv')
df_2= pd.read_csv('test.csv')


# In[176]:


df_1.head()


# In[177]:


col_to_drop='Id'


# In[178]:


df_1=df_1.drop(col_to_drop,axis=1)
df_2=df_2.drop(col_to_drop,axis=1)


# In[179]:


df_1.head()


# In[180]:


# Select only numerical columns
n_col = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
c_col=df_1.select_dtypes(include=['object']).columns
df_1 = df_1.select_dtypes(include=n_col)
df_2 = df_2.select_dtypes(include=n_col)


# In[ ]:





# In[181]:


df_1.head()


# In[182]:


df_1.isnull().sum()


# In[183]:


## Drop the columns containing null values
df_1 = df_1.loc[:,df_1.isnull().sum() == 0]
df_2 = df_2.loc[:,df_1.columns.drop(['SalePrice'])]


# In[184]:


df_1.describe()


# In[185]:


# Split train -> train, validation
from sklearn.model_selection import train_test_split
x = df_1[df_1.columns.drop(['SalePrice'])]
y = df_1[['SalePrice']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42) 


# In[186]:


# Scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:





# In[187]:


from sklearn.neighbors import KNeighborsRegressor
regressor =  KNeighborsRegressor()
regressor.fit(x_train, y_train)


# In[188]:


#Predicting the test set result  
y_pred= regressor.predict(x_test)


# In[ ]:





# In[ ]:





# In[189]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:",mse)
mae=mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error: ",mae)


# In[ ]:





# In[190]:


regressor.score(x_train,y_train)


# In[191]:


sc = StandardScaler()
x_train = sc.fit_transform(x)
x_test = sc.transform(df_2.fillna(0)) # Fill nan values in test dataframe with 0


regressor.fit(x_train,y)
test_pred = regressor.predict(x_test)


# In[193]:


sample_sub = pd.read_csv('sample_submission.csv')

sub = pd.DataFrame({'Id': sample_sub.Id, 'SalePrice': test_pred.reshape(-1)})

sub.to_csv('mysubmission.csv', index=False)


# In[194]:


column_names = df_1.columns.tolist()

print("List of column names:", column_names)


# In[195]:


sub


# In[196]:


import seaborn as sns
import matplotlib.pyplot as plt
# Set Seaborn style to "white"
sns.set_style("white")

# Create a distribution plot for 'SalePrice'
plt.figure(figsize=(8, 7))
sns.histplot(sub['SalePrice'], color="blue", kde=True)

# Customize plot labels and title
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.title("Distribution of SalePrice")

# Remove spines on the left and top for a clean look
sns.despine(left=True, top=True)
# Show the plot
plt.show()


# In[216]:


data_1 = df_1[['MSSubClass','LotArea','BsmtFinSF1','BsmtUnfSF','1stFlrSF','2ndFlrSF','GrLivArea']]


# In[217]:


data_1.describe()


# In[218]:


df_3=data_1.dropna()


# In[219]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_scaled=scaler.fit_transform(df_3)

data_s=pd.DataFrame(data_scaled).describe()

x = df_3.values


# In[ ]:





# In[220]:


#finding optimal number of clusters using the elbow method  
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming X is your preprocessed feature matrix
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_3)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # WCSS stands for "within-cluster sum of squares"
plt.show()


# In[221]:


from sklearn.cluster import KMeans
kmeans= KMeans(n_clusters=4,init='k-means++')
y_predict= kmeans.fit_predict(x)


# In[222]:


y_predict


# In[223]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming X is your preprocessed feature matrix and y is your target variable (sale price)
# Assuming clusters are already defined

# Combine X, y, and clusters into a DataFrame
import pandas as pd
df = pd.DataFrame(df_3, columns=['MSSubClass','LotArea','BsmtFinSF1','BsmtUnfSF','1stFlrSF','2ndFlrSF','GrLivArea'])
df['SalePrice'] = y
df['Cluster'] = y_predict

# Create scatter plots for each feature against Sale Price with hue=Cluster
for feature in ['MSSubClass','LotArea','BsmtFinSF1','BsmtUnfSF','1stFlrSF','2ndFlrSF','GrLivArea']:
    sns.scatterplot(data=df, x=feature, y='SalePrice', hue='Cluster', palette='viridis', edgecolor='k', s=50)
    plt.scatter(kmeans.cluster_centers_[:, df.columns.get_loc(feature)], kmeans.cluster_centers_[:, -2], s=300, c='red', marker='X')
    plt.title(f'Scatter Plot of {feature} vs Sale Price with Clusters')
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.show()


# In[81]:


data_2 = df_1[['MSSubClass','LotArea','BsmtFinSF1','BsmtUnfSF','1stFlrSF','2ndFlrSF','GrLivArea']]
df_4=data_2.dropna()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_scaled=scaler.fit_transform(df_4)

data_s=pd.DataFrame(data_scaled).describe()

x1 = df_4.values


# In[ ]:





# In[ ]:





# In[ ]:





# In[94]:


import gradio as gr

# Compute mean values for the remaining 26 features
mean_values = df_1[['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','BsmtFinSF2', 'TotalBsmtSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']].mean().to_dict()


# In[95]:


mean_values


# In[ ]:





# In[103]:


# Create Gradio interface for Deployment
def predict_house_price(MSSubClass,LotArea,BsmtFinSF1,BsmtUnfSF,_1stFlrSF,_2ndFlrSF,GrLivArea):

    # Convert the dictionary to a DataFrame
    input_data = pd.DataFrame([input_values])

    # Preprocess input features
    input_data_scaled = scaler.transform(input_data)

    # Use the trained KNN model to predict house price
    predicted_price = regressor.predict(input_data_scaled)[0]
    return predicted_price

default_values={'OverallQual': 6.0993150684931505,
 'OverallCond': 5.575342465753424,
 'YearBuilt': 1971.267808219178,
 'YearRemodAdd': 1984.8657534246574,
 'BsmtFinSF2': 46.54931506849315,
 'TotalBsmtSF': 1057.4294520547944,
 'LowQualFinSF': 5.844520547945206,
 'BsmtFullBath': 0.42534246575342466,
 'BsmtHalfBath': 0.057534246575342465,
 'FullBath': 1.5650684931506849,
 'HalfBath': 0.38287671232876713,
 'BedroomAbvGr': 2.8664383561643834,
 'KitchenAbvGr': 1.0465753424657533,
 'TotRmsAbvGrd': 6.517808219178082,
 'Fireplaces': 0.613013698630137,
 'GarageCars': 1.7671232876712328,
 'GarageArea': 472.9801369863014,
 'WoodDeckSF': 94.2445205479452,
 'OpenPorchSF': 46.66027397260274,
 'EnclosedPorch': 21.954109589041096,
 '3SsnPorch': 3.4095890410958902,
 'ScreenPorch': 15.060958904109588,
 'PoolArea': 2.758904109589041,
 'MiscVal': 43.489041095890414,
 'MoSold': 6.321917808219178,
 'YrSold': 2007.8157534246575}

# Create Gradio interface with named input boxes
input_interface = gr.Interface(
    fn=predict_house_price,
    inputs=[*default_values,gr.Slider(),gr.Slider(),gr.Slider(),gr.Slider(),gr.Slider(),gr.Slider(),gr.Slider()],
    outputs=gr.Textbox(label="Predicted House Price")
)

# Launch the Gradio interface
input_interface.launch(share=True)


# In[ ]:





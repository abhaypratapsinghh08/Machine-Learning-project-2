#!/usr/bin/env python
# coding: utf-8

# # Mini Project 2

# ### Import required libraries

# In[1]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install imblearn')
get_ipython().system('pip install xgboost')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
from datetime import datetime, date
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
pd.pandas.set_option('display.max_columns', None)


# ### Load given datasets

# In[3]:


df_test = pd.read_csv("C:\Python\Desktop\data science and ml edureka\data science assignment\Datasets MiniP2\Consumer_Complaints_test.csv")
df_train = pd.read_csv("C:\Python\Desktop\data science and ml edureka\data science assignment\Datasets MiniP2\Consumer_Complaints_train.csv")


# ### Print top 5 records of train dataset

# In[4]:


df_train.head()


# ### Print top 5 records of test dataset

# In[5]:


df_test.head(5)


# **Note: Please note that do all given tasks for test and train both datasets.**

# ### Print shape of train and test datasets 

# In[6]:


print('train data shape :',df_train.shape)
print('test data shape :',df_test.shape)


# ### Print columns of train and test datasets

# In[7]:


print('train data column :',df_train.columns)
print('test data column :',df_test.columns)


# ### Check data type for both datasets

# In[8]:


data_types_train = pd.DataFrame(df_train.dtypes, columns = ['Train'])
data_types_test = pd.DataFrame(df_test.dtypes, columns = ['Test'])
data_types = pd.concat([data_types_train, data_types_test], axis = 1)
data_types


# ### Print missing values in percentage

# In[9]:


missing_values_train = pd.DataFrame(df_train.isna().sum(), columns = ['Train'])
missing_values_test = pd.DataFrame(df_test.isna().sum(), columns = ['Test'])
missing_values = pd.concat([missing_values_train, missing_values_test], axis = 1)
missing_values


# ### Drop columns where more than 25% of the data are missing.

# In[10]:


columns_with_missing_values = ['Sub-product', 'Sub-issue', 'Consumer complaint narrative', 'Company public response', 'Tags', 'Consumer consent provided?']
df_train = df_train.drop(columns_with_missing_values, axis = 1)
df_test = df_test.drop(columns_with_missing_values, axis = 1)


# ### Extract Date, Month, and Year from the "Date Received" Column and create new fields for year, month, and day.
# 
# like, df_train['Year_Received'] = df_train['Date received']........(logic)

# In[11]:


df_train['Year_Received'] = df_train['Date received'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').year)
df_test['Year_Received'] = df_test['Date received'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').year)
df_train['Month_Received'] = df_train['Date received'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').month)
df_test['Month_Received'] = df_test['Date received'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').month)
df_train['Day_Received'] = df_train['Date received'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').day)
df_test['Day_Received'] = df_test['Date received'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').day)


# ### Convert dates from object type to datetime type

# In[12]:


df_train['Date received'] = pd.to_datetime(df_train['Date received'])
df_test['Date received'] = pd.to_datetime(df_test['Date received'])
df_train['Date sent to company'] = pd.to_datetime(df_train['Date sent to company'])
df_test['Date sent to company'] = pd.to_datetime(df_test['Date sent to company'])


# ### Calculate the number of days the complaint was with the company
# 
# create new field with help given logic<br>
# Like, Days held = Date sent to company - Date received

# In[13]:


df_train['Days held'] = df_train['Date sent to company'] - df_train['Date received']
df_test['Days held'] = df_test['Date sent to company'] - df_test['Date received']


# ### Convert "Days Held" to Int(above column)

# In[14]:


df_train['Days held'] = df_train['Days held'].astype('timedelta64[D]').astype(int)
df_test['Days held'] = df_test['Days held'].astype('timedelta64[D]').astype(int)


# ### Drop "Date Received","Date Sent to Company","ZIP Code", "Complaint ID"

# In[15]:


df_train = df_train.drop(['Date received', 'Date sent to company','ZIP code', 'Complaint ID'], axis = 1)
df_test = df_test.drop(['Date received', 'Date sent to company','ZIP code', 'Complaint ID'], axis = 1)


# ### Impute null values in "State" by Mode 
# (find mode and replace nan value)

# In[16]:


df_train['State'].mode(), df_test['State'].mode()


# In[17]:


df_train['State'] = df_train['State'].replace(np.nan, 'CA')
df_test['State'] = df_test['State'].replace(np.nan, 'CA')


# ### Check Missing Values in the dataset

# In[18]:


missing_values_train = pd.DataFrame(df_train.isna().sum(), columns = ['Train'])
missing_values_test = pd.DataFrame(df_test.isna().sum(), columns = ['Test'])
missing_values = pd.concat([missing_values_train, missing_values_test], axis = 1)
missing_values


# ### Categorize Days into Weeks with the help of 'Days Received'

# In[19]:


week_train = []
for i in df_train['Day_Received']:
    if i < 8:
        week_train.append(1)
    elif i >= 8 and i < 16:
        week_train.append(2)
    elif i >=16 and i < 22:
        week_train.append(3)
    else:
        week_train.append(4)
df_train['Week_Received'] = week_train
week_test = []
for i in df_test['Day_Received']:
    if i < 8:
        week_test.append(1)
    elif i >= 8 and i < 16:
        week_test.append(2)
    elif i >=16 and i < 22:
        week_test.append(3)
    else:
        week_test.append(4)
df_test['Week_Received'] = week_test


# ### Drop "Day_Received" column

# In[20]:


df_train = df_train.drop(['Day_Received'], axis = 1)
df_test = df_test.drop(['Day_Received'], axis = 1)


# ### Print head of train and test dataset and observe

# In[21]:


df_train.head()


# In[22]:


df_test.head()


# ### Store data of the disputed consumer in the new data frame as "disputed_cons"

# In[23]:


disputed_cons = df_train[df_train['Consumer disputed?'] == 'Yes'] 


# ### Plot bar graph for the total no of disputes with the help of seaborn

# In[24]:


sns.countplot(x = 'Consumer disputed?', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Consumer disputed', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# ### Plot bar graph for the total no of disputes products-wise with help of seaborn

# In[25]:


sns.countplot(x = 'Product', hue = 'Consumer disputed?', data = df_train)
plt.xticks(rotation = 90, fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Product', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


#     Roughly 37% consumer with mortgage have disputed.
#     Approx. 54% consumer who have disputed are from mortgage or debt collection.
#     68% consumer having disputes are from mortgage, debt collection and credit reporting.
#     Adding credit card consumers and bank account or services consumer will make it 80% and 91% respectively.

# ### Plot bar graph for the total no of disputes with Top Issues by Highest Disputes , with help of seaborn

# In[26]:


top_issues_disputes = disputed_cons['Issue'].value_counts().sort_values(ascending = False).head(10)
sns.barplot(x = top_issues_disputes.index, y = top_issues_disputes.values)
plt.xticks(rotation = 90, fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Issues', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# ### Plot bar graph for the total no of disputes by State with Maximum Disputes

# In[27]:


fig, ax = plt.subplots(figsize=(20, 10))
sns.countplot(x = df_train['State'])
plt.xticks(rotation = 90, fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('State', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


#     15% disputes from CA.
#     25% disputes from CA and FL
#     32% disputes from CA, FL and TX.
#     38% disputes from CA, FL, TX and NY.
#     43% disputes from CA, FL, TX, NY and GA.
#     47% disputes from CA, FL, TX, NY, GA and NJ.
#     50% disputes from CA, FL, TX, NY, GA, NJ and IL.
#     54% disputes from CA, FL, TX, NY, GA, NJ, IL and VA.
#     57% disputes from CA, FL, TX, NY, GA, NJ, IL, VA and PA.
#     61% disputes from CA, FL, TX, NY, GA, NJ, IL, VA, PA and MD.

# ### Plot bar graph for the total no of disputes by Submitted Via diffrent source 

# In[28]:


sns.countplot(x = 'Submitted via', data = disputed_cons)
plt.xticks(rotation = 90, fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Submitted Via', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


#     72% disputes are submitted via web.
#     88% disputes are submitted via web and referral.

# ### Plot bar graph for the total no of disputes wherevCompany's Response to the Complaints

# In[29]:


sns.countplot(x = 'Company response to consumer', data = df_train)
plt.xticks(rotation = 90, fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Company Response', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


#  74% complaints are closed with explanation.

# ### Plot bar graph for the total no of disputes where Company's Response Leading to Disputes

# In[30]:


sns.countplot(x = 'Company response to consumer', data = disputed_cons)
plt.xticks(rotation = 90, fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Company Response', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


#     82% disputes are closed with explanation at the initial stage.
#     89% disputes are either closed with explanation or non-monetary relief in the earlier stage.

# ### Plot bar graph for the total no of disputes Whether there are Disputes Instead of Timely Response

# In[31]:


sns.countplot(x = 'Timely response?', data = disputed_cons)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Timely Response', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


#  98% disputes were timely repsonded at the intial stages.

# ### Plot bar graph for the total no of disputes over Year Wise Complaints

# In[32]:


sns.countplot(x = 'Year_Received', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


#     28% complaints are raised in 2015.
#     53% complaints are raised in 2014 and 2015.
#     71% complaints are raised in 2013 to 2015.
#     88% complaints are raised in 2013 to 2016.

# ### Plot bar graph for the total no of disputes over Year Wise Disputes

# In[33]:


sns.countplot(x = 'Year_Received', data = disputed_cons)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


#     27% disputes are raised in 2015.
#     50% disputes are raised in 2014 and 2015.
#     69% disputes are raised in 2014 to 2016.
#     87% disputes are raised in 2013 to 2016.

# ### Plot  bar graph for the top companies with highest complaints

# In[34]:


worst_company_disputes = disputed_cons['Company'].value_counts().sort_values(ascending = False).head(10)
sns.barplot(x = worst_company_disputes.index, y = worst_company_disputes.values)
plt.xticks(rotation = 90, fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Company', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# 53% disputes are for these 10 companies.

# ### "Days Held" Column Analysis(describe)

# In[35]:


df_train['Days held'].describe()


# In[36]:


df_test['Days held'].describe()


# ### Convert Negative Days Held to Zero(it is the time taken by authority can't be negative)

# In[37]:


Days_held_train = []
for i in df_train['Days held']:
    if i < 0:
        Days_held_train.append(0)
    else:
        Days_held_train.append(i)
df_train['Days_held'] = Days_held_train
Days_held_test = []
for i in df_test['Days held']:
    if i < 0:
        Days_held_test.append(0)
    else:
        Days_held_test.append(i)
df_test['Days_held'] = Days_held_test


# ### Drop Days Held with Negative Values

# In[38]:


df_train = df_train.drop('Days held', axis = 1)
df_test = df_test.drop('Days held', axis = 1)


# # Text pre-processing
# (It will be cover in upcoming calsses)

# In[39]:


import nltk
nltk.download()


# In[40]:


from nltk.corpus import brown
brown.words()
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]


# In[41]:


nltk.set_proxy('http://proxy.example.com:3128', ('USERNAME', 'PASSWORD'))
nltk.download()


# In[42]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[43]:


relevant_text_train = df_train['Issue']
relevant_text_test = df_test['Issue']
tokenized_data_train = relevant_text_train.apply(lambda x: wordpunct_tokenize(x.lower()))
tokenized_data_test = relevant_text_test.apply(lambda x: wordpunct_tokenize(x.lower()))
def remove_punctuation(text):
    no_punctuation = []
    for w in text:
        if w not in string.punctuation:
            no_punctuation.append(w)
    return no_punctuation
no_punctuation_data_train = tokenized_data_train.apply(lambda x: remove_punctuation(x))
no_punctuation_data_test = tokenized_data_test.apply(lambda x: remove_punctuation(x))
stop_words = stopwords.words('english')
filtered_sentence_train = [w for w in no_punctuation_data_train if not w in stop_words]
filtered_sentence_train = pd.Series(filtered_sentence_train)
filtered_sentence_test = [w for w in no_punctuation_data_test if not w in stop_words]
filtered_sentence_test = pd.Series(filtered_sentence_test)
def lemmatize_text(text):
    lem_text = [WordNetLemmatizer().lemmatize(w,pos = 'v') for w in text]
    return lem_text
lemmatized_data_train = filtered_sentence_train.apply(lambda x:lemmatize_text(x))
lemmatized_data_test = filtered_sentence_test.apply(lambda x:lemmatize_text(x))
def stem_text(text):
    stem_text = [PorterStemmer().stem(w) for w in text]
    return stem_text
stemmed_data_train = lemmatized_data_train.apply(lambda x:stem_text(x))
stemmed_data_test = lemmatized_data_test.apply(lambda x:stem_text(x))
def word_to_sentence(text):
    text_sentence = " ".join(text)
    return text_sentence
clean_data_train = stemmed_data_train.apply(lambda x:word_to_sentence(x))
clean_data_test = stemmed_data_test.apply(lambda x:word_to_sentence(x))


# In[44]:


df_train['Issues_cleaned'] = clean_data_train
df_test['Issues_cleaned'] = clean_data_test
df_train = df_train.drop('Issue', axis = 1)
df_test = df_test.drop('Issue', axis = 1)


# ### Drop Unnecessary Columns for the Model Building<br>
# like:'Company', 'State', 'Year_Received', 'Days_held'

# In[45]:


drop_cols = ['Company', 'State', 'Year_Received', 'Days_held']
df_train = df_train.drop(drop_cols, axis = 1)
df_test = df_test.drop(drop_cols, axis = 1)


# ### Change Consumer Disputed Column to 0 and 1(yes to 1, and no to 0)

# In[46]:


df_train['Consumer disputed?'] = np.where(df_train['Consumer disputed?'] == "Yes", 1, 0)


# ### Create Dummy Variables for catagorical features 
# like: 'Product', 'Submitted via', 'Company response to consumer', 'Timely response?'

# In[47]:


dum_cols = ['Product', 'Submitted via', 'Company response to consumer', 'Timely response?']
df_train_dummies = pd.get_dummies(df_train[dum_cols], prefix_sep = '_', drop_first = True)
df_test_dummies = pd.get_dummies(df_test[dum_cols], prefix_sep = '_', drop_first = True)


# ### Concate Dummy Variables and Drop the Original Columns

# In[48]:


df_train = df_train.drop(dum_cols, axis = 1)
df_test = df_test.drop(dum_cols, axis = 1)
df_train = pd.concat([df_train, df_train_dummies], axis = 1)
df_test = pd.concat([df_test, df_test_dummies], axis = 1)


# ### Calculating TF-IDF

# In[49]:


tf = TfidfVectorizer()
issues_cleaned_train = tf.fit_transform(df_train['Issues_cleaned']).toarray()
issues_cleaned_test = tf.fit_transform(df_test['Issues_cleaned']).toarray()
tf_columns_train = []
tf_columns_test = []
for i in range(issues_cleaned_train.shape[1]):
    tf_columns_train.append('Feature' + str(i+1))
for i in range(issues_cleaned_test.shape[1]):
    tf_columns_test.append('Feature' + str(i+1))
issues_train = pd.DataFrame(issues_cleaned_train, columns = tf_columns_train)
issues_test = pd.DataFrame(issues_cleaned_test, columns = tf_columns_test)
weights = pd.DataFrame(tf.idf_, index = tf.get_feature_names(), columns = ['Idf_weights']).sort_values(by = 'Idf_weights', ascending = False)
weights.head()


# ### Replacing Issues_cleaned by Vectorized Issues

# In[50]:


df_train = df_train.drop('Issues_cleaned', axis = 1)
df_test = df_test.drop('Issues_cleaned', axis = 1)
df_train = pd.concat([df_train, issues_train], axis = 1)
df_test = pd.concat([df_test, issues_test], axis = 1)
Feature168 = [0] * 119606
df_test['Feature168'] = Feature168


# ### observe train and test datasets

# In[52]:


df_train.head()


# In[53]:


df_test.head()


# ### Observe Shape of new Train and Test Datasets

# In[54]:


df_train.shape, df_test.shape


# ### Scaling the Data Sets (note:discard dependent variable before doing standardization)

# In[55]:


df_train_scaled = pd.DataFrame(StandardScaler().fit_transform(df_train.drop('Consumer disputed?', axis = 1)), columns = df_test.columns)
df_test_scaled = pd.DataFrame(StandardScaler().fit_transform(df_test), columns = df_test.columns)


# ### Do feature selection with help of PCA

# In[56]:


pca_columns = []
for i in range(df_train_scaled.shape[1]):
    pca_columns.append('PC' + str(i+1))
pca_model = PCA()
pca_model.fit(df_train_scaled)
df_pca_train = pd.DataFrame(pca_model.transform(df_train_scaled), columns = pca_columns)
explained_info_train = pd.DataFrame(pca_model.explained_variance_ratio_, columns=['Explained Info']).sort_values(by = 'Explained Info', ascending = False)
imp = []
for i in range(explained_info_train.shape[0]):
    imp.append(explained_info_train.head(i).sum())
explained_info_train_sum = pd.DataFrame()
explained_info_train_sum['Variable'] = pca_columns
explained_info_train_sum['Importance'] = imp
explained_info_train_sum.head(60)


# So 53 variables are making upto 80% of the information.

# In[57]:


pca_columns = []
for i in range(53):
    pca_columns.append('PC' + str(i+1))
pca_model = PCA(n_components = 53)
pca_model.fit(df_train_scaled)
df_pca_train = pd.DataFrame(pca_model.transform(df_train_scaled), columns = pca_columns)


# In[58]:


df_pca_train.head()


# ### Select top features which are covering 80% of the information 
# (n=53),
# <br>store this data into new dataframe,

# In[60]:


pca_model = PCA(n_components = 53)
pca_model.fit(df_test_scaled)
df_pca_test = pd.DataFrame(pca_model.transform(df_test_scaled), columns = pca_columns)


# ### Split the Data Sets Into X and Y by dependent and independent variables (data selected by PCA)
# 

# In[61]:


X = df_pca_train
y = df_train['Consumer disputed?']


# ### Split data into Train and Test datasets
# (for test data use test excel file data)

# In[64]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 17)
X_test = df_pca_test


# ### Shapes of the datasets

# In[65]:


X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape


# **Model building**
# Build given models and mesure their test and validation accuracy 
# build given models:
# 1. LogisticRegression
# 2. DecisionTreeClassifier
# 3. RandomForestClassifier
# 4. AdaBoostClassifier
# 5. GradientBoostingClassifier
# 6. KNeighborsClassifier
# 7. XGBClassifier

# In[66]:


models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), KNeighborsClassifier(), XGBClassifier()]
model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'KNeighborsClassifier', 'XGBClassifier']
accuracy_train = []
accuracy_val = []
for model in models:
    mod = model
    mod.fit(X_train, y_train)
    y_pred_train = mod.predict(X_train)
    y_pred_val = mod.predict(X_val)
    accuracy_train.append(accuracy_score(y_train, y_pred_train))
    accuracy_val.append(accuracy_score(y_val, y_pred_val))
data = {'Modelling Algorithm' : model_names, 'Train Accuracy' : accuracy_train, 'Validation Accuracy' : accuracy_val}
data = pd.DataFrame(data)
data['Difference'] = ((np.abs(data['Train Accuracy'] - data['Validation Accuracy'])) * 100)/(data['Train Accuracy'])
data.sort_values(by = 'Validation Accuracy', ascending = False)


#  LogisticRegression is the best model to build the model.

# ### Final Model and Prediction for test data file

# In[67]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_test = lr.predict(X_test)
y_pred_test = pd.DataFrame(y_pred_test, columns = ['Prediction'])
y_pred_test.head()


# ### Export Predictions to CSV

# In[68]:


y_pred_test.to_csv('Prediction.csv')


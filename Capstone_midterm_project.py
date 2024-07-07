#!/usr/bin/env python
# coding: utf-8

# In[161]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,recall_score,accuracy_score
import seaborn as sns


# In[162]:


import pandas as pd

# Define the file paths
input_file = 'risk-train.txt'  # Replace with your input file path
output_file = 'output.csv'  # Replace with your desired output file path

# Read the text file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Extract headers
headers = lines[0].strip().split(',')

# Extract data rows
data = []
for line in lines[1:]:
    row = line.strip().split(',')
    data.append(row)

# Create a DataFrame
df = pd.DataFrame(data, columns=headers)

# Handle any potential data type issues, such as converting date strings to datetime
# For example, convert 'B_BIRTHDATE' column to datetime, if necessary:
df['B_BIRTHDATE'] = pd.to_datetime(df['B_BIRTHDATE'], errors='coerce')
df.replace('?', pd.NA, inplace=True)
# Save to CSV
df.to_csv(output_file, index=False)

print(f"Data successfully converted to {output_file}")


# In[163]:


data1 = pd.read_csv('output.csv')
data1


# In[164]:


data1.isna().sum()


# when i analysised the both columns they have the relation between both the columns i came to know that when there is a debitnote and check then there is Question mark ? so i replaced that with Noot applicable 

# In[165]:


data1.loc[data1['Z_METHODE'].isin(['check', 'debit_note']),'Z_CARD_ART']='Not Applicable'


# In[166]:


data1[['Z_METHODE','Z_CARD_ART']]


# In[167]:


data1 = data1.dropna(subset=['B_BIRTHDATE'])


# In[168]:


data1['B_BIRTHDATE'].isna().sum()


# In[169]:


data1['B_BIRTHDATE'] = pd.to_datetime(data1['B_BIRTHDATE'], errors='coerce')


data1['YEAR'] = data1['B_BIRTHDATE'].dt.year
data1['MONTH'] = data1['B_BIRTHDATE'].dt.month
data1['DAY'] = data1['B_BIRTHDATE'].dt.day


# In[170]:


data1[['ANUMMER_01','ANUMMER_02','ANUMMER_03','ANUMMER_04','ANUMMER_05','ANUMMER_06','ANUMMER_07','ANUMMER_08','ANUMMER_09','ANUMMER_10']].head(15)


# there are number of NAN Values in the columns from ANUMMER_02-10 so i am planning to drop this columns

# In[171]:


data1.drop(columns={'ANUMMER_02','ANUMMER_03','ANUMMER_04','ANUMMER_05','ANUMMER_06','ANUMMER_07','ANUMMER_08','ANUMMER_09','ANUMMER_10'},inplace=True)


# In[172]:


data1.isna().sum()


# In[173]:


data1.loc[data1['NEUKUNDE'] == 'yes', 'DATE_LORDER'] = None


# In[174]:


import pandas as pd

# Check for duplicate columns
duplicate_columns = data1.columns.duplicated()
print("Duplicate columns:", data1.columns[duplicate_columns])

# Drop duplicate columns (if any)
data1 = data1.loc[:, ~data1.columns.duplicated()]

# Rename 'TIME_ORDER' to 'time_order' for consistency
data1.rename(columns={'TIME_ORDER': 'time_order'}, inplace=True)

# Convert time_order to datetime
data1['time_order'] = pd.to_datetime(data1['time_order'], format='%H:%M', errors='coerce')

# Extract hour and minute components from the datetime formatted 'time_order'
data1['hour'] = data1['time_order'].dt.hour
data1['minute'] = data1['time_order'].dt.minute

# Define function to categorize period of day
def get_period_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

# Apply function to categorize period of day
data1['period_of_day'] = data1['hour'].apply(get_period_of_day)

print(data1)


# In[175]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data1' is your DataFrame containing the training data
numerical_cols = data1.select_dtypes(include=[np.number]).columns

# Calculate the correlation matrix
corr_matrix = data1[numerical_cols].corr()

# Plot a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Identify features with high correlation (e.g., correlation > 0.9)
high_corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()
high_corr_pairs = high_corr_pairs[high_corr_pairs > 0.9]

print("Highly Correlated Feature Pairs:\n", high_corr_pairs)


# In[176]:


data1.drop(columns={'CHK_COOKIE','CHK_IP','FAIL_LPLZ','FAIL_RPLZ','FAIL_RORT','FAIL_RPLZORTMATCH'},inplace=True)


# In[177]:


data1.shape


# In[178]:


mean_value = data1['MAHN_AKT'].mean()
data1['MAHN_AKT'].fillna(mean_value, inplace=True)


# In[179]:


mean_value = data1['MAHN_HOECHST'].mean()
data1['MAHN_HOECHST'].fillna(mean_value, inplace=True)


# In[180]:


data1.loc[data1['NEUKUNDE'] == 'yes', 'DATE_LORDER'] = 'Not Applicable'


# In[181]:


data1.drop(columns={'Z_LAST_NAME'},inplace=True)


# In[182]:


data1.isna().sum()


# In[183]:


data1.dropna(axis=1, inplace=True)


# In[184]:


train_df = data1


# In[185]:


categorical_cols = train_df.select_dtypes(include=['object']).columns
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
print(categorical_cols)
print(numerical_cols)


# In[186]:


# 1. Convert object columns to categories
categorical_cols = ['CLASS', 'B_EMAIL', 'B_TELEFON', 'FLAG_LRIDENTISCH', 'FLAG_NEWSLETTER', 'Z_METHODE', 
                    'Z_CARD_ART', 'WEEKDAY_ORDER', 'CHK_LADR', 'CHK_RADR', 'CHK_KTO', 'CHK_CARD', 
                    'FAIL_LORT', 'FAIL_LPLZORTMATCH', 'NEUKUNDE', 'period_of_day']

for col in categorical_cols:
    data1[col] = data1[col].astype('category')


# In[187]:


data1[col]


# In[188]:


data1['AGE'] = data1['B_BIRTHDATE'].apply(lambda x: pd.Timestamp('now').year - pd.to_datetime(x).year)
data1 = data1.drop(['B_BIRTHDATE', 'YEAR', 'MONTH', 'DAY'], axis=1)


# In[189]:


numeric_cols = data1.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    data1[col].fillna(data1[col].median(), inplace=True)


# In[190]:


numeric_cols


# In[191]:


data1 = pd.get_dummies(data1, drop_first=True)


# In[192]:


data1


# In[193]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

continuous_cols = ['VALUE_ORDER', 'SESSION_TIME', 'AMOUNT_ORDER', 'AMOUNT_ORDER_PRE', 'VALUE_ORDER_PRE', 'MAHN_AKT', 'MAHN_HOECHST', 'AGE']


# In[194]:


data1.dtypes


# In[195]:


data1[continuous_cols] = scaler.fit_transform(data1[continuous_cols])




# In[238]:


train_df = data1
train_df.to_csv('cleaned_train.csv', index=False)


# In[196]:


# Assuming X contains features and y contains the target variable 'CLASS_yes'
X = data1.drop('CLASS_yes', axis=1)  # Features
y = data1['CLASS_yes']  # Target


# In[197]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier (you can choose another classifier as needed)
clf = RandomForestClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report for detailed metrics
print(classification_report(y_test, y_pred))


# In[198]:


print(confusion_matrix(y_test, y_pred))


# In[199]:


feature_importances = clf.feature_importances_

# Create a DataFrame to visualize feature importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

top_n = 20
top_features = feature_importance_df.head(top_n)

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.title('Top 20 Feature Importance')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Adjust layout to make room for rotated labels
plt.show()


# In[200]:


import pandas as pd

# Specify the file paths
input_file = 'risk-test.txt'   # Replace with the path to your input TSV file
output_file = 'orders.csv'  # Replace with the desired path for your output CSV file

# Read the TSV file
df = pd.read_csv(input_file, delimiter='\t')

# Handle missing values (optional)
# For example, replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)

print(f"File converted successfully from {input_file} to {output_file}")


# In[201]:


df.shape


# In[202]:


# Replace 'Z_CARD_ART' with 'Not Applicable' where 'Z_METHODE' is 'check' or 'debit_note'
df.loc[df['Z_METHODE'].isin(['check', 'debit_note']), 'Z_CARD_ART'] = 'Not Applicable'


# In[203]:


df[['Z_METHODE','Z_CARD_ART']]


# In[204]:


df = df.dropna(subset=['B_BIRTHDATE'])


# In[205]:


df['B_BIRTHDATE'].isna().sum()


# In[206]:


# Convert 'B_BIRTHDATE' to datetime with errors='coerce'
df['B_BIRTHDATE'] = pd.to_datetime(df['B_BIRTHDATE'], errors='coerce')

# Extract year, month, and day from 'B_BIRTHDATE'
df['YEAR'] = df['B_BIRTHDATE'].dt.year
df['MONTH'] = df['B_BIRTHDATE'].dt.month
df['DAY'] = df['B_BIRTHDATE'].dt.day


# In[207]:


df.drop(columns={'ANUMMER_02','ANUMMER_03','ANUMMER_04','ANUMMER_05','ANUMMER_06','ANUMMER_07','ANUMMER_08','ANUMMER_09','ANUMMER_10'},inplace=True)


# In[208]:


# Set 'DATE_LORDER' to None where 'NEUKUNDE' is 'yes'
df.loc[df['NEUKUNDE'] == 'yes', 'DATE_LORDER'] = None


# In[209]:


import pandas as pd

# Check for duplicate columns
duplicate_columns = df.columns.duplicated()
print("Duplicate columns:", df.columns[duplicate_columns])

# Drop duplicate columns (if any)
df = df.loc[:, ~df.columns.duplicated()]

# Rename 'TIME_ORDER' to 'time_order' for consistency
df.rename(columns={'TIME_ORDER': 'time_order'}, inplace=True)

# Convert time_order to datetime
df['time_order'] = pd.to_datetime(df['time_order'], format='%H:%M', errors='coerce')

# Extract hour and minute components from the datetime formatted 'time_order'
df['hour'] = df['time_order'].dt.hour
df['minute'] = df['time_order'].dt.minute

# Define function to categorize period of day
def get_period_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

# Apply function to categorize period of day
df['period_of_day'] = df['hour'].apply(get_period_of_day)

print(df)


# In[210]:


df


# In[211]:


df.drop(columns={'CHK_COOKIE','CHK_IP','FAIL_LPLZ','FAIL_RPLZ','FAIL_RORT','FAIL_RPLZORTMATCH'},inplace=True)


# In[212]:


df.shape


# In[213]:


unique_values = df['MAHN_AKT'].unique()
print(unique_values)


df['MAHN_AKT'] = pd.to_numeric(df['MAHN_AKT'], errors='coerce')


mean_value = df['MAHN_AKT'].mean()


df['MAHN_AKT'].fillna(mean_value, inplace=True)



# In[214]:


# Check unique values in 'MAHN_AKT' column
unique_values = df['MAHN_HOECHST'].unique()
print(unique_values)

# Example of how to handle the mean calculation
# Replace non-numeric values with NaN
df['MAHN_HOECHST'] = pd.to_numeric(df['MAHN_HOECHST'], errors='coerce')

# Calculate mean after converting to numeric
mean_value = df['MAHN_HOECHST'].mean()

# Fill NaN values with mean_value
df['MAHN_HOECHST'].fillna(mean_value, inplace=True)

# Proceed with further preprocessing or analysis


# In[215]:


# Set 'DATE_LORDER' to 'Not Applicable' where 'NEUKUNDE' is 'yes'
df.loc[df['NEUKUNDE'] == 'yes', 'DATE_LORDER'] = 'Not Applicable'


# In[216]:


df.drop(columns={'Z_LAST_NAME'},inplace=True)


# In[217]:


df.dropna(axis=1, inplace=True)


# In[218]:


df.isna().sum()


# In[219]:


test_df = df


# In[220]:


test_df.shape


# In[221]:


categorical_cols = test_df.select_dtypes(include=['object']).columns
numerical_cols = test_df.select_dtypes(include=[np.number]).columns
print(categorical_cols)
print(numerical_cols)


# In[222]:


# 1. Convert object columns to categories
categorical_cols = ['B_EMAIL','B_TELEFON', 'FLAG_LRIDENTISCH', 'FLAG_NEWSLETTER', 'Z_METHODE', 
                    'Z_CARD_ART', 'WEEKDAY_ORDER', 'CHK_LADR', 'CHK_RADR', 'CHK_KTO', 'CHK_CARD', 
                    'FAIL_LORT', 'FAIL_LPLZORTMATCH', 'NEUKUNDE', 'period_of_day']

for col in categorical_cols:
    df[col] = df[col].astype('category')


# In[223]:


df[col]


# In[224]:


# Calculate age based on 'B_BIRTHDATE'
df['AGE'] = df['B_BIRTHDATE'].apply(lambda x: pd.Timestamp('now').year - pd.to_datetime(x).year)

# Drop 'B_BIRTHDATE', 'YEAR', 'MONTH', 'DAY' columns
df = df.drop(['B_BIRTHDATE', 'YEAR', 'MONTH', 'DAY'], axis=1)


# In[225]:


import pandas as pd

# Assuming df is your DataFrame
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Fill missing values with median for numeric columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)


# In[226]:


numeric_cols


# In[227]:


df = pd.get_dummies(df, drop_first=True)


# In[228]:


df


# In[229]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

continuous_cols = ['VALUE_ORDER', 'SESSION_TIME', 'AMOUNT_ORDER', 'AMOUNT_ORDER_PRE', 'VALUE_ORDER_PRE', 'MAHN_AKT', 'MAHN_HOECHST', 'AGE']


# In[230]:


df.dtypes


# In[231]:


df[continuous_cols] = scaler.fit_transform(df[continuous_cols])


# In[240]:


test_df = df
test_df.to_csv('cleaned_test.csv', index=False)


# In[244]:


train_df.dtypes


# In[246]:


test_df.dtypes


# In[252]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

X_test_new = test_df

# Ensure X_test_new has the same columns as X_train after preprocessing
X_test_new = pd.get_dummies(X_test_new)  # One-hot encoding for categorical variables
X_test_new = X_test_new.reindex(columns=X.columns, fill_value=0)

# Predict using the trained classifier
y_pred_new = clf.predict(X_test_new)

# Prepare results dataframe with ORDER_ID and predicted CLASS
result_df = pd.DataFrame({'ORDER_ID': test_df['ORDER_ID'], 'CLASS': y_pred_new})

# Save predictions to a CSV file (optional)
# result_df.to_csv('predicted_classes.csv', index=False)

# Print shape of result_df for sanity check
print(result_df.shape)

# Calculate accuracy (sanity check using predicted labels)
# Since we don't have true labels for test set, this is just for illustration
accuracy_sanity = accuracy_score(y_pred_new, y_pred_new)  # Using predictions as "true" labels
print(f'Accuracy (Sanity Check): {accuracy_sanity:.1f}')

# Generate confusion matrix (sanity check)
conf_matrix_sanity = confusion_matrix(y_pred_new, y_pred_new)
print('Confusion Matrix (Sanity Check):')
print(conf_matrix_sanity)

# Plot confusion matrix (sanity check)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_sanity, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix (Sanity Check)')
plt.show()


# In[ ]:





# In[ ]:





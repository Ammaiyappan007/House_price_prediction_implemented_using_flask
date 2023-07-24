import numpy as np
import pandas as pd
df = pd.read_csv("C:\\Users\\Mariyaan\\Downloads\\archive (7)\\Bengaluru_House_Data.csv")
#df.head()
df1 = df.drop(['area_type','availability','society','balcony'],axis = 'columns')
#df1.head()
df2 = df1.dropna()
#df2.isnull().sum()
df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))
#df2.head()
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
df2[~df2['total_sqft'].apply(is_float)]
def convert_num_to_float(x):
    token = x.split("-")
    if len(token) == 2:
        return (float(token[0])+float(token[1]))//2
    try:
        return float (x)
    except:
        return None
df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_num_to_float)
#phase 2 - feature Engineering
df4 = df3.copy()
df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft']
df4.location = df4['location'].apply(lambda x: x.strip())
location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<10]
df4['location'] = df4['location'].apply(lambda x:"others" if x in location_stats_less_than_10 else x)
#phase 3 - Removing outliers
# We are using 300 as certain threshold value
df5 = df4[~(df4.total_sqft/df4.bhk<300)]
df5.dropna()
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key,sub_df in df.groupby('location'):
        m = np.mean(sub_df.price_per_sqft)
        st = np.std(sub_df.price_per_sqft)
        reduced_df = sub_df[(sub_df.price_per_sqft > (m-st)) & (sub_df.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index = True)
    return df_out
df6 = remove_pps_outliers(df5)
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df7 = remove_bhk_outliers(df6)
# df8 = df7.copy()
df8 = df7[df7.bath<df7.bhk+2]
df9 = df8.drop(['size','price_per_sqft'],axis='columns')
#phase 4 - model selection and creation
#One hot encoding
dummies = pd.get_dummies(df9.location)
dummies = dummies.replace(True,1)
dummies = dummies.replace(False,0)
df10 = pd.concat([df9,dummies.drop('others',axis='columns')],axis='columns')
df11 = df10.drop('location',axis = 'columns')
X = df11.drop(['price'],axis='columns')
y = df11.price
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
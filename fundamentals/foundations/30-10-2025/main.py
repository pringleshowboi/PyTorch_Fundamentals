import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#import data
reviews = pd.read_csv('reviews.csv')
 
#print column names
print(reviews.columns)
print('---')
 
#print .info
print(reviews.info())
print('---')
#look at the counts of recommended
print(reviews['recommended'].value_counts())
print('---')
#create binary dictionary
binary_dict = {True: 1, False: 0}
 
#transform column
reviews['recommeded_binary_dict'] = reviews['recommended'].map(binary_dict)

#print your transformed column
print(reviews['recommeded_binary_dict'].value_counts())
print('---')
#look at the counts of rating
print(reviews['rating'].value_counts())
print('---')
#create dictionary
rating_dict = {'Loved it': 5,
               'Liked it': 4,
               'Was okay': 3,
               'Not great': 2,
               'Hated it': 1}
 
#transform rating column
reviews['rating_transformed'] = reviews['rating'].map(rating_dict)

#print your transformed column values
print(reviews['rating_transformed'].value_counts())
print('---')

#get the number of categories in a feature
print(reviews['department_name'].value_counts())
print('---')
#perform get_dummies
one_hot = pd.get_dummies(reviews['department_name'])
 
#join the new columns back onto the original
reviews.join(one_hot)

#print column names
print(reviews.columns)
print('---')
#transform review_date to date-time data
reviews['review_date_transformed'] = pd.to_datetime(reviews['review_date'])

#print review_date data type 
print(reviews['review_date_transformed'].dtype)
print('---')
#get numerical columns
numerical_cols = reviews.select_dtypes(include=np.number).columns.tolist()
print("Numerical columns: ", numerical_cols)
print('---')
#reset index
reviews = reviews.reset_index(drop=True)

# 17. Instantiate standard scaler
scaler = StandardScaler()

# 18. Fit and transform ONLY the numerical columns
numerical_data_for_scaler = reviews[numerical_cols]
scaled_reviews = scaler.fit_transform(numerical_data_for_scaler)

# The result is a NumPy array. You can convert it back to a DataFrame for convenience.
scaled_reviews_df = pd.DataFrame(scaled_reviews, columns=numerical_cols)

print("Scaled DataFrame (first 5 rows):")
print(scaled_reviews_df.head())




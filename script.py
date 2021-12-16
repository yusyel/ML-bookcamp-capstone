# %%
import pandas as pd
import numpy as np 

# %%
df = pd.read_csv('CO2 Emissions_Canada.csv')


# %%
df.head()

# %%
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.head()

# %%
cat = list(df.dtypes[df.dtypes == 'object'].index)

for c in cat:
    df[c] = df[c].str.lower().str.replace(' ', '_')



# %%
df.vehicle_class.value_counts()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df.nunique()

# %%
df.dtypes

# %%
df.fuel_type.value_counts()

# %%
df = df.rename(columns={'engine_size(l)': 'engine_size',
               'fuel_consumption_city_(l/100_km)': 'fuel_consumption_city',
                        'fuel_consumption_hwy_(l/100_km)': 'fuel_consumption_hwy',
                        'fuel_consumption_comb_(l/100_km)': 'fuel_consumption_comb',
                        'fuel_consumption_comb_(mpg)': 'fuel_consumption_comb_mpg',
                        'co2_emissions(g/km)': 'co2_emissions'})


# %%
df.head()

# %%
sns.histplot(df.co2_emissions, bins=100)

# %%
sns.histplot(np.log(df.co2_emissions), bins = 100)

# %%
df['log_co2'] = np.log(df.co2_emissions)
del df['co2_emissions']

# %% [markdown]
# # Setting up the Validation Framework

# %%
from sklearn.model_selection import train_test_split

# %%
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_test), len(df_val)

# %%
df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %% [markdown]
# # 1.Exploratory Data Analysis

# %% [markdown]
# ### 1.1 Checking NA

# %%
df.isna().sum()

# %%
df.head()

# %%
df.nunique()

# %% [markdown]
# ### 1.2 Exploratory Data Analysis For Categorical Variable

# %% [markdown]
# #### 1.2.1 Car Manufacturer

# %%
plt.figure(figsize=(15,10))
sns.countplot(y = "make", data=df_full_train, orient='v' )

# %% [markdown]
# #### 1.2.2 Transmission

# %%
plt.figure(figsize=(15,8))
sns.boxplot(x = df_full_train.transmission, y = np.exp(df_full_train.log_co2), data = df_full_train)
plt.xticks(rotation=45)
plt.ylabel("CO^2 Emission")
plt.xlabel("Transmission")

# %% [markdown]
# #### 1.2.3 Vehicle Class

# %%
plt.figure(figsize=(15,8))
sns.boxplot(x = df_full_train.vehicle_class, y = np.exp(df_full_train.log_co2), data = df_full_train)
plt.xticks(rotation=45)
plt.ylabel("CO^2 Emission")
plt.xlabel("Vehicle Class")



# %% [markdown]
# #### 1.2.4 Fuel Type

# %%

plt.figure(figsize=(15,8))
sns.boxplot(x = df_full_train.fuel_type, y = np.exp(df_full_train.log_co2), data = df_full_train)
plt.xticks(rotation=45)
plt.ylabel("CO^2 Emission")
plt.xlabel("Fuel Type")

# %% [markdown]
# ### 1.3 Exploratory Data Analysis For Numerical Variable

# %% [markdown]
# #### 1.3.1 Engine Size

# %%
plt.figure(figsize=(10,8))
sns.lineplot(data=df_full_train, x = df_full_train.engine_size, y = np.exp(df_full_train.log_co2))
plt.ylabel("CO^2 Emission")
plt.xlabel("Engine Size")

# %%
df_full_train[
	df_full_train['engine_size'] == 8
]

# %% [markdown]
# #### 1.3.2 Cylinders

# %%
plt.figure(figsize=(10,8))
sns.lineplot(data=df_full_train, x = df_full_train.cylinders, y = np.exp(df_full_train.log_co2))
plt.ylabel("CO^2 Emission")
plt.xlabel("Cylinders")

# %%
df_full_train[
	df_full_train['cylinders'] == 16
]

# %% [markdown]
# #### 1.3.3 Transmission

# %%
plt.figure(figsize=(10,8))
sns.lineplot(data=df_full_train, x = df_full_train.transmission, y = np.exp(df_full_train.log_co2))
plt.xticks(rotation=45)
plt.ylabel("CO^2 Emission")
plt.xlabel("Transmission")

# %% [markdown]
# #### 1.3.4 Fuel Consumpiton for City

# %%
plt.figure(figsize=(10,8))
sns.lineplot(data=df_full_train, x = df_full_train.fuel_consumption_city, y = np.exp(df_full_train.log_co2))
plt.xticks(rotation=45)
plt.ylabel("CO^2 Emission")
plt.xlabel("fuel_consumption_city (l/100_km)")

# %%
df_cylinders = df_full_train[
	df_full_train['fuel_consumption_city'] <= 17
]

# %%
df_cylinders2 = df_full_train[
	df_full_train['fuel_consumption_city'] > 17
]

# %%
plt.figure(figsize=(8,6))
sns.countplot(y = "fuel_type", data=df_cylinders, palette="Set2")

# %%
plt.figure(figsize=(8,6))
sns.countplot(y = "fuel_type", data=df_cylinders2, palette="Set1")

# %% [markdown]
# #### 1.3.5 Fuel Consumption Highway

# %%
plt.figure(figsize=(10,8))
sns.lineplot(data=df_full_train, x = df_full_train.fuel_consumption_hwy, y = np.exp(df_full_train.log_co2))
plt.xticks(rotation=45)
plt.ylabel("CO^2 Emission")
plt.xlabel("fuel_consumption_hwy (l/100_km)")

# %% [markdown]
# #### 1.3.6 Fuel Consumpiton Combined (55% city & 45% highway)

# %%
plt.figure(figsize=(10,8))
sns.lineplot(data=df_full_train, x = df_full_train.fuel_consumption_comb, y = np.exp(df_full_train.log_co2))
plt.xticks(rotation=45)
plt.ylabel("CO^2 Emission")
plt.xlabel("fuel_consumption_comb (l/100_km)")

# %% [markdown]
# #### 1.3.7 Fuel Consumption Combined Miles Per Gallon

# %%
plt.figure(figsize=(10,8))
sns.lineplot(data=df_full_train, x = df_full_train.fuel_consumption_comb_mpg, y = np.exp(df_full_train.log_co2))
plt.xticks(rotation=45)
plt.ylabel("CO^2 Emission")
plt.xlabel("fuel_consumption_comb (mpg)")

# %% [markdown]
# ### 1.4 Exploratory Data Analysis For Correlated Data

# %%

num = ['engine_size', 'cylinders', 'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb', 'fuel_consumption_comb_mpg']

# %%

plt.figure(figsize=(10,7))  
sns.heatmap(df[num].corr(),annot=True,linewidths=.5)
plt.show()

# %% [markdown]
# ### 1.4 Models

# %% [markdown]
# #### 1.4.1 Linear Regression

# %%
selected=['vehicle_class', 'engine_size', 'cylinders', 'fuel_type', 'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb_mpg', 'log_co2']

# %%
df_train = df_train[selected]
df_val = df_val[selected]
df_test = df_test[selected]

# %%
df_train.nunique()

# %%
df_val.nunique()

# %%
df_val.fuel_type.value_counts()

# %%
df_val.drop(df_val[
	df_val.fuel_type == 'n'
].index, inplace = True)

# %%
df_train[
	df_train.cylinders == 16
]

# %%
df_train.drop(df_train[
	df_train.cylinders == 16
].index, inplace = True)

# %%
df_val.drop(df_val[
	df_val.cylinders == 16
].index, inplace = True)


# %%
df_train.cylinders.value_counts()

# %%
df_val.cylinders.value_counts()

# %%
from sklearn.feature_extraction import DictVectorizer


# %%
y_train = df_train['log_co2'].values
y_val = df_val['log_co2'].values
y_test = df_test['log_co2'].values

# %%
del df_train['log_co2']
del df_val['log_co2']
del df_test['log_co2']

# %%
train_dicts = df_train.to_dict(orient = 'records')
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(train_dicts)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# %%
X_train.shape

# %%
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# %%
linear_model.score(X_train, y_train)

# %%
y_pred_train = linear_model.predict(X_train)


# %%
r2_score(y_pred_train, y_train), mean_squared_error(y_pred_train, y_train, squared=False)

# %%
val_dicts = df_val.to_dict(orient = 'records')
dv = DictVectorizer(sparse = False)
X_val = dv.fit_transform(val_dicts)

# %%
X_val.shape

# %%
y_pred_val = linear_model.predict(X_val)

# %%
r2_score(y_pred_val, y_val), mean_squared_error(y_pred_val, y_val, squared=False)

# %% [markdown]
# ### 1.4.1 Decision Tree

# %%
from sklearn.tree import DecisionTreeRegressor

# %%
dt = DecisionTreeRegressor()

# %%
dt.fit(X_train, y_train)

# %%
y_pred_train = dt.predict(X_train)
r2_score(y_pred_train, y_train), mean_squared_error(y_pred_train, y_train, squared=False)

# %%
y_pred_val = dt.predict(X_val)
r2_score(y_pred_val, y_val), mean_squared_error(y_pred_val, y_val, squared=False)

# %%
for d in [1,2,3,4,5,6,10,15,20, None]:
	dt = DecisionTreeRegressor(max_depth=d)
	dt.fit(X_train, y_train)
	y_pred_val = dt.predict(X_val)
	r2 = r2_score(y_pred_val, y_val)
	rmse = mean_squared_error(y_pred_val, y_val, squared=False)
	print('%4s -> %4s -> %.3f' % (d, r2, rmse))



# %%
scores= []
for d in [3,4,5,6,10,15,20,25,30]:
	for s in [1,2,5,10,20,50,100,150,200, 500]:
		dt = DecisionTreeRegressor(max_depth=d, min_samples_leaf=s)
		dt.fit(X_train, y_train)
		y_pred_val = dt.predict(X_val)
		r2=r2_score(y_pred_val, y_val)
		rmse = mean_squared_error(y_pred_val, y_val , squared=False)
		print(('%4s, %3d) -> %.3f -> %.3f' % (d,s, r2, rmse )))
		scores.append((s,d, r2, rmse))



# %%
col = ['depth', 'leaf_sample', 'r2', 'rmse']
df_params = pd.DataFrame(scores, columns=col)

# %%
df_params.sort_values('rmse', ascending=True)

# %% [markdown]
# ### 1.4.3 Random Forest

# %%
from sklearn.ensemble import RandomForestRegressor


# %%
rf_scores = []
for n in range(10, 201, 10):
	rf = RandomForestRegressor(n_estimators=n, random_state=1)
	rf.fit(X_train, y_train)
	y_pred_val = rf.predict(X_val)
	r2 = r2_score(y_pred_val, y_val)
	rmse = mean_squared_error(y_pred_val, y_val, squared=False)


	rf_scores.append((n, r2, rmse))


# %%
columns = ['n_estimator', 'r2', 'rmse']
df_rf_scores = pd.DataFrame(rf_scores, columns=columns)


# %%
df_rf_scores.sort_values('rmse', ascending=False)

# %%
rf_scores = []
for d in range(10,201, 10):
	rf = RandomForestRegressor(n_estimators=10, max_depth=d, random_state=1)
	rf.fit(X_train, y_train)
	y_pred_val = rf.predict(X_val)
	r2 = r2_score(y_pred_val, y_val)
	rmse = mean_squared_error(y_pred_val, y_val, squared=False)
	rf_scores.append((d, r2, rmse))

# %%
col = ['max_depth', 'r2', 'rmse']
df_rf_scores = pd.DataFrame(rf_scores, columns=col)
df_rf_scores.sort_values('rmse', ascending=False)

# %% [markdown]
# # 2. Summary: Selecting Best Model

# %% [markdown]
# # 2.1 Linear Regression

# %%
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_val = linear_model.predict(X_val)

r2_score(y_pred_val, y_val), mean_squared_error(y_pred_val, y_val, squared=False)

# %% [markdown]
# # 2.2  Decision Tree

# %%
dt = DecisionTreeRegressor(max_depth=1, min_samples_leaf=30)
dt.fit(X_train, y_train)
y_pred_val = dt.predict(X_val)
r2 = r2_score(y_pred_val, y_val)
rmse = mean_squared_error(y_pred_val, y_val, squared=False)
print(r2, rmse)


# %% [markdown]
# ## 2.3 Random Forest

# %%
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1)
rf.fit(X_train, y_train)
y_pred_val = rf.predict(X_val)
r2 = r2_score(y_pred_val, y_val)
rmse = mean_squared_error(y_pred_val, y_val, squared=False)
print(r2, rmse)


# %% [markdown]
# # 3. Saving Best Model

# %%
import pickle

# %%
output_file = 'model.bin'


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

# %%
test_dicts = df_test.to_dict(orient = 'records')
dv = DictVectorizer(sparse = False)
X_test = dv.fit_transform(test_dicts)


# %%
test_pred = rf.predict(X_test)
r2_score(test_pred, y_test), mean_squared_error(test_pred, y_test)

# %%
df_test

# %%
np.exp(test_pred[2])

# %%
import pickle
input_file = 'model.bin'

with open(input_file, 'rb') as f_in:
    (dv, rf) = pickle.load(f_in)

# %%
test = {
    "vehicle_class": "subcompact",
    "engine_size": 1.5,
    "cylinders": 3,
    "fuel_type": "z",
    "fuel_consumption_city": 8.3,
    "fuel_consumption_hwy": 6.4,
    "fuel_consumption_comb_mpg": 38


}


# %%
x = dv.transform(test)

# %%
y_pred = rf.predict(x)
import numpy as np
np.exp(y_pred)

# %%




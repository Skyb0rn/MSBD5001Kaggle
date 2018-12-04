import numpy as np 
import pandas as pd 
import csv
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

csvTrainFeatures = "train.csv"  # csv file with the training dataset
csvTestFeatures = "test.csv"    # csv file with the testing dataset
csvPrediction = "prediction.csv" # csv file to output the predictions to

# Drop irrelevant features, and add new features that may be useful to the model. Columns that seems wont be relevant to the execution time: random_state, n_clusters_per_class, scale
def FeaturesProcessing(data):
    data = pd.get_dummies(data)

    # n_jobs = -1 means use all cores available. Number of cores in the system is at least 16 from the training data, if not more
    data['n_jobs'].replace(to_replace=-1, value=16, inplace=True) #try 16, 24, 32
    # Time will not be improved if there are more cores (n_jobs) than classes
    data['n_jobs'] = data[['n_jobs','n_classes']].min(axis=1)

    # # Feature set 1
    # data['f1_a'] = data['alpha']/data['n_jobs']
    # data['f1_b'] = data['max_iter']/data['n_jobs']
    # data['f1_c'] = data['n_samples']/data['n_jobs']
    # data['f1_d'] = data['n_features']/data['n_jobs']
    # data['f1_e'] = data['n_classes']/data['n_jobs']

    # # Feature set 2
    # data['f2_a'] = data['alpha']*data['max_iter']/data['n_jobs']
    # data['f2_b'] = data['alpha']*data['n_samples']/data['n_jobs']
    # data['f2_c'] = data['alpha']*data['n_features']/data['n_jobs']
    # data['f2_d'] = data['alpha']*data['n_classes']/data['n_jobs']

    # # Feature set 3
    # data['f3_a'] = data['alpha']*data['max_iter']*data['n_samples']/data['n_jobs']
    # data['f3_b'] = data['alpha']*data['max_iter']*data['n_features']/data['n_jobs']
    # data['f3_c'] = data['alpha']*data['max_iter']*data['n_classes']/data['n_jobs']

    # # Feature set 4
    # data['f4_a'] = data['max_iter']*data['n_samples']/data['n_jobs']
    # data['f4_b'] = data['max_iter']*data['n_features']/data['n_jobs']
    # data['f4_c'] = data['max_iter']*data['n_classes']/data['n_jobs']

    # # Feature set 5
    # data['f5_a'] = data['alpha']*data['max_iter']*data['n_samples']*data['n_features']/data['n_jobs']
    # data['f5_b'] = data['alpha']*data['max_iter']*data['n_samples']*data['n_classes']/data['n_jobs']

    # # Feature set 6
    # data['f6_a'] = data['max_iter']*data['n_samples']*data['n_features']/data['n_jobs']
    # data['f6_b'] = data['max_iter']*data['n_samples']*data['n_classes']/data['n_jobs']

    # # Feature set 7
    data['f7_a'] = data['max_iter']*data['n_samples']*data['n_features']*data['n_classes']/data['n_jobs']
    
    # # Feature set 8
    # data['f8_a'] = data['alpha']*data['max_iter']*data['n_samples']*data['n_features']*data['n_classes']/data['n_jobs']


    # # Features to drop
    # data.drop(columns=['random_state','n_clusters_per_class','flip_y','scale','alpha','max_iter','n_samples','n_features','n_classes','n_jobs'], inplace=True)

    # data.drop(columns=['random_state','n_clusters_per_class','n_informative','scale','alpha','max_iter','n_samples','n_features','n_classes','n_jobs'], inplace=True)

    data.drop(columns=['random_state','n_clusters_per_class','scale','alpha','max_iter','n_samples','n_features','n_classes','n_jobs'], inplace=True)

    # data.drop(columns=['random_state','n_clusters_per_class','scale','max_iter','n_samples','n_features','n_classes','n_jobs'], inplace=True)

    # data.drop(columns=['random_state','scale','max_iter','n_samples','n_features','n_classes','n_jobs'], inplace=True)

    return data


raw_data = pd.read_csv(csvTrainFeatures)
raw_data = FeaturesProcessing(raw_data)

# y = raw_data['time'].values
# X = raw_data.drop(columns='time').values

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# model = LinearRegression()
# model.fit(X,y)
# print(cross_val_score(model,X,y,cv=10))

# Split X and y into 4 separate groups, 1 for each penalty type
data = raw_data[raw_data['penalty_elasticnet']>0]
X_elasticnet = data.drop(columns=['id','penalty_elasticnet','penalty_l1','penalty_l2','penalty_none','time']).values
y_elasticnet = data['time'].values

data = raw_data[raw_data['penalty_l1']>0]
X_l1 = data.drop(columns=['id','penalty_elasticnet','penalty_l1','penalty_l2','penalty_none','time']).values
y_l1 = data['time'].values

data = raw_data[raw_data['penalty_l2']>0]
X_l2 = data.drop(columns=['id','penalty_elasticnet','penalty_l1','penalty_l2','penalty_none','time']).values
y_l2 = data['time'].values

data = raw_data[raw_data['penalty_none']>0]
X_none = data.drop(columns=['id','penalty_elasticnet','penalty_l1','penalty_l2','penalty_none','time']).values
y_none = data['time'].values



# OUTPUT FILE
test_data = pd.read_csv(csvTestFeatures)
test_data = FeaturesProcessing(test_data)

# X_test_data = test_data.values
# X_test_data = scaler.transform(X_test_data)
# predictions = model.predict(X_test_data)

# Split X_test_data into the same 4 penalty groups, keeping track of the id at the same time for merging back later
data = test_data[test_data['penalty_elasticnet']>0]
X_elasticnet_test = data.drop(columns=['id','penalty_elasticnet','penalty_l1','penalty_l2','penalty_none']).values
X_elasticnet_testID = data['id'].values

data = test_data[test_data['penalty_l1']>0]
X_l1_test = data.drop(columns=['id','penalty_elasticnet','penalty_l1','penalty_l2','penalty_none']).values
X_l1_testID = data['id'].values

data = test_data[test_data['penalty_l2']>0]
X_l2_test = data.drop(columns=['id','penalty_elasticnet','penalty_l1','penalty_l2','penalty_none']).values
X_l2_testID = data['id'].values

data = test_data[test_data['penalty_none']>0]
X_none_test = data.drop(columns=['id','penalty_elasticnet','penalty_l1','penalty_l2','penalty_none']).values
X_none_testID = data['id'].values

predictions = np.zeros(test_data.shape[0])

# model = DecisionTreeRegressor()
model = LinearRegression()
# model = RandomForestRegressor()

model.fit(X_elasticnet,y_elasticnet)
results = model.predict(X_elasticnet_test)
for index,result in enumerate(results):
    predictions[X_elasticnet_testID[index]] = result

model.fit(X_l1,y_l1)
results = model.predict(X_l1_test)
for index,result in enumerate(results):
    predictions[X_l1_testID[index]] = result

model.fit(X_l2,y_l2)
results = model.predict(X_l2_test)
for index,result in enumerate(results):
    predictions[X_l2_testID[index]] = result

model.fit(X_none,y_none)
results = model.predict(X_none_test)
for index,result in enumerate(results):
    predictions[X_none_testID[index]] = result

pd.DataFrame({"time":predictions}).to_csv(csvPrediction,index_label="id")


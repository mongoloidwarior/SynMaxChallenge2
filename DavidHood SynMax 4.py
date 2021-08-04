# -*- coding: utf-8 -*-
"""
David Hood 21Jul2021

SynMax Programming Contest
https://synmax.com/home/synmax-twitter-contest/
"""

import pandas as pd

#import ports and tracking data
ports = pd.read_csv(r'C:\Users\hoodd02\SynMax\ports.csv')
tracking = pd.read_csv(r'C:\Users\hoodd02\SynMax\tracking.csv')

"""
Create a voyage table which specifies for each vessel and each unique voyage
a starting port, starting date, ending port and ending date. Voyage starting 
and ending dates should generally begin at the time when the vessel leaves port
(starts moving) and end when the vessel arrives at the next port 
(stops moving.) This is sometimes ambiguous so some leeway will be given in 
grading these dates. However, origin and destination port determinations will 
be strict.
"""

import geopandas as gpd
import numpy as np

from scipy.spatial import cKDTree
from shapely.geometry import Point

#will need date later
import datetime
tracking['datetime'] = pd.to_datetime(tracking['datetime'])
tracking['date'] = tracking['datetime'].dt.date

#tracking.hist(column='speed', bins=25)


#convert to geo data frame 
gports = gpd.GeoDataFrame(
        ports, geometry=gpd.points_from_xy(ports.lat, ports.long)).drop(['lat','long'],axis=1)

gtracking = gpd.GeoDataFrame(
        tracking, geometry=gpd.points_from_xy(tracking.lat, tracking.long)).drop(['lat','long'],axis=1)


def ckdnearest(gdA, gdB):

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

voyage = ckdnearest(gtracking, gports)
voyage = voyage.sort_values(by=["vessel","datetime"])
voyage = voyage.drop_duplicates()

"""
testing filtering on speed
"""

#keep only rows with speed=0 or >10
voyage = voyage[(voyage['speed']==0) | (voyage['speed'] >=10)]


"""
end testing
"""


#use speed=0 as start and end points

#ended up not needing/using
#mydata = voyage.groupby(['vessel'], as_index=False)['date'].min().copy()
#mydata.rename(columns = {'date':'first_date'}, inplace=True)
#speed0 = voyage.loc[voyage['speed']==0]
#mydata2 = speed0.groupby(['vessel'], as_index=False)['date'].min()
#mydata2.rename(columns = {'date':'first0_date'}, inplace=True)
#matching = pd.concat([mydata, mydata2], axis=1, join="inner")
#matching = matching.loc[:,~matching.columns.duplicated()]
#matching
#
#pd.merge(voyage,matching)



"""
UNCOMMENT TO TEST FASTER
"""
#testing with vessel 1 for speed
#voyage = voyage.loc[voyage['vessel']==1]
"""
UNCOMMENT TO TEST FASTER
"""


#stop and start columns based on speed changing to or from 0
#takes too long, need to break into pieces
regroup = pd.DataFrame()

for x in (voyage.vessel.unique()):
#for x in range (1,2+1): #takes 4 seconds per ship
    subset = voyage.loc[voyage['vessel']==x].copy()
    for i in range (1, subset.shape[0]):
        subset.loc[subset.index[i],'stop'] = subset.iloc[i-1,3]>0 and subset.iloc[i,3]==0
        subset.loc[subset.index[i],'start'] = subset.iloc[i-1,3]==0 and subset.iloc[i,3]>0
    regroup = pd.concat([regroup,subset])

voyage = regroup.copy()


#keep only starts and stops
voyage_ss = voyage[(voyage['start']==1) | (voyage['stop']==1)]
voyage_ss = voyage_ss[['vessel','date','port','speed','start','stop']]

#assign begin and end port/date
for i in range (0, voyage_ss.shape[0]-1):
    if voyage_ss.loc[voyage_ss.index[i],'start'] == 1 & voyage_ss.loc[voyage_ss.index[i+1],'stop'] == 1:
        voyage_ss.loc[voyage_ss.index[i],'begin_date'] = voyage_ss.loc[voyage_ss.index[i],'date']
        voyage_ss.loc[voyage_ss.index[i],'begin_port_id'] = voyage_ss.loc[voyage_ss.index[i],'port']
        voyage_ss.loc[voyage_ss.index[i],'end_date'] = voyage_ss.loc[voyage_ss.index[i+1],'date']
        voyage_ss.loc[voyage_ss.index[i],'end_port_id'] = voyage_ss.loc[voyage_ss.index[i+1],'port']

#keep only full records
voyage_ss = voyage_ss.dropna()
voyage_ss = voyage_ss.drop(['date','port','start','stop','speed'],axis=1)

#final voyages dataset with begin and end    
#need to remove rows where both ports are the same (maneuvering withing harbor?)
voyages = voyage_ss.loc[voyage_ss['begin_port_id'] != voyage_ss['end_port_id']].copy()

#some data we first see the ship moving they are already closer to another port
#set begin port to previous end port
for i in range (1, voyages.shape[0]):
    if voyages.loc[voyages.index[i-1],'vessel'] == voyages.loc[voyages.index[i],'vessel']:
        voyages.loc[voyages.index[i],'begin_port_id'] = voyages.loc[voyages.index[i-1],'end_port_id']
    
    
#issue where ships show last voyage using next ships data, delete if startdt<enddt
voyages = voyages[voyages.begin_date < voyages.end_date]    
    
    
voyages.to_csv(r'C:\Users\hoodd02\SynMax\voyages.csv', index=False)



#pd.set_option('max_columns',None)
#print(voyage.loc[voyage['vessel'] ==1])
#pd.reset_option('max_columns')
#voyage.to_csv(r'C:\Users\hoodd02\SynMax\test.csv', index=False)

"""

machine learning section

predict the next port using last (2?) port as inputs to a decision tree model
ref: https://www.analyticsvidhya.com/blog/2020/10/all-about-decision-tree-from-scratch-with-python-implementation/
ref: https://www.datacamp.com/community/tutorials/decision-tree-classification-python

"""


### add seasonal variables
### adapted from: https://randyperez.tech/blog/seasons-column
def get_season(date_time):
    # dummy leap year to include leap days(year-02-29) in our range
    leap_year = 2000
    seasons = [('winter', (datetime.date(leap_year, 1, 1), datetime.date(leap_year, 3, 20))),
               ('spring', (datetime.date(leap_year, 3, 21), datetime.date(leap_year, 6, 20))),
               ('summer', (datetime.date(leap_year, 6, 21), datetime.date(leap_year, 9, 22))),
               ('autumn', (datetime.date(leap_year, 9, 23), datetime.date(leap_year, 12, 20))),
               ('winter', (datetime.date(leap_year, 12, 21), datetime.date(leap_year, 12, 31)))]

    if isinstance(date_time, datetime.datetime):
        date_time = date_time.date()
    # we don't really care about the actual year so replace it with our dummy leap_year
    date_time = date_time.replace(year=leap_year)
    # return season our date falls in.
    return next(season for season, (start, end) in seasons
                if start <= date_time <= end)


def create_season_column(data_set, date_column):
    # cloning the input dataset.
    local = data_set.copy()
    # The apply method calls a function on each row
    local['Season'] = local[date_column].apply(get_season)
    return local

voyages = create_season_column(voyages, date_column='end_date')

#convert to dummy variables
voyages['dwinter'] = np.where(voyages['Season'] == 'winter', 1, 0)
voyages['dspring'] = np.where(voyages['Season'] == 'spring', 1, 0)
voyages['dsummer'] = np.where(voyages['Season'] == 'summer', 1, 0)
voyages['dautumn'] = np.where(voyages['Season'] == 'autumn', 1, 0)


#add back last heading for each ship
last_heading = voyage.groupby(['vessel','date']).head(1)[['vessel','date','heading']]
last_heading.columns = ['vessel','begin_date','heading']

df = pd.merge(voyages, last_heading, how='left', on=['vessel','begin_date'])

df['heading_n'] =  df['heading'].between(0-22.5,0+22.5, inclusive=True)
df['heading_ne'] = df['heading'].between(45-22.5,45+22.5, inclusive=True)
df['heading_e'] =  df['heading'].between(90-22.5,90+22.5, inclusive=True)
df['heading_se'] = df['heading'].between(135-22.5,135+22.5, inclusive=True)
df['heading_s'] =  df['heading'].between(180-22.5,180+22.5, inclusive=True)
df['heading_sw'] = df['heading'].between(225-22.5,225+22.5, inclusive=True)
df['heading_w'] =  df['heading'].between(270-22.5,270+22.5, inclusive=True)
df['heading_nw'] = df['heading'].between(315-22.5,315+22.5, inclusive=True)

df = df.drop('heading', axis=1)

#drop dates and add previous port variables
df = df.sort_values(by=["vessel","begin_date"])
df = df.drop(['end_date','begin_date','Season'],axis=1)


### might add .02 accuracy... would have to add data steps between pred 1,2,3
for i in range (0, df.shape[0]):
    if df.loc[df.index[i],'vessel'] == df.loc[df.index[i-1],'vessel']:
        df.loc[df.index[i],'port_pre1'] = df.loc[df.index[i-1],'end_port_id']
    else: df.loc[df.index[i],'port_pre1'] = 0
    if df.loc[df.index[i],'vessel'] == df.loc[df.index[i-2],'vessel']:
        df.loc[df.index[i],'port_pre2'] = df.loc[df.index[i-2],'end_port_id']
    else: df.loc[df.index[i],'port_pre2'] = 0
    if df.loc[df.index[i],'vessel'] == df.loc[df.index[i-3],'vessel']:
        df.loc[df.index[i],'port_pre3'] = df.loc[df.index[i-3],'end_port_id']
    else: df.loc[df.index[i],'port_pre3'] = 0
    if df.loc[df.index[i],'vessel'] == df.loc[df.index[i-4],'vessel']:
        df.loc[df.index[i],'port_pre4'] = df.loc[df.index[i-4],'end_port_id']
    else: df.loc[df.index[i],'port_pre4'] = 0
    if df.loc[df.index[i],'vessel'] == df.loc[df.index[i-5],'vessel']:
        df.loc[df.index[i],'port_pre5'] = df.loc[df.index[i-5],'end_port_id']
    else: df.loc[df.index[i],'port_pre5'] = 0



p1 = df.copy()


"""
adding model that includes heading to predict first port as data appears
to end mid route for most ships

"""
"""
testing removing infrequently used ports
i.e. ports in data where avg uses as start and end <=8
increased accuracy from 30% to 43%

5-8 starts at 11
"""

infrequent_ports = [6,	15,	16,	18,	24,	31,	33,	39,	40,	42,	44,	45,	46,	53,	55,	
                    56,	59,	60,	62,	63,	71,	74,	75,	79,	80,	82,	87,	93,	95,	98,	
                    108,	109,	112,	113,	122,	123,	126,	136,	
                    140,	141,	142,	143,	144,	146,	150,	153,	
                    157,	159,	162,	165,	168,	170,	174,	177,	
                    178, 11,	20,	27,	30,	35,	36,	37,	38,	48,	57,	58,	64,	67,	
                    76,	84,	88,	96,	102,	117,	121,	130,	139,	148,
                    175]

p1 = p1[~p1['end_port_id'].isin(infrequent_ports)]
p1 = p1[~p1['begin_port_id'].isin(infrequent_ports)]

"""
end testing section
"""
#ref https://towardsdatascience.com/dealing-with-categorical-data-fast-an-example-d4329b44253d

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', None)  # Unlimited columns.
pd.options.mode.use_inf_as_na = True        # Any inf or -inf is 
                                            # treated as NA.


target = p1['end_port_id'].astype(str)
df1 = p1.copy()
df1 = df1.drop('end_port_id', axis=1)

X = df1
y = target

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)
                                            
X_train_numerical = X_train.select_dtypes(
                         include = np.number).copy()
X_train_numerical.head()

X_train_numerical_indices = X_train_numerical.index.values
y_train_numerical = y_train[y_train.index.isin(X_train_numerical_indices)]

#clf = DecisionTreeClassifier()
#cv_score = cross_val_score(clf, 
#                            X_train_numerical, y_train_numerical,
#                            scoring = 'accuracy',
#                            cv = 3,
#                            n_jobs = -1,
#                            verbose = 1)
#cv_score

#clf.fit(X_train_numerical, y_train_numerical)

X_test_numerical = X_test.select_dtypes(include = np.number).copy()

#y_pred = clf.predict(X_test_numerical)

X_non_nulls = X_train.dropna(axis = 1)

X_non_nulls.nunique().sort_values(ascending = True)

X_selected = X_non_nulls.loc[:, X_non_nulls.nunique().sort_values() < 200]
cat_cols = list(X_selected.select_dtypes(['float64','int64']).columns.values)
X_categorical = X_selected[cat_cols].apply(lambda x: x.astype('category').cat.codes)
#X_train_selected = X_train_numerical.merge(X_categorical, on='vessel', how='left')
X_train_selected = pd.concat([X_train_numerical,X_categorical],axis=1)

#param_grid = {
#    'n_estimators': [10, 20, 30],
#    'max_depth': [6, 10, 20, 30]
#}
#gridsearch = GridSearchCV(RandomForestClassifier(n_jobs = -1), 
#                          param_grid=param_grid, 
#                          scoring='accuracy', cv=3, 
#                          return_train_score=True, verbose=10)
#gridsearch.fit(X_train, y_train)
#
#pd.DataFrame(gridsearch.cv_results_).sort_values(by='rank_test_score')


clf = RandomForestClassifier(max_depth = 10, 
                             n_estimators = 30, 
                             n_jobs = -1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)




"""
predict and add predicted ports
"""
columns = ['vessel','begin_port_id','dwinter','dspring','dsummer','dautumn'
           ,'heading_n','heading_ne','heading_e','heading_se','heading_s'
           ,'heading_sw','heading_w','heading_nw'
           ,'port_pre1','port_pre2','port_pre3','port_pre4','port_pre5']

pred1 = p1.groupby('vessel').tail(1).drop('begin_port_id', axis=1)
pred1.columns = columns
cols = pred1.columns.tolist()
pred1['end_port_id'] = clf.predict(pred1)
pred1['voyage'] = 1
#pred1

columns2 = ['vessel','port_pre1','dwinter','dspring','dsummer','dautumn'
           ,'heading_n','heading_ne','heading_e','heading_se','heading_s'
           ,'heading_sw','heading_w','heading_nw'
           ,'port_pre2','port_pre3','port_pre4','port_pre5','begin_port_id']

pred2 = pred1.groupby('vessel').tail(1).drop(['port_pre5','voyage'], axis=1)
pred2.columns = columns2
pred2 = pred2[cols]
pred2['end_port_id'] = clf.predict(pred2)
pred2['voyage'] = 2
#pred2


pred3 = pred2.groupby('vessel').tail(1).drop(['port_pre5','voyage'], axis=1)
pred3.columns = columns2
pred3 = pred3[cols]
pred3['end_port_id'] = clf.predict(pred3)
pred3['voyage'] = 3
#pred3


#final dataset
predict = pd.concat([pred1,pred2,pred3])
predict = predict.drop(['dwinter','dspring','dsummer','dautumn'
                        ,'port_pre1','port_pre2','port_pre3','port_pre4','port_pre5'
                        ,'heading_n','heading_ne','heading_e','heading_se','heading_s'
                        ,'heading_sw','heading_w','heading_nw'
                        ],axis=1)

predict.to_csv(r'C:\Users\hoodd02\SynMax\predict.csv', index=False)

#save model
import pickle
pickle.dump(clf, open(r'C:\Users\hoodd02\SynMax\dtree.sav', 'wb'))


pd.set_option('max_columns',None)
#print(voyage.loc[voyage['SMA_3'] < .1])
print(pred2.head())
pd.reset_option('max_columns')



# -*- coding: utf-8 -*-
"""
David Hood 21Jul2021

SynMax Programming Contest
https://synmax.com/home/synmax-twitter-contest/
"""

import pandas as pd

#import ports and tracking data
ports = pd.read_csv(r'C:\Users\hoodd02\downloads\ports.csv')
tracking = pd.read_csv(r'C:\Users\hoodd02\downloads\tracking.csv')

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


"""
OG code (delete if the above works):
        for i in range (1, voyage.shape[0]):
        voyage.loc[voyage.index[i],'stop'] = voyage.iloc[i-1,3]>0 and voyage.iloc[i,3]==0
        voyage.loc[voyage.index[i],'start'] = voyage.iloc[i-1,3]==0 and voyage.iloc[i,3]>0

"""


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
voyage_ss = voyage_ss.drop(['date','port','start','stop'],axis=1)

#final voyages dataset with begin and end    
#need to remove rows where both ports are the same (maneuvering withing harbor?)
voyages = voyage_ss.loc[voyage_ss['begin_port_id'] != voyage_ss['end_port_id']]
    
voyages.to_csv(r'C:\Users\hoodd02\SynMax\voyages.csv', index=False)



"""

machine learning section

predict the next port using last (2?) port as inputs to a decision tree model
ref: https://www.analyticsvidhya.com/blog/2020/10/all-about-decision-tree-from-scratch-with-python-implementation/
ref: https://www.datacamp.com/community/tutorials/decision-tree-classification-python

"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder #for train test splitting
from sklearn.model_selection import train_test_split #for decision tree object
from sklearn.tree import DecisionTreeClassifier #for checking testing results
from sklearn.metrics import classification_report, confusion_matrix #for visualizing tree 
#from sklearn.tree import plot_tree


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




#drop dates and add previous port variables
df = voyages.drop(['end_date','begin_date','speed','Season'],axis=1)


### only adds .006 accuracy... not worth it
#for i in range (2, df.shape[0]):
#    if df.loc[df.index[i],'vessel'] == df.loc[df.index[i-1],'vessel']:
#        df.loc[df.index[i],'port_pre1'] = df.loc[df.index[i-1],'end_port_id']
#    if df.loc[df.index[i],'vessel'] == df.loc[df.index[i-2],'vessel']:
#        df.loc[df.index[i],'port_pre2'] = df.loc[df.index[i-2],'end_port_id']

df = df.dropna()

        
target = df['end_port_id']
df1 = df.copy()
df1 = df1.drop('end_port_id', axis=1)
X = df1

le = LabelEncoder()
target = le.fit_transform(target)
y = target

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)

dtree=DecisionTreeClassifier(criterion="entropy", max_depth=10)
dtree.fit(X_train,y_train)
print('Decision Tree Classifier Created')

# Predicting the values of test data
y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#df = df.drop(['dspring','dsummer','dautumn','dwinter'],axis=1)

"""
predict and add predicted ports
"""

pred1 = df.groupby('vessel').tail(1).drop('begin_port_id', axis=1)
pred1.columns = ['vessel','begin_port_id','dwinter','dspring','dsummer','dautumn']
pred1['end_port_id'] = dtree.predict(pred1)
pred1['voyage'] = 1
#pred1


pred2 = pred1.groupby('vessel').tail(1).drop(['begin_port_id','voyage'], axis=1)
pred2.columns = ['vessel','begin_port_id','dwinter','dspring','dsummer','dautumn']
pred2['end_port_id'] = dtree.predict(pred2)
pred2['voyage'] = 2
#pred2


pred3 = pred2.groupby('vessel').tail(1).drop(['begin_port_id','voyage'], axis=1)
pred3.columns = ['vessel','begin_port_id','dwinter','dspring','dsummer','dautumn']
pred3['end_port_id'] = dtree.predict(pred3)
pred3['voyage'] = 3
#pred3


#final dataset
predict = pd.concat([pred1,pred2,pred3])
predict = predict.drop(['dwinter','dspring','dsummer','dautumn'],axis=1)

predict.to_csv(r'C:\Users\hoodd02\SynMax\predict.csv', index=False)

#save model
import pickle
pickle.dump(dtree, open(r'C:\Users\hoodd02\SynMax\dtree.sav', 'wb'))


#pd.set_option('max_columns',None)
#print(voyage.loc[voyage['SMA_3'] < .1])
#print(voyage_ss.head())
#pd.reset_option('max_columns')



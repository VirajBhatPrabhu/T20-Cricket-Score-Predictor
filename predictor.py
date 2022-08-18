# #### Import the libraries first

import pandas as pd
import numpy as np
import pickle




pd.set_option('display.max_columns', None)
df = pickle.load(open('rawcsv.pkl', 'rb'))


df.columns.get_loc('info.teams')

# #### Lets remove the redundant columns



df.drop(df.columns[48:], axis=1, inplace=True)


df.drop(df.columns[11:40], axis=1, inplace=True)

# #### Lets check how far back the data goes



df['year'] = pd.to_datetime(df['meta.created']).dt.year



df['year'].value_counts()


print(df['info.match_type'].value_counts())
print(df['info.gender'].value_counts())
print(df['info.overs'].value_counts())

# ##### The Game has changed a lot since it was first played in 2003.Taking that into account for current prediction i wanted to use only the data since 2015 but because of ack of data i have decided to use all the data since 2010

# ###### Also we will only make this model fo the mens T20 internationals. so lets remove the 50 over data and the matches for gender='male



df1 = df[(df['year'] >= 2010) & (df['info.gender'] == 'male') & (df['info.overs'] == 20)]


# ##### We will also remove the redundant columns



cols_to_drop = ['info.registry.people.WU Tharanga', 'meta.data_version', 'meta.created', 'meta.revision',
                'info.balls_per_over', 'info.dates', 'year', 'info.gender', 'info.match_type',
                'info.outcome.by.wickets', 'info.overs']
df1.drop(cols_to_drop, axis=1, inplace=True)


# #### lets first make a dataframe for the first innings data

count = 1
delivery_df = pd.DataFrame()
for index, row in df1.iterrows():
    if count in [75, 108, 150, 180, 268, 360, 443, 458, 584, 748, 982, 1052, 1111, 1226, 1345, 1450, 1680, 1900, 210,
                 2300, 2500, 2800, 3111, 3500, 3900, 4200, 4500, 5000, 6000]:
        count += 1
        continue
    count += 1
    ball_of_match = []
    batsman = []
    bowler = []
    runs = []
    player_of_dismissed = []
    teams = []
    batting_team = []
    match_id = []
    city = []
    venue = []
    toss_winner = []
    toss_decision = []
    for ball in row['innings'][0]['1st innings']['deliveries']:
        for key in ball.keys():
            match_id.append(count)
            batting_team.append(row['innings'][0]['1st innings']['team'])
            teams.append(row['info.teams'])
            ball_of_match.append(key)
            batsman.append(ball[key]['batsman'])
            bowler.append(ball[key]['bowler'])
            runs.append(ball[key]['runs']['total'])
            city.append(row['info.city'])
            venue.append(row['info.venue'])
            toss_winner.append(row['info.toss.winner'])
            toss_decision.append(row['info.toss.decision'])

            try:
                player_of_dismissed.append(ball[key]['wicket']['player_out'])
            except:
                player_of_dismissed.append('0')
    loop_df = pd.DataFrame({'match_id': match_id, 'teams': teams, 'batting_team': batting_team, 'ball': ball_of_match,
                            'batsman': batsman, 'bowler': bowler, 'runs': runs, 'player_dismissed': player_of_dismissed,
                            'city': city, 'venue': venue, 'toss_winner': toss_winner, 'toss_decision': toss_decision})
    delivery_df = delivery_df.append(loop_df)

# ##### We have the batiing team lets get the bowling team from the data and also do some other preprocessing

#
def bowling(row):
    for team in row['teams']:
        if team != row['batting_team']:
            return team


delivery_df['bowling_team'] = delivery_df.apply(bowling, axis=1)


delivery_df.drop(['teams', 'city', 'batsman', 'bowler'], axis=1, inplace=True)

# ##### We are only making this predictor for the top10 cricket playing nations



teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan',
         'Pakistan', 'Sri Lanka']



delivery_df = delivery_df[delivery_df['batting_team'].isin(teams)]
delivery_df = delivery_df[delivery_df['bowling_team'].isin(teams)]



# #### Lets start getting additional data out of our present data

# ##### Things we need to get are -- current score, CRR(current run rate), wickets remaining,  balls left, total runs ,score in last 5 overs


total_runs = delivery_df.groupby('match_id')['runs'].sum().reset_index()
total_runs.rename(columns={'runs': 'total_runs'})


output = delivery_df.merge(total_runs, on='match_id')


output['current_score'] = output.groupby('match_id')['runs_x'].cumsum()


output['player_dismissed'] = output['player_dismissed'].apply(lambda x: 0 if x == '0' else 1)
output['players_dismissed'] = output.groupby('match_id')['player_dismissed'].cumsum()


output['wickets_remaining'] = 10 - output['players_dismissed']


output['over_no'] = output['ball'].apply(lambda x: str(x).split('.')[0])
output['ball_no'] = output['ball'].apply(lambda x: str(x).split('.')[1])


output['balls_bowled'] = (output['over_no'].astype('int') * 6) + (output['ball_no'].astype('int'))
output['crr'] = round((output['current_score'] * 6) / output['balls_bowled'], 2)


output['balls_left'] = 126 - output['balls_bowled']
output['balls_left'] = output['balls_left'].apply(lambda x: 0 if x < 0 else x)


matches = output.groupby('match_id')
scores_lastfive = []
for i in output['match_id'].unique():
    scores_lastfive.extend(matches.get_group(i).rolling(window=30).sum()['runs_x'].values.tolist())
output['scores_lastfive'] = scores_lastfive


output.dropna(inplace=True)

output['scores_lastfive'] = output['scores_lastfive'].astype('int')

output.drop(
    ['runs_x', 'player_dismissed', 'players_dismissed', 'ball_no', 'over_no', 'balls_bowled', 'match_id', 'ball',
     'match_id'], axis=1, inplace=True)


output.rename(columns={'runs_y': 'total_runs'}, inplace=True)


final_df = output

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ##### Now its time to separate our data into dependent and independent variables and split the data into train and test


X = final_df.drop('total_runs', axis=1)
y = final_df['total_runs']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse=False, drop='first'),
     ['batting_team', 'bowling_team', 'venue', 'toss_decision', 'toss_winner'])
]
    , remainder='passthrough')

##lets build the pipeline
model = Pipeline(steps=[
    ('step1', trf),
    ('step2', StandardScaler()),
    ('step3', XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=12, random_state=1))
])


## lets fit the model

model.fit(X_train, y_train)

##prediction time

y_pred = model.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))

##Lets pickle the model
pickle.dump(model, open('model.pkl', 'wb'))


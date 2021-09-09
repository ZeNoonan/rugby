import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import os
import base64 
import altair as alt
# from st_aggrid import AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

st.set_page_config(layout="wide")
url='C:/Users/Darragh/Documents/Python/rugby/rugby_data.xlsx'
data=pd.read_excel(url,sheet_name='data')
# st.write(data)

team_names_id=pd.read_excel(url,sheet_name='ID')

fb_ref_2020=pd.merge(data,team_names_id,on='Home Team').rename(columns={'ID':'Home ID'})
st.write(fb_ref_2020)
team_names_id_2=team_names_id.rename(columns={'Home Team':'Away Team'})
data=pd.merge(fb_ref_2020,team_names_id_2,on='Away Team').rename(columns={'ID':'Away ID'})
st.write(data)


def spread_workings(data):
    data['home_win']=data['Home Points'] - data['Away Points']
    data['home_win'] = np.where((data['Home Points'] > data['Away Points']), 1, np.where((data['Home Points'] < data['Away Points']),-1,0))
    data['home_cover']=(np.where(((data['Home Points'] + data['Spread']) > data['Away Points']), 1,
    np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))
    data['home_cover']=data['home_cover'].astype(int)
    data['away_cover'] = -data['home_cover']
    # data=data.rename(columns={'Net Turnover':'home_turnover'})
    # data['away_turnover'] = -data['home_turnover']
    return data

def turnover_workings(data,week_start):
    turnover_df=data[data['Week']>week_start].copy()
    turnover_df['home_turned_over_sign'] = np.where((turnover_df['Turnover'] > 0), 1, np.where((turnover_df['Turnover'] < 0),-1,0))
    turnover_df['away_turned_over_sign'] = - turnover_df['home_turned_over_sign']
    # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_turnover_df = (turnover_df.loc[:,['Week','Date','Home ID','home_turned_over_sign']]).rename(columns={'Home ID':'ID','home_turned_over_sign':'turned_over_sign'})
    # st.write('checking home turnover section', home_turnover_df[home_turnover_df['ID']==0])
    away_turnover_df = (turnover_df.loc[:,['Week','Date','Away ID','away_turned_over_sign']]).rename(columns={'Away ID':'ID','away_turned_over_sign':'turned_over_sign'})
    # st.write('checking away turnover section', away_turnover_df[away_turnover_df['ID']==0])
    season_cover=pd.concat([home_turnover_df,away_turnover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=['True','True','True'])

def turnover_2(season_cover_df):    
    # https://stackoverflow.com/questions/53335567/use-pandas-shift-within-a-group
    season_cover_df['prev_turnover']=season_cover_df.groupby('ID')['turned_over_sign'].shift()
    return season_cover_df.sort_values(by=['ID','Week'],ascending=True)
    # return season_cover_df

def season_cover_3(data,column_sign,name):
    data[column_sign] = np.where((data[name] > 0), 1, np.where((data[name] < 0),-1,0))
    return data

with st.beta_expander('Last Game Turnover'):
    turnover=spread_workings(data)
    # st.write(turnover)
    # st.write('lets have a look at the data',data[(data['Home Team']=='Arizona Cardinals') | (data['Away Team']=='Arizona Cardinals')].sort_values(by=['Week','Date','Time']))
    turnover_1 = turnover_workings(turnover,-1)
    # st.write('turnover 1', turnover_1[turnover_1['ID']==0])
    
    turnover_2=turnover_2(turnover_1)
    # st.write('turnover 2 NEXT CHECK', turnover_2[turnover_2['ID']==0])
    turnover_3=season_cover_3(turnover_2,'turnover_sign','prev_turnover')
    # st.write('this is last game turnover')
    st.write(turnover_3.sort_values(by=['ID','Week'],ascending=['True','True']))

def season_cover_workings(data,home,away,name,week_start):
    season_cover_df=data[data['Week']>week_start].copy()
    # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_cover_df = (season_cover_df.loc[:,['Week','Date','Home ID',home]]).rename(columns={'Home ID':'ID',home:name})
    # st.write('checking home turnover section', home_cover_df[home_cover_df['ID']==0])
    away_cover_df = (season_cover_df.loc[:,['Week','Date','Away ID',away]]).rename(columns={'Away ID':'ID',away:name})
    # st.write('checking away turnover section', away_cover_df[away_cover_df['ID']==0])
    season_cover=pd.concat([home_cover_df,away_cover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=['True','True','True'])

def season_cover_2(season_cover_df,column_name):    
    # https://stackoverflow.com/questions/54993050/pandas-groupby-shift-and-cumulative-sum
    # season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].transform(lambda x: x.cumsum().shift())
    # THE ABOVE DIDN'T WORK IN 2020 PRO FOOTBALL BUT DID WORK IN 2019 DO NOT DELETE FOR INFO PURPOSES
    season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].apply(lambda x: x.cumsum().shift())
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop('index',axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers
    return season_cover_df

spread=spread_workings(data)

with st.beta_expander('Season to date Cover'):
    # st.write('this is spread #0', spread)
    spread_1 = season_cover_workings(spread,'home_cover','away_cover','cover',0)

    # st.write ('this is spread showing the actual cover in the week',spread_1[spread_1['ID']==31])
    # test_1 = spread_1.reset_index().drop('index',axis=1)
    # st.write(spread_1)
    # test_1['season cover test'] = test_1.groupby ('ID')['cover'].apply(lambda x: x.cumsum().shift())
    # st.write( test_1.groupby ('ID')['cover'].transform(lambda x: x.cumsum().shift()) )
    # st.write(test_1[test_1['ID']==0])
    # st.write(test_1[test_1['ID']==31])
    # st.write(test_1[test_1['ID']==17])

    spread_2=season_cover_2(spread_1,'cover')
    # st.write('this cumsum cover to date and shifted')
    # st.write(spread_2[spread_2['ID']==31])
    spread_3=season_cover_3(spread_2,'cover_sign','cover')
    # st.write('this is season to date cover')
    st.write(spread_3.sort_values(by=['ID','Week'],ascending=['True','True']))


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
url='C:/Users/Darragh/Documents/Python/rugby/rugby_test.xlsx'
data=pd.read_excel(url,sheet_name='data')
# st.write(data)

team_names_id=pd.read_excel(url,sheet_name='ID',converters={'Date':pd.to_datetime})

fb_ref_2020=pd.merge(data,team_names_id,on='Home Team').rename(columns={'ID':'Home ID'})
# st.write(fb_ref_2020)
team_names_id_2=team_names_id.rename(columns={'Home Team':'Away Team'})
data=pd.merge(fb_ref_2020,team_names_id_2,on='Away Team').rename(columns={'ID':'Away ID'})
cols_to_move=['Week','Date','Home ID','Home Team','Away ID','Away Team','Spread']
cols = cols_to_move + [col for col in data if col not in cols_to_move]
data=data[cols]


# st.write(data.sort_values(by=['Week'], ascending=[True]))
st.write('The Power Ranking in spreadsheet is incorrect to do with the home 3 points being deducted rather than added on')

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
    st.write('Testing season cover', season_cover.sort_values(by=['Week','ID'],ascending=['True','True']))
    return season_cover.sort_values(by=['Week','ID'],ascending=['True','True'])

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

matrix_df=spread_workings(data)
matrix_df=matrix_df.reset_index().rename(columns={'index':'unique_match_id'})
test_df = matrix_df.copy()
# st.write('check for unique match id', test_df)
matrix_df['at_home'] = 1
matrix_df['at_away'] = -1
matrix_df['home_pts_adv'] = -3
matrix_df['away_pts_adv'] = 3
matrix_df['away_spread']=-matrix_df['Spread']
matrix_df=matrix_df.rename(columns={'Spread':'home_spread'})
# st.write('LOOKS OKMatrix Df check for date time', matrix_df.head())
# matrix_df=matrix_df.reset_index().rename(columns={'index':'unique_match_id'})
matrix_df_1=matrix_df.loc[:,['unique_match_id','Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv','Date','Home Points','Away Points']].copy()
# st.write('checking #1 matrix_df_1',matrix_df_1.head())

with st.beta_expander('Games Played to be used in Matrix Multiplication'):
    first_qtr=matrix_df_1.copy()
    start=-3
    finish=0
    first_4=first_qtr[first_qtr['Week'].between(start,finish)].copy()
    # st.write('checking first 4 #2',first_4)
    st.write('just want to see what year am i taking', first_qtr)
    def games_matrix_workings(first_4):
        group_week = first_4.groupby('Week')
        raw_data_2=[]
        game_weights = iter([-0.125, -0.25,-0.5,-1])
        for name, group in group_week:
            group['game_adj']=next(game_weights)
            # st.write('looking at for loop',group)
            raw_data_2.append(group)

        df3 = pd.concat(raw_data_2, ignore_index=True)
        adj_df3=df3.loc[:,['Home ID', 'Away ID', 'game_adj']].copy()
        test_adj_df3 = adj_df3.rename(columns={'Home ID':'Away ID', 'Away ID':'Home ID'})
        concat_df_test=pd.concat([adj_df3,test_adj_df3]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
        test_concat_df_test=concat_df_test.groupby('Home ID')['game_adj'].sum().abs().reset_index()
        test_concat_df_test['Away ID']=test_concat_df_test['Home ID']
        full=pd.concat([concat_df_test,test_concat_df_test]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
        # st.write('full',full)
        full_stack=pd.pivot_table(full,index='Away ID', columns='Home ID',aggfunc='sum')
        # st.write('full stack pivot THIS IS WHERE ISSUE IS', full_stack)
        # st.write('Check sum looks good all zero', full_stack.sum())
        full_stack=full_stack.fillna(0)
        full_stack.columns = full_stack.columns.droplevel(0)
        return full_stack
    # st.write('Check that First_4 is working', first_4)
    full_stack=games_matrix_workings(first_4)
    st.write('Check sum if True all good', full_stack.sum().sum()==0)
    st.write('this is 1st part games played, need to automate this for every week')

with st.beta_expander('CORRECT Testing reworking the DataFrame'):
    test_df['at_home'] = 1
    test_df['at_away'] = -1
    test_df['home_pts_adv'] = 3
    test_df['away_pts_adv'] = -3
    test_df['away_spread']=-test_df['Spread']
    test_df=test_df.rename(columns={'Spread':'home_spread'})
    # st.write('checking for unique match id',test_df)
    test_df_1=test_df.loc[:,['unique_match_id','Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv']].copy()
    
    # st.write(test_df_1.sort_values(by=['ID','Week'],ascending=True))
    test_df_home=test_df_1.loc[:,['Week','Home ID','at_home','home_spread','home_pts_adv']].rename(columns={'Home ID':'ID','at_home':'home','home_spread':'spread','home_pts_adv':'home_pts_adv'}).copy()
    test_df_away=test_df_1.loc[:,['Week','Away ID','at_away','away_spread','away_pts_adv']].rename(columns={'Away ID':'ID','at_away':'home','away_spread':'spread','away_pts_adv':'home_pts_adv'}).copy()
    test_df_2=pd.concat([test_df_home,test_df_away],ignore_index=True)
    test_df_2=test_df_2.sort_values(by=['ID','Week'],ascending=True)
    test_df_2['spread_with_home_adv']=test_df_2['spread']+test_df_2['home_pts_adv']
    st.write(test_df_2)

def test_4(matrix_df_1):
    weights = np.array([0.125, 0.25,0.5,1])
    sum_weights = np.sum(weights)
    matrix_df_1['adj_spread']=matrix_df_1['spread_with_home_adv'].rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    return matrix_df_1

with st.beta_expander('CORRECT Power Ranking to be used in Matrix Multiplication'):
    # # https://stackoverflow.com/questions/9621362/how-do-i-compute-a-weighted-moving-average-using-pandas
    grouped = test_df_2.groupby('ID')
    # https://stackoverflow.com/questions/16974047/efficient-way-to-find-missing-elements-in-an-integer-sequence
    # https://stackoverflow.com/questions/62471485/is-it-possible-to-insert-missing-sequence-numbers-in-python
    ranking_power=[]
    for name, group in grouped:
        dfseq = pd.DataFrame.from_dict({'Week': range( -3,21 )}).merge(group, on='Week', how='outer').fillna(np.NaN)
        dfseq['ID']=dfseq['ID'].fillna(method='ffill')
        dfseq['home_pts_adv']=dfseq['home_pts_adv'].fillna(0)
        dfseq['spread']=dfseq['spread'].fillna(0)
        dfseq['spread_with_home_adv']=dfseq['spread_with_home_adv'].fillna(0)
        dfseq['home']=dfseq['home'].fillna(0)
        df_seq_1 = dfseq.groupby(['Week','ID'])['spread_with_home_adv'].sum().reset_index()
        update=test_4(df_seq_1)
        ranking_power.append(update)
    df_power = pd.concat(ranking_power, ignore_index=True)
    st.write('power ranking',df_power.sort_values(by=['ID','Week'],ascending=[True,True]))
    st.write('power ranking',df_power.sort_values(by=['Week','ID'],ascending=[True,True]))

with st.beta_expander('CORRECT Power Ranking Matrix Multiplication'):
    # https://stackoverflow.com/questions/62775018/matrix-array-multiplication-whats-excel-doing-mmult-and-how-to-mimic-it-in#62775508
    # st.write('check')
    inverse_matrix=[]
    power_ranking=[]
    list_inverse_matrix=[]
    list_power_ranking=[]
    power_df=df_power.loc[:,['Week','ID','adj_spread']].copy()

    games_df=matrix_df_1.copy()
    st.write('Checking the games df', games_df.sort_values(by='Week'))
    first=list(range(-3,18))
    last=list(range(0,21))
    for first,last in zip(first,last):
        st.header('start xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # st.write('week no.', last)
        st.write('this is first',first)
        # st.write('this is first week ADJUSTED MATRIX is below',first)
        first_section=games_df[games_df['Week'].between(first,last)]
        # st.write('Checking the first_section',first_section.sort_values(by='Week'))
        full_game_matrix=games_matrix_workings(first_section)
        # st.write('this if full game matrix',full_game_matrix)
        adjusted_matrix=full_game_matrix.loc[0:12,0:12]
        # st.write('adjusted matrix',adjusted_matrix)
        # st.write('this is the last number ADJUSTED MATRIX is back up',last)
        
        df_inv = pd.DataFrame(np.linalg.pinv(adjusted_matrix.values), adjusted_matrix.columns, adjusted_matrix.index)
        # st.write('this is the inverse matrix',df_inv)
        # st.write('this is shape of inverse matrix', df_inv.shape)

        # st.write('check ID 10 AND week number 0 I think is last')
        # st.write('this is last', last)
        # st.write(power_df)
        power_df_week=power_df[power_df['Week']==last].drop_duplicates(subset=['ID'],keep='last').set_index('ID')\
        .drop('Week',axis=1).rename(columns={'adj_spread':0}).loc[:12,:]
        
        st.write('this is the power_df_week PROBLEM IS HERE ID 10',power_df_week)
        # st.write('this is the shape', power_df_week.shape)
        # st.write(pd.DataFrame(power_df_week).dtypes)
        # st.write('this is PD Dataframe power df week',pd.DataFrame(power_df_week) )
        result = df_inv.dot(pd.DataFrame(power_df_week))
        st.header('this is result of matrix multplication')
        st.write(result)
        result.columns=['power']
        avg=(result['power'].sum())/14
        result['avg_pwr_rank']=(result['power'].sum())/14
        result['final_power']=result['avg_pwr_rank']-result['power']
        df_pwr=pd.DataFrame(columns=['final_power'],data=[avg])
        result=pd.concat([result,df_pwr],ignore_index=True)
        
        result['week']=last+1
        power_ranking.append(result)
        # st.write('check result after concat', result)
        st.write('week no.', last)
        st.header('end xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    power_ranking_combined = pd.concat(power_ranking).reset_index().rename(columns={'index':'ID'})
    st.write('power ranking combined', power_ranking_combined)

with st.beta_expander('Adding Power Ranking to Matches'):
    matches_df = spread.copy()
    st.write('This is matches_df', matches_df.head())
    st.write('This is power ranking combined', power_ranking_combined.head())
    home_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Home ID'})
    away_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Away ID'})
    updated_df=pd.merge(matches_df,home_power_rank_merge,on=['Home ID','Week']).rename(columns={'final_power':'home_power'})
    updated_df=pd.merge(updated_df,away_power_rank_merge,on=['Away ID','Week']).rename(columns={'final_power':'away_power'})
    updated_df['calculated_spread']=updated_df['away_power']-updated_df['home_power']
    updated_df['spread_working']=updated_df['home_power']-updated_df['away_power']+updated_df['Spread']
    updated_df['power_pick'] = np.where(updated_df['spread_working'] > 0, 1,
    np.where(updated_df['spread_working'] < 0,-1,0))
    st.write(updated_df)

with st.beta_expander('Adding Season to Date Cover to Matches'):
    # df = pd.DataFrame([['mon',19,'cardinals', 3], ['tue',20,'patriots', 4], ['wed',20,'patriots', 5]], columns=['date','week','team', 'stdc'])
    # st.write('df1',df)
    # df2 = pd.DataFrame([['sun',18,'saints'], ['tue',20,'patriots'], ['wed',20,'patriots']], columns=['date','week','team'])
    # st.write('df2',df2)
    # df3=df2.merge(df,on=['date','week','team'], how='left')
    # st.write('merged on left',df3)  # merges on columns A

    # st.write('this is season to date cover', spread_3)
    stdc_home=spread_3.rename(columns={'ID':'Home ID'})
    stdc_home['cover_sign']=-stdc_home['cover_sign']
    stdc_away=spread_3.rename(columns={'ID':'Away ID'})
    updated_df=updated_df.drop(['away_cover'],axis=1)
    # st.header('Check')
    # st.write('check updated df #1',updated_df)
    updated_df=updated_df.rename(columns={'home_cover':'home_cover_result'})
    updated_df=updated_df.merge(stdc_home,on=['Date','Week','Home ID'],how='left').rename(columns={'cover':'home_cover','cover_sign':'home_cover_sign'})
    # st.write('check updated df #2', updated_df)
    updated_df=pd.merge(updated_df,stdc_away,on=['Date','Week','Away ID'],how='left').rename(columns={'cover':'away_cover','cover_sign':'away_cover_sign'})
    # st.write('check that STDC coming in correctly', updated_df)
    # st.write('Check Total')
    # st.write('home',updated_df['home_cover_sign'].sum())
    # st.write('away',updated_df['away_cover_sign'].sum())
    # st.write('Updated for STDC', updated_df)
    # st.write('Get STDC by Week do something similar for Power Rank')
    # last_occurence = spread_3.groupby(['ID'],as_index=False).last()
    # st.write(last_occurence)
    stdc_df=pd.merge(spread_3,team_names_id,on='ID').rename(columns={'Home Team':'Team'})
    st.write('Check for Team as causing an issue?', stdc_df)
    stdc_df=stdc_df.loc[:,['Week','Team','cover']].copy()
    # stdc_df['last_week']=
    # stdc_df['Week']=stdc_df['Week'].replace({17:'week_17'})
    
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    # st.write(stdc_df.sort_values(by=['Team','Week']))
    
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)
    # st.write(stdc_pivot)

    chart_cover= alt.Chart(stdc_df).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.beta_expander('Adding Turnover to Matches'):
    st.write('this is turnovers', turnover_3)
    turnover_matches = turnover_3.loc[:,['Date','Week','ID','prev_turnover', 'turnover_sign']].copy()
    turnover_home=turnover_matches.rename(columns={'ID':'Home ID'})
    
    turnover_away=turnover_matches.rename(columns={'ID':'Away ID'})
    turnover_away['turnover_sign']=-turnover_away['turnover_sign']
    updated_df=pd.merge(updated_df,turnover_home,on=['Date','Week','Home ID'],how='left').rename(columns={'prev_turnover':'home_prev_turnover','turnover_sign':'home_turnover_sign'})
    updated_df=pd.merge(updated_df,turnover_away,on=['Date','Week','Away ID'],how='left').rename(columns={'prev_turnover':'away_prev_turnover','turnover_sign':'away_turnover_sign'})
    # st.write('check matches week 20', updated_df)
    # TEST Workings
    # st.write('check that Turnover coming in correctly', updated_df[updated_df['Week']==18])
    # st.write('Check Total')
    # st.write('home',updated_df['home_turnover_sign'].sum())
    # st.write('away',updated_df['away_turnover_sign'].sum())
    # turnover_excel=test_data_2020.loc[:,['Week','Home ID','Home Team', 'Away ID', 'Away Team','excel_home_prev_turnover','excel_away_prev_turnover','excel_home_turnover_sign','excel_away_turnover_sign']].copy()
    # test_turnover=pd.merge(updated_df,turnover_excel)
    # test_turnover['test_1']=test_turnover['home_prev_turnover']-test_turnover['excel_home_prev_turnover']
    # test_turnover['test_2']=test_turnover['away_prev_turnover']-test_turnover['excel_away_prev_turnover']
    # st.write(test_turnover[test_turnover['test_1']!=0])
    # st.write(test_turnover[test_turnover['test_2']!=0])
    # st.write(test_turnover)

with st.beta_expander('Betting Slip Matches'):
    betting_matches=updated_df.loc[:,['Week','Date','Home ID','Home Team','Away ID', 'Away Team','Spread','Home Points','Away Points',
    'home_power','away_power','home_cover','away_cover','home_turnover_sign','away_turnover_sign','home_cover_sign','away_cover_sign','power_pick','home_cover_result']]
    # st.write('check for duplicate home cover', betting_matches.sort_values(by=['Week','Date']))
    betting_matches['total_factor']=betting_matches['home_turnover_sign']+betting_matches['away_turnover_sign']+betting_matches['home_cover_sign']+\
    betting_matches['away_cover_sign']+betting_matches['power_pick']
    betting_matches['bet_on'] = np.where(betting_matches['total_factor']>2,betting_matches['Home Team'],np.where(betting_matches['total_factor']<-2,betting_matches['Away Team'],''))
    betting_matches['bet_sign'] = (np.where(betting_matches['total_factor']>2,1,np.where(betting_matches['total_factor']<-2,-1,0)))
    betting_matches['bet_sign'] = betting_matches['bet_sign'].astype(float)
    betting_matches['home_cover'] = betting_matches['home_cover'].astype(float)
    # st.write('this is bet sign',betting_matches['bet_sign'].dtypes)
    # st.write('this is home cover',betting_matches['home_cover'].dtypes)
    betting_matches['result']=betting_matches['home_cover_result'] * betting_matches['bet_sign']
    st.write('testing sum of betting result',betting_matches['result'].sum())

    # this is for graphing anlaysis on spreadsheet
    betting_matches['bet_sign_all'] = (np.where(betting_matches['total_factor']>0,1,np.where(betting_matches['total_factor']<-0,-1,0)))
    betting_matches['result_all']=betting_matches['home_cover_result'] * betting_matches['bet_sign_all']
    st.write('testing sum of betting all result',betting_matches['result_all'].sum())
    # st.write('testing factor')
    # st.write(betting_matches['total_factor'].sum())
    cols_to_move=['Week','Date','Home Team','Away Team','bet_on','Spread','home_power','away_power','Home Points','Away Points','total_factor']
    cols = cols_to_move + [col for col in betting_matches if col not in cols_to_move]
    betting_matches=betting_matches[cols]
    betting_matches=betting_matches.sort_values('Date')
    # st.write(betting_matches)
    # st.write(betting_matches.dtypes)
    presentation_betting_matches=betting_matches.sort_values(by=['Week','Date']).copy()

    
    # def color_negative_red(val):
    #     color = 'red' if val < 0 else 'black'
    #     return 'color: %s' % color
    # presentation_betting_matches['Spread'] = presentation_betting_matches['Spread'].apply(color_negative_red)




    # presentation_betting_matches['home_power'] = presentation_betting_matches['home_power'].apply("{:.1f}".format)
    # presentation_betting_matches['away_power'] = presentation_betting_matches['away_power'].apply("{:.1f}".format)
    # presentation_betting_matches['Date'] = presentation_betting_matches['Date'].dt.strftime('%m/%d/%Y')
    
    

    # AgGrid(presentation_betting_matches)
    

    # https://towardsdatascience.com/7-reasons-why-you-should-use-the-streamlit-aggrid-component-2d9a2b6e32f0
    grid_height = st.number_input("Grid height", min_value=400, value=550, step=100)
    gb = GridOptionsBuilder.from_dataframe(presentation_betting_matches)
    gb.configure_column("Spread", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("home_power", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("away_power", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("Date", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='dd-MM-yyyy', pivot=True)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)



    test_cellsytle_jscode = JsCode("""
    function(params) {
        if (params.value < 0) {
        return {
            'color': 'red',
        }
        } else {
            return {
                'color': 'black',
            }
        }
    };
    """)
    # # https://github.com/PablocFonseca/streamlit-aggrid/blob/main/st_aggrid/grid_options_builder.py
    gb.configure_column(field="Spread", cellStyle=test_cellsytle_jscode)
    gb.configure_column("home_power", cellStyle=test_cellsytle_jscode)
    gb.configure_column("away_power", cellStyle=test_cellsytle_jscode)


    # gb.configure_pagination()
    # gb.configure_side_bar()
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    grid_response = AgGrid(
        presentation_betting_matches, 
        gridOptions=gridOptions,
        height=grid_height, 
        width='100%',
        # data_return_mode=return_mode_value, 
        # update_mode=update_mode_value,
        # fit_columns_on_grid_load=fit_columns_on_grid_load,
        allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
        enable_enterprise_modules=True,
    )

    # container.grid_response
    # AgGrid(betting_matches.sort_values('Date').style.format({'home_power':"{:.1f}",'away_power':"{:.1f}"}))
    




    # st.write('Below is just checking an individual team')
    # st.write( betting_matches[(betting_matches['Home Team']=='Arizona Cardinals') | 
    # (betting_matches['Away Team']=='Arizona Cardinals')].set_index('Week').sort_values(by='Date') )

with st.beta_expander('Power Ranking by Week'):
    power_week=power_ranking_combined.copy()
    # st.write('power', power_week)

    # pivot_df=power_week.loc[:,['ID','final_power','week']].copy()
    team_names_id=team_names_id.rename(columns={'Home Team':'Team'})
    id_names=team_names_id.drop_duplicates(subset=['ID'], keep='first')
    pivot_df=pd.merge(power_week,id_names, on='ID')
    # st.write('after merge', pivot_df)
    pivot_df=pivot_df.loc[:,['Team','final_power','week']].copy()
    # st.write('graphing?',pivot_df)
    power_pivot=pd.pivot_table(pivot_df,index='Team', columns='week')
    pivot_df_test = pivot_df.copy()
    pivot_df_test=pivot_df_test[pivot_df_test['week']<19]
    pivot_df_test['average']=pivot_df.groupby('Team')['final_power'].transform(np.mean)
    # st.write('graphing?',pivot_df_test)
    power_pivot.columns = power_pivot.columns.droplevel(0)
    power_pivot['average'] = power_pivot.mean(axis=1)
    # st.write(power_pivot)
    # https://stackoverflow.com/questions/67045668/altair-text-over-a-heatmap-in-a-script
    pivot_df=pivot_df.sort_values(by='final_power',ascending=False)
    chart_power= alt.Chart(pivot_df_test).mark_rect().encode(alt.X('week:O',axis=alt.Axis(title='week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('final_power:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text=chart_power.mark_text().encode(text=alt.Text('final_power:N',format=",.0f"),color=alt.value('black'))
    st.altair_chart(chart_power + text,use_container_width=True)
    # https://github.com/altair-viz/altair/issues/820#issuecomment-386856394
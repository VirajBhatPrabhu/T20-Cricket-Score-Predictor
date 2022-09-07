import pandas as pd
import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))

page_bg_img = '''
    <style>
    .stApp {
    background-image: url("https://wallpaperset.com/w/full/5/b/6/239891.jpg#.YxiVlSqGJcE.link");
    background-size: cover;
    }
    </style>
    '''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
        f'<h1 style="text-align: center; color:black;">T20 score predictor</h1>',
        unsafe_allow_html=True)

teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan',
         'Pakistan', 'Sri Lanka']

venue = ['Melbourne Cricket Ground', 'Simonds Stadium, South Geelong',
         'Adelaide Oval', 'McLean Park', 'Bay Oval', 'Eden Park',
         'The Rose Bowl', 'County Ground', 'Sophia Gardens',
         'Riverside Ground', 'Green Park',
         'Vidarbha Cricket Association Stadium, Jamtha',
         'M Chinnaswamy Stadium',
         'Central Broward Regional Park Stadium Turf Ground',
         'Dubai International Cricket Stadium', 'Sheikh Zayed Stadium',
         'Sydney Cricket Ground', 'Bellerive Oval', 'Westpac Stadium',
         'Seddon Park', 'Mangaung Oval', 'Senwes Park',
         'Kensington Oval, Bridgetown', "Queen's Park Oval, Port of Spain",
         'R Premadasa Stadium', 'Warner Park, Basseterre',
         'Sabina Park, Kingston', 'R.Premadasa Stadium, Khettarama',
         'Saxton Oval', 'JSCA International Stadium Complex', 'Edgbaston',
         'Old Trafford', 'Arun Jaitley Stadium',
         'Saurashtra Cricket Association Stadium',
         'Greenfield International Stadium', 'Gaddafi Stadium',
         'The Wanderers Stadium', 'SuperSport Park', 'Newlands',
         'Barabati Stadium', 'Holkar Cricket Stadium', 'Wankhede Stadium',
         'Shere Bangla National Stadium, Mirpur',
         'Sylhet International Cricket Stadium', 'National Stadium',
         'Harare Sports Club', 'Carrara Oval',
         'Brisbane Cricket Ground, Woolloongabba',
         'Rajiv Gandhi International Cricket Stadium, Dehradun',
         'Rajiv Gandhi International Cricket Stadium', 'Eden Gardens',
         'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
         'MA Chidambaram Stadium, Chepauk',
         'Darren Sammy National Cricket Stadium, St Lucia',
         'Warner Park, St Kitts',
         'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
         'M.Chinnaswamy Stadium', 'Manuka Oval', 'Perth Stadium',
         'Buffalo Park', 'Kingsmead', "St George's Park",
         'Punjab Cricket Association IS Bindra Stadium, Mohali',
         'Rajiv Gandhi International Stadium, Uppal', 'Hagley Oval',
         'Providence Stadium, Guyana',
         'Pallekele International Cricket Stadium',
         'Zahur Ahmed Chowdhury Stadium',
         'Maharashtra Cricket Association Stadium', 'University Oval',
         'Sky Stadium', 'Boland Park', 'Trent Bridge, Nottingham',
         'Headingley, Leeds', 'Old Trafford, Manchester',
         'Narendra Modi Stadium', 'Sophia Gardens, Cardiff',
         'The Rose Bowl, Southampton',
         'The Wanderers Stadium, Johannesburg',
         'SuperSport Park, Centurion', 'Coolidge Cricket Ground, Antigua',
         'Kensington Oval, Bridgetown, Barbados',
         'R Premadasa Stadium, Colombo',
         "National Cricket Stadium, St George's, Grenada",
         'Daren Sammy National Cricket Stadium, Gros Islet, St Lucia',
         'Manuka Oval, Canberra', 'Zayed Cricket Stadium, Abu Dhabi',
         'Sharjah Cricket Stadium', 'Edgbaston, Birmingham',
         'County Ground, Bristol', 'Sawai Mansingh Stadium, Jaipur',
         'JSCA International Stadium Complex, Ranchi',
         'Eden Gardens, Kolkata',
         'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
         'Himachal Pradesh Cricket Association Stadium, Dharamsala',
         'Arun Jaitley Stadium, Delhi', 'Barabati Stadium, Cuttack',
         'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
         'Saurashtra Cricket Association Stadium, Rajkot',
         'National Stadium, Karachi', 'Gaddafi Stadium, Lahore',
         'Windsor Park, Roseau, Dominica',
         'Brian Lara Stadium, Tarouba, Trinidad',
         'Warner Park, Basseterre, St Kitts', 'New Wanderers Stadium',
         'Kennington Oval', 'Western Australia Cricket Association Ground',
         'Brabourne Stadium', 'Jade Stadium', 'Gymkhana Club Ground',
         'Maple Leaf North-West Ground', 'Eden Park, Auckland',
         'AMI Stadium', 'Providence Stadium',
         'Beausejour Stadium, Gros Islet',
         'Sir Vivian Richards Stadium, North Sound',
         'Moses Mabhida Stadium', 'Stadium Australia',
         'Shere Bangla National Stadium',
         'Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa',
         'Trent Bridge', 'Subrata Roy Sahara Stadium',
         'Sardar Patel Stadium, Motera', 'Arnos Vale Ground, Kingstown',
         'Windsor Park, Roseau',
         'Himachal Pradesh Cricket Association Stadium', 'Feroz Shah Kotla']

col1, col2, col3 = st.columns(3)

with col1:
    batting_team = st.selectbox('Batting Team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Bowling Team', sorted(teams))

with col3:
    venue = st.selectbox('Venue', sorted(venue))

col4, col5 = st.columns(2)
with col4:
    toss_winner = st.selectbox('Toss winner', [batting_team, bowling_team])

with col5:
    toss_decision = st.selectbox('Toss Decision', ['field', 'bat'])


col6, col7, col8,col9 = st.columns(4)

with col6:
    over_no = st.number_input('Ongoing Over', min_value=5, max_value=20, step=1,help='Works for overs > 4')

with col7:
    current_score = st.number_input('Current Score', min_value=1, max_value=300, step=1)

with col8:
    player_dismissed = st.number_input('Wickets Fallen', min_value=0, max_value=9, step=1)

with col9:
    scores_lastfive = st.number_input('Score Last 5 Overs', min_value=0, max_value=200, step=1)



if st.button('Predict Score'):
    balls_left = 120 - (over_no)
    wickets_remaining = 10 - player_dismissed
    crr = current_score / over_no

    input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'venue': [venue], 'toss_winner': [toss_winner],
         'toss_decision': [toss_decision], 'bowling_team': [bowling_team], 'current_score': [current_score],
         'wickets_remaining': [wickets_remaining], 'crr': [crr], 'balls_left': [balls_left],
         'scores_lastfive': [scores_lastfive]})
    result = model.predict(input_df)


    st.markdown(
        f'<h1 style="text-align: center; color:#1793C0;">{batting_team} WIll Score {str(int(result[0]))}  Runs</h1>',
        unsafe_allow_html=True)

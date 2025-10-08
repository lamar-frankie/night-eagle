import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="Operation Night Eagle",
    page_icon="🐅",
    layout="wide"
)

# --- Web Scraping Function for Real-time Rankings ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_college_football_rankings():
    """
    Scrapes the ESPN website for AP Top 25 College Football rankings.
    Returns a dictionary mapping team names to their rank.
    Unranked teams are assigned a rank of 26.
    """
    url = "https://www.espn.com/college-football/rankings"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching rankings: {e}")
        return {}

    soup = BeautifulSoup(response.content, 'html.parser')
    teams = {}

    # Find the table containing the rankings
    ranking_table = soup.find('table', class_='Table')

    if not ranking_table:
        st.warning("Could not find the rankings table on the page. The page structure may have changed.")
        return {}

    rows = ranking_table.find('tbody').find_all('tr')

    for row in rows:
        cells = row.find_all('td')
        if len(cells) > 1:
            try:
                rank = int(cells[0].text.strip())
                team_name_tag = cells[1].find('span', class_='hide-mobile')
                if team_name_tag:
                    team_name = team_name_tag.text.strip()
                    teams[team_name] = rank
            except (ValueError, AttributeError):
                continue # Skip header rows or malformed rows

    return teams

# --- Statistical Model ---
@st.cache_resource
def get_trained_model():
    """
    Creates a sample historical dataset, trains a Logistic Regression model,
    and returns the trained model.
    """
    # Simulate 10 years of historical data (approx. 7 home games per year)
    np.random.seed(42)
    num_years = 10
    games_per_year = 7
    total_games = num_years * games_per_year

    data = []
    for _ in range(total_games):
        # --- Generate Features ---
        auburn_rank = np.random.randint(5, 27) # Auburn's rank usually ranges from good to unranked
        opponent_rank = np.random.randint(1, 27)

        # Simulate SEC vs. non-conference schedule
        is_sec = np.random.choice([1, 0], p=[0.6, 0.4]) # 60% of home games are SEC

        # Rivalry games (mostly Georgia at home every other year)
        is_rivalry = 1 if is_sec and np.random.rand() < 0.15 else 0 # Small chance of being the key rival

        # Game month (non-conf early, SEC later)
        if is_sec:
            game_month = np.random.choice([9, 10, 11])
        else:
            game_month = np.random.choice([9, 11], p=[0.8, 0.2]) # Most non-conf in Sep

        # --- Determine Target (Night_Game) based on rules ---
        night_game_prob = 0.1 # Base probability

        # Higher rank = higher chance
        if opponent_rank <= 10: night_game_prob += 0.3
        if auburn_rank <= 10: night_game_prob += 0.2

        # SEC games are more likely to be at night
        if is_sec: night_game_prob += 0.2

        # Rivalry games are prime candidates
        if is_rivalry: night_game_prob += 0.4

        # November games are slightly less likely to be night games unless major matchup
        if game_month == 11 and not is_rivalry: night_game_prob -= 0.1

        # September non-conf games are almost never night games
        if game_month == 9 and not is_sec: night_game_prob -= 0.2

        # Clip probability to be between 0 and 1
        night_game_prob = np.clip(night_game_prob, 0.05, 0.95)

        night_game = 1 if np.random.rand() < night_game_prob else 0

        data.append([opponent_rank, auburn_rank, is_rivalry, is_sec, game_month, night_game])

    columns = ['Opponent_Rank', 'Auburn_Rank', 'Is_Rivalry', 'Is_SEC', 'Game_Month', 'Night_Game']
    df = pd.DataFrame(data, columns=columns)

    # Define features (X) and target (y)
    X = df[['Opponent_Rank', 'Auburn_Rank', 'Is_Rivalry', 'Is_SEC', 'Game_Month']]
    y = df['Night_Game']

    # Train the logistic regression model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)

    return model

def main():
    # --- Custom CSS for Auburn Theme ---
    primary_blue = "#0C2340"
    primary_orange = "#E87722"

    st.markdown(f"""
        <style>
            .stApp {{
                background-color: {primary_blue};
                color: white;
            }}
            .header-container {{
                text-align: center;
                margin-bottom: 2rem;
            }}
            .header-container img {{
                width: 150px;
            }}
            .header-container h1 {{
                color: {primary_orange};
                padding-top: 10px;
            }}
            h2, h3 {{
                color: {primary_orange};
            }}
            [data-testid="stMetric"] {{
                background-color: #0C2340;
                border: 1px solid {primary_orange};
                border-radius: 10px;
                padding: 10px;
                text-align: center;
            }}
            [data-testid="stMetricLabel"] {{
                color: {primary_orange};
            }}
            [data-testid="stMetricValue"] {{
                color: white;
            }}
            /* Make analysis text white */
            .stAlert p {{
                color: white !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="header-container">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Auburn_Tigers_logo.svg/1200px-Auburn_Tigers_logo.svg.png">
            <h1>Operation Night Eagle</h1>
            <h2>Auburn Football Night Game Predictor</h2>    
        </div>
    """, unsafe_allow_html=True)

    rankings = get_college_football_rankings()

    if not rankings:
        st.error("Could not retrieve team rankings. Please check the connection or try again later.")
        return

    # --- UI for Opponent Selection ---
    st.header("Choose an Opponent for the 2025 Season")
    opponents = ['Alabama A&M', 'Ball State', 'Georgia', 'LSU', 'Oklahoma', 'South Alabama', 'Vanderbilt']

    selected_opponent = st.selectbox(
        "Select an opponent from the list of 2025 home games:",
        options=opponents
    )

    # --- Display Ranks ---
    st.header("Current Team Rankings (AP Top 25)")

    # Use .get() to handle unranked teams, defaulting to 26
    auburn_rank = rankings.get('Auburn', 26)
    opponent_rank = rankings.get(selected_opponent, 26)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Auburn's Rank", value=f"#{auburn_rank}")
    with col2:
        st.metric(label=f"{selected_opponent}'s Rank", value=f"#{opponent_rank}")

    # --- Prediction Logic ---
    st.header("Night Game Prediction")

    # Get the trained model
    model = get_trained_model()

    # Define game-specific data for the 2025 schedule
    game_data = {
        'Alabama A&M': {'Is_SEC': 0, 'Is_Rivalry': 0, 'Game_Month': 9},
        'Ball State': {'Is_SEC': 0, 'Is_Rivalry': 0, 'Game_Month': 9},
        'Georgia': {'Is_SEC': 1, 'Is_Rivalry': 1, 'Game_Month': 10}, # Moved to Oct in 2025
        'LSU': {'Is_SEC': 1, 'Is_Rivalry': 0, 'Game_Month': 10},
        'Oklahoma': {'Is_SEC': 1, 'Is_Rivalry': 0, 'Game_Month': 9},
        'South Alabama': {'Is_SEC': 0, 'Is_Rivalry': 0, 'Game_Month': 9},
        'Vanderbilt': {'Is_SEC': 1, 'Is_Rivalry': 0, 'Game_Month': 11},
    }

    # Prepare input for the model based on user selection
    opponent_info = game_data[selected_opponent]
    input_data = pd.DataFrame({
        'Opponent_Rank': [opponent_rank],
        'Auburn_Rank': [auburn_rank],
        'Is_Rivalry': [opponent_info['Is_Rivalry']],
        'Is_SEC': [opponent_info['Is_SEC']],
        'Game_Month': [opponent_info['Game_Month']]
    })

    # Predict the probability
    prediction_proba = model.predict_proba(input_data)[0][1] # Probability of 'Night_Game' = 1
    prediction_percentage = prediction_proba * 100

    # Display the prediction score
    st.metric(
        label="Probability of a Night Game",
        value=f"{prediction_percentage:.1f}%"
    )

    # --- Analysis Section ---
    st.subheader("Analysis")

    # Critical logic for CBS Game of the Week
    is_cbs_candidate = (opponent_info['Is_SEC'] == 1 and
                        opponent_rank <= 15 and
                        auburn_rank <= 25 and
                        selected_opponent in ['Georgia', 'LSU', 'Oklahoma'])

    if is_cbs_candidate:
        st.warning(
            """
            **High Likelihood Warning:** This game is a prime candidate for the CBS 2:30 PM CT 'Game of the Week' slot.
            This is the most-watched SEC broadcast window, and if selected, it would prevent the game from being played at night.
            """
        )

    if prediction_proba > 0.65:
        st.info(
            f"""
            **High Probability:** A score this high suggests a very attractive matchup for a primetime broadcast.
            Factors like a highly-ranked SEC opponent ({selected_opponent}), rivalry implications, or a key conference game
            make it a strong candidate for an evening kickoff on ESPN or ABC.
            """
        )
    elif prediction_proba < 0.35:
        st.info(
            """
            **Low Probability:** This score indicates the game is likely to be an early kickoff.
            This is common for non-conference games against unranked opponents or less anticipated SEC matchups.
            Expect an 11 AM CT start time, likely broadcast on the SEC Network or ESPN+.
            """
        )
    else:
        st.info(
            """
            **Toss-Up:** This matchup falls into a flexible scheduling window. Kickoff time could be determined later in the season
            based on team performance. It could be an afternoon game (like 2:30 or 3:00 PM CT) or an early evening game (6:00 PM CT).
            """
        )

if __name__ == "__main__":
    main()

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd


def scrape_website(url, element_type=None, class_name=None, element_id=None):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove status divs
        for div in soup.find_all('div', class_='event-cell__status event-cell__status__position'):
            div.decompose()

        elements = soup.find_all(element_type) if element_type else soup.find_all()

        if class_name:
            elements = [elem for elem in elements if elem.has_attr('class') and class_name in elem['class']]

        if element_id:
            elements = [elem for elem in elements if elem.has_attr('id') and elem['id'] == element_id]

        return elements

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the webpage: {str(e)}")
        return []
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return []


def parse_event_row(row):
    try:
        th = row.find("th")
        team_div = th.find("div", class_="event-cell__name-logo-wrapper")
        team_name = team_div.get_text(strip=False) if team_div else "Unknown"

        tds = row.find_all("td")
        if len(tds) < 3:
            return None

        outcomes = []
        for td in tds[:3]:
            outcome_div = td.find("div", class_="sportsbook-outcome-body-wrapper")
            outcome_text = outcome_div.get_text(strip=False) if outcome_div else "N/A"
            outcomes.append(outcome_text)

        return {
            "team": team_name,
            "spread": outcomes[0],
            "total": outcomes[1],
            "moneyline": outcomes[2]
        }

    except Exception as e:
        print(f"Error parsing row: {e}")
        return None


def table_to_custom_dataframe(table):
    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    rows = []
    for tr in tbody.find_all("tr"):
        parsed = parse_event_row(tr)
        if parsed:
            rows.append(parsed)

    return pd.DataFrame(rows)


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def main():
    st.title("Upcoming NBA Game Odds")

    draft_kings_url = "https://sportsbook.draftkings.com/leagues/basketball/nba"
    elements = scrape_website(draft_kings_url, "table", "sportsbook-table", None)

    if not elements:
        st.warning("No tables found or an error occurred.")
    else:
        for table in elements:
            df = table_to_custom_dataframe(table)
            if not df.empty:
                # Group teams in pairs (assuming each game has 2 teams)
                for i in range(0, len(df), 2):
                    if i + 1 < len(df):  # Make sure we have a pair
                        team1 = df.iloc[i]['team']
                        team2 = df.iloc[i + 1]['team']

                        # Create dropdown for this game
                        with st.expander(f"{team1} @ {team2}"):
                            # Display both teams' data


                            # ADD TEXT HERE ABOUT WHICH TEAM WE PREDICT TO WIN



                            game_df = df.iloc[i:i + 2].reset_index(drop=True)
                            st.dataframe(game_df,
                                         hide_index=True
                                         )
            else:
                st.markdown("### Game Odds Table (no valid rows parsed)")
                st.code(str(table), language='html')


if __name__ == "__main__":
    main()
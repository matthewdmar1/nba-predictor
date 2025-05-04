import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def scrape_website(url, element_type=None, class_name=None, element_id=None):
    try:
        # Send a GET request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove divs with class 'event-cell__status event-cell__status__position'
        for div in soup.find_all('div', class_='event-cell__status event-cell__status__position'):
            div.decompose()

        # Filter elements based on user input
        if element_type or class_name or element_id:
            # Find elements with the specified tag
            if element_type:
                elements = soup.find_all(element_type)
            else:
                elements = soup.find_all()

            # Filter by class if specified
            if class_name:
                elements = [elem for elem in elements if elem.has_attr('class') and class_name in elem['class']]

            # Filter by id if specified
            if element_id:
                elements = [elem for elem in elements if elem.has_attr('id') and elem['id'] == element_id]

            # If no elements found with filters, return all text
            if not elements:
                return soup.get_text(separator='\n', strip=True)

            # Return the HTML of filtered elements
            return ''.join([str(elem) for elem in elements])
        else:
            # Return all HTML if no filters are specified
            return str(soup)

    except requests.exceptions.RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def main():
    st.title("Sports Books Odds")

    draft_kings_url = "https://sportsbook.draftkings.com/leagues/basketball/nba"
    draft_kings_content = scrape_website(draft_kings_url, "div", "parlay-card-10-a", None)

    if draft_kings_content.startswith("Error"):
        st.error(draft_kings_content)
    else:
        css = """
        <style>
            .event-cell__logo {
                max-width: 20px;
                max-height: 20px;
            }
            .sportsbook-table {
                width: 100%;
                table-layout: fixed;
                border-collapse: separate;
                border-spacing: 0;
                padding: 8px;
                background-color: #121212;
            }
            .wrapper-event-cell {
                width: 100%;
            }
            .event-cell {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 10px;
                flex-wrap: nowrap;
                flex-direction: row;
                color: white;
            }
            .event-cell:link {
                text-decoration: none;
            }
            .event-cell-link {
                width: 100%;
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                flex-direction: column;
                padding-left: 10px;
            }
            .event-cell__name-logo-wrapper {
              display: flex;
              flex-direction: row;
              align-items: flex-start;
              justify-content: space-between;
              width: 100%
              padding-left: 10px;
            }
        </style>
        """
        st.markdown(css + draft_kings_content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

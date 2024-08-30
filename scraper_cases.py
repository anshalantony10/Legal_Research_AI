import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import json

def cases_keyword_search_scrape_bailii(keywords):
    base_url = "https://www.bailii.org/cgi-bin/lucy_search_1.cgi"
    
    # Prepare the query parameters
    query = ' OR '.join(f'({keyword.strip()})' for keyword in keywords)
    params = {
        'method': 'boolean',
        'datehigh': '',
        'sort': 'rank',
        'highlight': '1',
        'query': query,
        'mask_path': 'ie/cases',
        'datelow': ''
    }
    # Construct the URL
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    print(f"Searching URL: {url}")
    
    # Set up a session with headers that mimic a real browser
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.bailii.org/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })

    # Implement a delay before making the request
    time.sleep(2)
    
    # Send a GET request to the search results page
    response = session.get(url)
    print(f"Response status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Print the title of the page to verify we're on the right page
    print(f"Page title: {soup.title.string if soup.title else 'No title found'}")
    
    # Extract and print the results
    results = soup.select('ol li p i a')
    print(f"Number of results found: {len(results)}")
    
    if len(results) == 0:
        print("No results found. Printing page content for debugging:")
        print(soup.prettify()[:1000])  # Print first 1000 characters of the page
    
    case_data = []
    for result in results:
        if 'View without highlighting' in result.text:
            case_url = f"https://www.bailii.org{result['href']}"
            case_id = case_url.split('/')[-1].replace('.html', '')
            print(f"Fetching case: {case_id}")
            case_content = fetch_case_content(session, case_url)
            case_data.append({
                'case_id': case_id,
                'url': case_url,
                'content': case_content
            })
            time.sleep(1)  # Delay between requests to be respectful
    
    print("Scraping complete.")
    return case_data

def fetch_case_content(session, url):
    response = session.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract the main content of the case
        # This selector might need to be adjusted based on the actual structure of the case pages
        content = soup.select_one('body')
        if content:
            return content.get_text(strip=True)
    return "Failed to fetch content"

def save_to_json(data, filename='case_data.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Example usage
if __name__ == "__main__":
    keywords = input("Enter keywords to search for: ")
    case_data = cases_keyword_search_scrape_bailii(keywords)
    save_to_json(case_data)
    print(f"Data saved to case_data.json. Total cases: {len(case_data)}")
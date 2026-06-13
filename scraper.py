from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
import time

# Setup Chrome Driver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

print("Starting IMDb 2024 movie scraper...")

# IMDb feature films sorted by popularity
url = "https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31"

driver.get(url)

time.sleep(5)

movies = []

try:
    movie_cards = driver.find_elements(
        By.CSS_SELECTOR,
        "li.ipc-metadata-list-summary-item"
    )

    print(f"Found {len(movie_cards)} movies")

    for card in movie_cards[:50]:

        try:
            title = card.find_element(
                By.CSS_SELECTOR,
                "h3"
            ).text

        except:
            title = "N/A"

        try:
            storyline = card.find_element(
                By.CSS_SELECTOR,
                "div.ipc-html-content-inner-div"
            ).text

        except:
            storyline = "Storyline not available"

        movies.append({
            "Movie Name": title,
            "Storyline": storyline
        })

        print(f"Collected: {title}")

except Exception as e:
    print("Error:", e)

driver.quit()

# Save CSV
df = pd.DataFrame(movies)

df.to_csv(
    "data/imdb_movies_2024.csv",
    index=False,
    encoding="utf-8"
)

print("\n✅ Scraping completed!")
print(f"Total movies scraped: {len(df)}")
print("Saved to: data/imdb_movies_2024.csv")
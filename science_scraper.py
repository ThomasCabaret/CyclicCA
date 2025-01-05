import requests
import time
import random
from bs4 import BeautifulSoup
import pandas as pd
import os
import csv
import re
import json

# Updated HEADERS with more browser-like headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://scholar.google.com/",
    "Connection": "keep-alive",
    "DNT": "1",  # Do Not Track request header
}

# Helper function to rotate User-Agent strings
def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    ]
    return random.choice(user_agents)

search_keywords = {
    "origin_of_life": [
        "origin of life",
        "abiogenesis",
        "chemical evolution"
    ],
    "synthetic_biology": [
        "synthetic biology",
        "artificial cell",
        "synthetic cell"
    ],
}

with open("scoring_keywords.json", "r", encoding="utf-8") as f:
    scoring_keywords = json.load(f)

for theme, kws in search_keywords.items():
    for kw in kws:
        if kw not in scoring_keywords.get(theme, []):
            raise ValueError(f"Search keyword '{kw}' is missing from scoring keywords for theme '{theme}'.")

CSV_FILE = "articles.csv"
START_YEAR = 2010
START_DATE = "2010-01-01"
END_DATE = "2100-01-01"
DEFAULT_MAX_PAGES = 3
MAX_GS_PAGES = DEFAULT_MAX_PAGES
MAX_ARXIV_PAGES = DEFAULT_MAX_PAGES
SLEEP_SECONDS = 3

def load_existing_articles():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE).to_dict(orient="records")
    return []

def clean_string(s):
    if not s:
        return ""
    s = s.replace('"', "'").replace("\n", " ")
    s = re.sub(r"[^\x00-\x7F]+", "_", s)
    return s

def clean_article(article):
    for k, v in article.items():
        if isinstance(v, str):
            article[k] = clean_string(v)
    return article

def save_articles_to_csv(articles):
    df = pd.DataFrame(articles)
    df.to_csv(
        CSV_FILE,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

def is_duplicate(article, existing_articles):
    return any(existing["title"] == article["title"] for existing in existing_articles)

def compute_theme_scores(title, abstract):
    text = (title + " " + abstract).lower()
    scores = {}
    for theme, keywords in scoring_keywords.items():
        score = 0
        for kw in keywords:
            if kw.lower() in text:
                score += 1
        scores[theme] = score
    return scores

def determine_dominant_theme(scores):
    mx = max(scores.values()) if scores else 0
    if mx == 0:
        return ""
    top = [t for t, s in scores.items() if s == mx]
    if len(top) == 1:
        return top[0]
    return "multiple"

def keyword_in_text(keyword, text):
    return keyword.lower() in text.lower()

def parse_date(date_text, separator="-"):
    if not date_text:
        return ""
    parts = date_text.split(separator)
    if parts:
        return parts[-1].strip()
    return ""

def scrape_site(
    keyword,
    url_template,
    item_selector,
    title_selector,
    abstract_selector,
    date_selector,
    link_selector,
    pagination_step,
    max_pages,
    source,
    date_separator="-"
):
    articles = []
    session = requests.Session()  # Use a session to handle cookies
    for page in range(max_pages):
        print(f"Scraping {source} page {page} for '{keyword}'.")
        start_val = page * pagination_step
        url = url_template.format(
            keyword=keyword.replace(" ", "+"),
            start=start_val,
            year=START_YEAR
        )

        # Rotate User-Agent for each request
        headers = HEADERS.copy()
        headers["User-Agent"] = get_random_user_agent()

        try:
            r = session.get(url, headers=headers)
        except requests.exceptions.RequestException as e:
            print(f"Error accessing {source} for '{keyword}': {e}")
            break

        # Detect if captcha or block is encountered
        if r.status_code != 200 or "captcha" in r.text.lower() or "recaptcha" in r.text.lower():
            print(f"Captcha or block detected on {source} for '{keyword}', stopping.")
            break

        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select(item_selector)
        if not items:
            print(f"No items found on page {page} for {source}, stopping pagination.")
            break

        for item in items:
            title_tag = item.select_one(title_selector) if title_selector else None
            abstract_tag = item.select_one(abstract_selector) if abstract_selector else None
            date_tag = item.select_one(date_selector) if date_selector else None
            link_tag = item.select_one(link_selector) if link_selector else None

            title = title_tag.text.strip() if title_tag else "No title"
            abstract = abstract_tag.text.strip() if abstract_tag else "No abstract"
            link = link_tag.get("href").strip() if link_tag else ""
            date_text = date_tag.text.strip() if date_tag else ""
            parsed_date = parse_date(date_text, separator=date_separator)

            if keyword_in_text(keyword, title) or keyword_in_text(keyword, abstract):
                articles.append({
                    "title": title,
                    "abstract": abstract,
                    "date": parsed_date,
                    "source": source,
                    "doi_url": link
                })

        # Introduce a random delay to avoid bot detection
        time.sleep(SLEEP_SECONDS + random.uniform(0, 5))

    print(f"Found {len(articles)} articles from {source} for '{keyword}'.")
    return articles

def scrape_google_scholar(keyword):
    return scrape_site(
        keyword=keyword,
        # Use {year} inside the URL to inject the global START_YEAR
        url_template="https://scholar.google.com/scholar?start={start}&q={keyword}&as_ylo={year}",
        item_selector=".gs_r .gs_ri",
        title_selector=".gs_rt",
        abstract_selector=".gs_rs",
        date_selector=".gs_a",
        link_selector=".gs_rt a",
        pagination_step=10,
        max_pages=MAX_GS_PAGES,
        source="Google Scholar",
        date_separator="-"
    )

def scrape_arxiv(keyword):
    return scrape_site(
        keyword=keyword,
        url_template=(
            "https://arxiv.org/search/?query={keyword}"
            "&searchtype=all&abstracts=show&order=-announced_date_first&start={start}"
        ),
        item_selector=".arxiv-result",
        title_selector=".title",
        abstract_selector=".abstract",
        date_selector=".submitted",
        link_selector=".list-title a",
        pagination_step=50,
        max_pages=MAX_ARXIV_PAGES,
        source="arXiv",
        date_separator=";"
    )

def main():
    existing = load_existing_articles()
    all_articles = existing.copy()

    for theme, kws in search_keywords.items():
        for kw in kws:
            try:
                gs_items = scrape_google_scholar(kw)
                for a in gs_items:
                    a = clean_article(a)
                    if not is_duplicate(a, all_articles):
                        scores = compute_theme_scores(a["title"], a["abstract"])
                        if sum(scores.values()) == 0:
                            continue
                        a.update(scores)
                        a["dominant_theme"] = determine_dominant_theme(scores)
                        all_articles.append(a)
            except Exception as e:
                print(f"Error on Google Scholar for '{kw}': {e}")

            try:
                ax_items = scrape_arxiv(kw)
                for a in ax_items:
                    a = clean_article(a)
                    if not is_duplicate(a, all_articles):
                        scores = compute_theme_scores(a["title"], a["abstract"])
                        if sum(scores.values()) == 0:
                            continue
                        a.update(scores)
                        a["dominant_theme"] = determine_dominant_theme(scores)
                        all_articles.append(a)
            except Exception as e:
                print(f"Error on arXiv for '{kw}': {e}")

    save_articles_to_csv(all_articles)
    print(f"{len(all_articles)} articles saved in '{CSV_FILE}'.")

if __name__ == "__main__":
    main()

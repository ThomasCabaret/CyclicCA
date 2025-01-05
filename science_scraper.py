import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
import os
import csv
import re
import json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:114.0) Gecko/20100101 Firefox/114.0"
    )
}

# We keep the same search keywords
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
MAX_GS_PAGES = 10       # number of pages to fetch from Google Scholar
MAX_ARXIV_PAGES = 10    # number of pages to fetch from arXiv
MAX_PUBMED_PAGES = 3   # number of pages to fetch from PubMed
SLEEP_SECONDS = 20      # small delay between requests

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

def scrape_google_scholar(keyword):
    articles = []
    for page in range(MAX_GS_PAGES):
        print(f"Scraping Google Scholar page {page} for '{keyword}'.")
        start_val = page * 10
        url = (
            f"https://scholar.google.com/scholar?start={start_val}"
            f"&q={keyword.replace(' ', '+')}&as_ylo={START_YEAR}"
        )
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select('[data-lid]')
        if not items:
            # no more results
            break
        for item in items:
            title_tag = item.select_one(".gs_rt")
            abstract_tag = item.select_one(".gs_rs")
            date_tag = item.select_one(".gs_a")
            link = ""
            if title_tag and title_tag.select_one("a"):
                link = title_tag.select_one("a").get("href", "")
            title = title_tag.text if title_tag else "No title"
            abstract = abstract_tag.text if abstract_tag else "No abstract"
            date = ""
            if date_tag:
                parts = date_tag.text.split("-")
                date = parts[-1].strip() if parts else ""
            articles.append({
                "title": title.strip(),
                "abstract": abstract.strip(),
                "date": date,
                "source": "Google Scholar",
                "doi_url": link.strip()
            })
        time.sleep(SLEEP_SECONDS)
    print(f"Found {len(articles)} from Google Scholar for '{keyword}'.")
    return articles

def scrape_arxiv(keyword):
    articles = []
    for page in range(MAX_ARXIV_PAGES):
        print(f"Scraping arXiv page {page} for '{keyword}'.")
        start_val = page * 50
        url = (
            f"https://arxiv.org/search/?query={keyword.replace(' ', '+')}"
            f"&searchtype=all&abstracts=show&order=-announced_date_first"
            f"&date-filter_by=last_update_date&date-from_date={START_DATE}"
            f"&date-to_date={END_DATE}"
            f"&start={start_val}"
        )
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select(".arxiv-result")
        if not items:
            break
        for item in items:
            title_tag = item.select_one(".title")
            abstract_tag = item.select_one(".abstract")
            date_tag = item.select_one(".submitted")
            link_tag = item.select_one(".list-title a")
            link = link_tag.get("href") if link_tag else ""
            title = title_tag.text.strip() if title_tag else "No title"
            abstract = abstract_tag.text.strip() if abstract_tag else "No abstract"
            date = ""
            if date_tag:
                parts = date_tag.text.split(";")
                date = parts[-1].strip() if parts else ""
            articles.append({
                "title": title,
                "abstract": abstract,
                "date": date,
                "source": "arXiv",
                "doi_url": link.strip()
            })
        time.sleep(SLEEP_SECONDS)
    print(f"Found {len(articles)} from arXiv for '{keyword}'.")
    return articles

# def scrape_pubmed(keyword):
#     articles = []
#     for page in range(1, MAX_PUBMED_PAGES + 1):
#         url = (
#             f"https://pubmed.ncbi.nlm.nih.gov/?term={keyword.replace(' ', '+')}"
#             f"&filter=years.{START_YEAR}-&page={page}"
#         )
#         r = requests.get(url, headers=HEADERS)
#         soup = BeautifulSoup(r.text, "html.parser")
#         items = soup.select(".docsum-content")
#         if not items:
#             break
#         for item in items:
#             title_tag = item.select_one(".docsum-title")
#             link_tag = title_tag.get("href") if title_tag else ""
#             link = f"https://pubmed.ncbi.nlm.nih.gov{link_tag}" if link_tag else ""
#             date_tag = item.select_one(".docsum-journal-citation.full-journal-citation")
#             date = date_tag.text.strip() if date_tag else ""
#             title = title_tag.text.strip() if title_tag else "No title"
#             abstract = "No abstract"
#             articles.append({
#                 "title": title,
#                 "abstract": abstract,
#                 "date": date,
#                 "source": "PubMed",
#                 "doi_url": link.strip()
#             })
#         time.sleep(SLEEP_SECONDS)
#     print(f"Found {len(articles)} from PubMed for '{keyword}'.")
#     return articles

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

#             try:
#                 pm_items = scrape_pubmed(kw)
#                 for a in pm_items:
#                     a = clean_article(a)
#                     if not is_duplicate(a, all_articles):
#                         scores = compute_theme_scores(a["title"], a["abstract"])
#                         if sum(scores.values()) == 0:
#                             continue
#                         a.update(scores)
#                         a["dominant_theme"] = determine_dominant_theme(scores)
#                         all_articles.append(a)
#             except Exception as e:
#                 print(f"Error on PubMed for '{kw}': {e}")

    save_articles_to_csv(all_articles)
    print(f"{len(all_articles)} articles saved in '{CSV_FILE}'.")

if __name__ == "__main__":
    main()

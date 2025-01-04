import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import csv
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    )
}

# Ensure all search keywords are included in scoring keywords
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

scoring_keywords = {
    "origin_of_life": [
        "origin of life",
        "abiogenesis",
        "chemical evolution",
        "prebiotic",
        "pre-biotic",
        "rna world",
        "protocell",
        "early life",
        "primitive cell",
        "autopoeisis",
        "autopoeitic",
        "metabolism-first",
        "metabolism first",
        "LUCA",
        "vesicle",
        "emergence",
        "formation"
    ],
    "synthetic_biology": [
        "synthetic biology",
        "bottom-up",
        "artificial cell",
        "synthetic cell",
        "autopoiesis",
        "autopoietic",
        "self-replication",
        "self-replicating",
        "self-assembly",
        "self-organization",
        "homeostasis",
        "metabolism",
        "proto-metabolism",
        "energy transfer",
        "energy conversion",
        "atp production",
        "chemiosmosis",
        "membrane transport",
        "ion channel",
        "vesicle",
        "compartmentalization",
        "molecular network",
        "gene circuit",
        "genetic network",
        "feedback loop",
        "signaling pathway",
        "information processing",
        "minimal cell",
        "protocell",
        "liposome",
        "vesicle fusion",
        "protein synthesis",
        "rna replication",
        "dna synthesis",
        "protein folding",
        "enzyme",
        "reaction network",
        "biosynthesis",
        "biochemical pathway",
        "chemical communication",
        "reproduction",
        "division cycle",
        "cell cycle",
        "growth",
        "evolution",
        "adaptation"
    ],
}

# Validate that all search keywords are in scoring keywords
for theme, keywords in search_keywords.items():
    for keyword in keywords:
        if keyword not in scoring_keywords[theme]:
            raise ValueError(f"Search keyword '{keyword}' is missing from scoring keywords for theme '{theme}'.")

CSV_FILE = "articles.csv"
START_YEAR = 2010
START_DATE = "2010-01-01"
END_DATE = "2100-01-01"

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
    max_score = max(scores.values()) if scores else 0
    if max_score == 0:
        return ""
    top_themes = [t for t, s in scores.items() if s == max_score]
    if len(top_themes) == 1:
        return top_themes[0]
    return "multiple"

def scrape_google_scholar(keyword):
    url = f"https://scholar.google.com/scholar?q={keyword.replace(' ', '+')}&as_ylo={START_YEAR}"
    print(f"Scraping Google Scholar for '{keyword}'...")
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    articles = []
    for item in soup.select('[data-lid]'):
        title_tag = item.select_one(".gs_rt")
        abstract_tag = item.select_one(".gs_rs")
        date_tag = item.select_one(".gs_a")
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
            "source": "Google Scholar"
        })
    print(f"Found {len(articles)} articles from Google Scholar for '{keyword}'.")
    return articles

def scrape_arxiv(keyword):
    print(f"Scraping arXiv for '{keyword}'...")
    url = (
        f"https://arxiv.org/search/?query={keyword.replace(' ', '+')}"
        "&searchtype=all&abstracts=show&order=-announced_date_first"
        f"&date-filter_by=last_update_date&date-from_date={START_DATE}"
        f"&date-to_date={END_DATE}"
    )
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    articles = []
    for item in soup.select(".arxiv-result"):
        title_tag = item.select_one(".title")
        abstract_tag = item.select_one(".abstract")
        date_tag = item.select_one(".submitted")
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
            "source": "arXiv"
        })
    print(f"Found {len(articles)} articles from arXiv for '{keyword}'.")
    return articles

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
                            continue  # Skip articles with zero scores
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
                            continue  # Skip articles with zero scores
                        a.update(scores)
                        a["dominant_theme"] = determine_dominant_theme(scores)
                        all_articles.append(a)
            except Exception as e:
                print(f"Error on arXiv for '{kw}': {e}")

    save_articles_to_csv(all_articles)
    print(f"{len(all_articles)} articles saved in '{CSV_FILE}'.")

if __name__ == "__main__":
    main()

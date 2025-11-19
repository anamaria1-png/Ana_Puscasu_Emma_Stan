import requests
import time
import os
import random
from statistics import median
from collections import Counter
import pandas as pd


UA = {"User-Agent": "Mozilla/5.0"}

# List of genres to iterate over in search.json
QUERIES = [
    "fiction", "novel", "classics", "literature", "fantasy",
    "science fiction", "mystery", "thriller", "horror",
    "romance", "young adult", "historical fiction", "nonfiction"
]


def safe_get_json(url, timeout=15):
    """
    Small wrapper around requests.get that returns JSON or {} if there is an error (with the request).
    """
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code != 200:
            return {}
        return r.json() or {}
    except Exception:
        return {}


def get_ratings(work_key, timeout=15):
    """
    Read /ratings.json for a given work and return (count, average).
    If ratings are missing or any error occurs -> (None, None).

    Features:
    - ratings_count: proxy for popularity / sample size.
    - ratings_average: target for regression / ranking models.

    Source: https://openlibrary.org/works/<key>/ratings.json
    """
    url = f"https://openlibrary.org{work_key}/ratings.json"
    data = safe_get_json(url, timeout=timeout)
    summary = data.get("summary") or {}
    cnt = summary.get("count")
    avg = summary.get("average")
    cnt = int(cnt) if isinstance(cnt, (int, float)) else None
    avg = float(avg) if isinstance(avg, (int, float)) else None
    return cnt, avg


def parse_languages_from_work(work_json):
    """
    Extract language codes from the 'languages' field in a work JSON.

    Example:
        "languages": [{"key": "/languages/eng"}, {"key": "/languages/fre"}]
    ->  ["eng", "fre"]

    Feature 'language' can be used as:
    - categorical input;
    - basis for one-hot encoding;
    - proxy for market / audience.
    """
    langs = work_json.get("languages", []) or []
    out = []
    for it in langs:
        if isinstance(it, dict):
            k = it.get("key", "")
            if isinstance(k, str) and k.startswith("/languages/"):
                out.append(k.split("/")[-1])
    return out


def extract_primary_author_key(search_doc, work_json):
    """
    Try to extract the primary author key for a work.

    Priority:
    1) work_json['authors'][0]['author']['key']   (e.g. '/authors/OL23919A')
    2) search_doc['author_key'][0]                (e.g. 'OL23919A')

    This author key will be used to query /authors/<id>.json
    in order to obtain 'author_work_count' (how many works the
    author has in OpenLibrary).
    """
    # 1) from work JSON
    authors = work_json.get("authors") or []
    if authors:
        a0 = authors[0]
        if isinstance(a0, dict):
            aobj = a0.get("author") or {}
            key = aobj.get("key")
            if isinstance(key, str) and key:
                return key  # already like '/authors/OLxxxA'

    # 2) fallback from search.json
    keys = search_doc.get("author_key") or []
    if isinstance(keys, list) and keys:
        k0 = keys[0]
        if isinstance(k0, str):
            # build "/authors/<id>" if needed
            if not k0.startswith("/authors/"):
                return f"/authors/{k0}"
            return k0

    return None


def editions_snapshot(work_key, limit=200, timeout=20):
    """
    Fetch a snapshot of editions for a given work.

    Each entry in 'entries' is an edition with fields like:
    - publish_date / publish_year
    - publishers / publish_places
    - number_of_pages
    - languages, etc.

    Source: https://openlibrary.org/works/<key>/editions.json
    """
    url = f"https://openlibrary.org{work_key}/editions.json?limit={limit}"
    data = safe_get_json(url, timeout=timeout)
    return data.get("entries") or []


def infer_from_editions_for_first_year(work_key, first_publish_year, limit=200, timeout=20):
    """
    Use editions.json to infer several features:

    - number_of_pages_median: median 'number_of_pages' over all editions
      (this helps when the work JSON does not provide it).
    - series_name: most frequent textual 'series' label among editions.
    - publisher: publisher of an edition that matches first_publish_year
    - publish_place: place of an edition that matches first_publish_year
    - languages_ed: languages that appear in editions (fallback for work).

    Observations:
    - Publisher and publish_place are completed using ONLY from editions whose
      year == first_publish_year (the first publication year of the work
      according to OpenLibrary).
    - This means we approximate: "publisher / place of the oldest edition
      *recorded in OpenLibrary for that year*".
    - It is NOT guaranteed to be the true historical first edition.
    - If no edition matches first_publish_year, publisher and publish_place
      are left as None, so in the final DataFrame they will appear as NaN. 
    """
    entries = editions_snapshot(work_key, limit=limit, timeout=timeout)

    pages = []
    series_labels = []
    langs = set()

    # Candidates for publisher/place (only for editions matching first_publish_year)
    candidate_pubs = []
    candidate_places = []

    for e in entries:
        #   number_of_pages for median estimation
        nop = e.get("number_of_pages")
        if isinstance(nop, (int, float)) and nop > 0:
            pages.append(int(nop))

        # series labels
        s = e.get("series")
        if isinstance(s, str) and s.strip():
            series_labels.append(s.strip())
        elif isinstance(s, list):
            for it in s:
                if isinstance(it, str) and it.strip():
                    series_labels.append(it.strip())

        #  determine year of this edition
        year = None
        py = e.get("publish_year")
        if isinstance(py, list) and py:
            ints = [y for y in py if isinstance(y, int)]
            if ints:
                year = min(ints)

        if year is None:
            pdate = e.get("publish_date")
            if isinstance(pdate, str):
                # look for a 4-digit year pattern
                import re
                m = re.search(r"(\d{4})", pdate)
                if m:
                    year = int(m.group(1))

        # parse languages from the edition
        for it in (e.get("languages") or []):
            if isinstance(it, dict):
                k = it.get("key", "")
                if isinstance(k, str) and k.startswith("/languages/"):
                    langs.add(k.split("/")[-1])

        # if this edition matches first_publish_year, collect publisher/place
        if isinstance(first_publish_year, int) and year == first_publish_year:
            # publisher
            pubs = e.get("publishers")
            if isinstance(pubs, list) and pubs:
                p0 = pubs[0]
                if isinstance(p0, str) and p0.strip():
                    candidate_pubs.append(p0.strip())

            # publish_place
            place = None
            pl = e.get("publish_places")
            if isinstance(pl, list) and pl:
                fp = pl[0]
                if isinstance(fp, str) and fp.strip():
                    place = fp.strip()

            if place is None:
                pl1 = e.get("publish_place")
                if isinstance(pl1, list) and pl1:
                    fp = pl1[0]
                    if isinstance(fp, str) and fp.strip():
                        place = fp.strip()
                elif isinstance(pl1, str) and pl1.strip():
                    place = pl1.strip()

            if place is None:
                pc = e.get("publish_country")
                if isinstance(pc, str) and pc.strip():
                    place = pc.strip()

            if place:
                candidate_places.append(place)

    # aggregate pages & series over ALL editions
    pages_median = int(median(pages)) if pages else None
    series_name = Counter(series_labels).most_common(1)[0][0] if series_labels else None

    # aggregate publisher / place over editions in first_publish_year only
    publisher = Counter(candidate_pubs).most_common(1)[0][0] if candidate_pubs else None
    publish_place = Counter(candidate_places).most_common(1)[0][0] if candidate_places else None

    return pages_median, series_name, publisher, publish_place, list(langs)


def get_work_details(work_key):
    """
    Fetch /works/<key>.json and extract several high-level features:

    - subjects: topical tags (themes, genres, etc.)
    - subject_people: people / characters mentioned (e.g. 'Dracula', 'Napoleon')
    - subject_places: locations (e.g. 'London', 'Transylvania')
    - subject_times: periods (e.g. '19th century')
      -> These four are very useful for NLP, topic modelling, and as
         categorical features (one-hot, bag-of-words).

    - languages: language codes of the work
    - number_of_pages_median: pages median provided at work level
    - series_name: any series name present at work level

    Source: https://openlibrary.org/works/<key>.json
    """
    url = f"https://openlibrary.org{work_key}.json"
    wj = safe_get_json(url, timeout=20)

    subjects = wj.get("subjects", []) or []
    subject_people = wj.get("subject_people", []) or []
    subject_places = wj.get("subject_places", []) or []
    subject_times = wj.get("subject_times", []) or []

    languages = parse_languages_from_work(wj)
    num_pages_median = wj.get("number_of_pages_median")

    series_name = None
    s = wj.get("series")
    if isinstance(s, str) and s.strip():
        series_name = s.strip()
    elif isinstance(s, list) and s:
        series_name = str(s[0]).strip()

    return {
        "subjects": subjects,
        "subject_people": subject_people,
        "subject_places": subject_places,
        "subject_times": subject_times,
        "languages": languages,
        "pages_median": num_pages_median,
        "series_name": series_name,
        "work_json": wj,   # keep for author extraction
    }


def harvest_openlibrary(
    target_books=1000,
    min_ratings=5,
    max_skips_without_rating=50,
    pause=0.2,
    max_pages_per_query=20,
    use_editions_fallback=True,
    output_csv=True
):
    """
    Main harvesting function.

    Parameters
    ----------
    target_books : int
        How many books (rows) we want to collect in total across all queries.
        This directly controls the dataset size.

    min_ratings : int
        Minimum number of ratings required for a work to be kept.
        This ensures that ratings_average is based on sufficient votes.

    max_skips_without_rating : int
        If we scan this many works in a row (within a query) that do not
        meet the ratings filter, we break early and move to the next query.
        Using 50 is a good balance: enough to find popular books, but not
        wasting time on many obscure records.

    pause : float
        Sleep time (seconds) between HTTP requests to be polite with the API.

    max_pages_per_query : int
        Maximum number of pages to scan in search.json for each textual query.

    use_editions_fallback : bool
        If True, use editions.json to fill missing:
        - pages_median, series_name, languages
        - publisher & publish_place (ONLY if we find editions in first_publish_year)

    output_csv : bool
        If True, save the final DataFrame to 'openlibrary_<target_books>.csv'.

    Returns
    -------
    df : pandas.DataFrame
        Final dataset with all features for the selected books.
    """
    rows = []
    seen = set()
    author_cache = {}  # for author_work_count memoization

    for q in QUERIES:
        if len(rows) >= target_books:
            break

        print(f"\n🔎 Query: {q}")
        page = 1
        consecutive_skips = 0

        while len(rows) < target_books and page <= max_pages_per_query:
            url = f"https://openlibrary.org/search.json?q={q}&limit=100&page={page}"
            data = safe_get_json(url, timeout=25)
            docs = data.get("docs") or []
            if not docs:
                break

            for d in docs:
                if len(rows) >= target_books:
                    break

                work_key = d.get("key")  # e.g. "/works/OL45883W"
                if not work_key or work_key in seen:
                    continue

                # 1) Ratings filter (main filter) 
                cnt, avg = get_ratings(work_key)
                if (cnt is None) or (cnt < min_ratings):
                    consecutive_skips += 1
                    if consecutive_skips >= max_skips_without_rating:
                        # stop this query, go to next genre
                        print(f"  ⏭  Reached {max_skips_without_rating} low-rated works in a row, skipping to next query.")
                        break
                    time.sleep(pause)
                    continue

                # we found a valid rated work
                consecutive_skips = 0

                #  2) Work-level details 
                w_details = get_work_details(work_key)
                subjects = w_details["subjects"]
                subject_people = w_details["subject_people"]
                subject_places = w_details["subject_places"]
                subject_times = w_details["subject_times"]
                languages = w_details["languages"]
                num_pages_median = w_details["pages_median"]
                series_name = w_details["series_name"]
                work_json = w_details["work_json"]

                # 3) Fallback languages from search.json if missing 
                if not languages:
                    langs_from_search = d.get("language", []) or []
                    if isinstance(langs_from_search, list):
                        languages = [str(x) for x in langs_from_search if isinstance(x, (str, int, float))]

                #  4) Editions fallback (pages, series, publisher, place, languages) 
                publisher = None
                publish_place = None

                first_year = d.get("first_publish_year")
                if use_editions_fallback:
                    p_med, s_name, pub, p_place, langs_ed = infer_from_editions_for_first_year(
                        work_key,
                        first_publish_year=first_year
                    )
                    # pages median
                    if num_pages_median is None:
                        num_pages_median = p_med
                    # series
                    if series_name is None:
                        series_name = s_name
                    # publisher / publish_place:
                    publisher = pub
                    publish_place = p_place
                    # languages fallback
                    if not languages and langs_ed:
                        languages = langs_ed

                #  5) Author features (author_work_count) 
                author_key = extract_primary_author_key(d, work_json)

                #  6) Final row assembly 
                rows.append({
                    # Basic identification / core bibliographic fields
                    "title": d.get("title", ""),
                    "author": ", ".join(d.get("author_name", []) or []),

                    # First publication year (from search.json)
                    "first_publish_year": first_year,

                    # Edition count from search.json:
                    # proxy for how many physical/recorded editions a work has.
                    "edition_count": d.get("edition_count"),

                    # Topic-related textual features
                    "subject": ", ".join(subjects),
                    "subject_people": ", ".join(subject_people),
                    "subject_places": ", ".join(subject_places),
                    "subject_times": ", ".join(subject_times),

                    # Language information (work + editions fallback)
                    "language": ", ".join(languages) if languages else None,

                    # Series information (work + editions fallback)
                    "series": series_name,

                    # Pages median: estimate of book length,
                    # useful for correlating length with ratings/popularity.
                    "number_of_pages_median": num_pages_median,

                    # Publisher and publishing place:
                    # these refer to an edition from first_publish_year as recorded in OpenLibrary, NOT necessarily the real
                    # historical first edition of the work.
                    # If no edition matches first_publish_year, these stay NaN.
                    "publisher": publisher,
                    "publish_place": publish_place,

                    # Ratings features (popularity / quality indicators)
                    "ratings_count": cnt,
                    "ratings_average": avg,
                })
                seen.add(work_key)

                if len(rows) % 20 == 0:
                    last = rows[-1]
                    print(f"{len(rows)} — '{last['title']}' "
                          f"({last['ratings_average']}, n={last['ratings_count']}, "
                          f"lang={last['language']}, series={last['series']})")

                time.sleep(pause + random.uniform(0, 0.1))

            if consecutive_skips >= max_skips_without_rating:
                break

            page += 1
            time.sleep(pause)

    df = pd.DataFrame(rows)

    if output_csv:
        out_path = os.path.join(os.getcwd(), f"openlibrary_{len(df)}_v2.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nFinished: {len(df)} rows x {df.shape[1]} columns | saved to: {out_path}")

    return df


if __name__ == "__main__":
    # Example: collect 3000 books, at least 5 ratings each,
    # skip to next genre after 50 consecutive low-rated works.
    df_books = harvest_openlibrary(
        target_books=3000,
        min_ratings=5,
        max_skips_without_rating=50
    )

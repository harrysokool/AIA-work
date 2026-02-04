from bs4 import BeautifulSoup
import threading
import time
import requests
import pickle
import random


class RetryableFetchError(Exception):
    """Signals a transient error that may succeed on retry."""


physio_name = {"name_repeated_count": 0}
BASE_URL = "https://www.smp-council.org.hk/hkifd/browse.php"
MAX_RETRIES = 5


def fetch_page(session: requests.Session, page: int):
    params = {"serach": f"PT{str(page)}"}
    resp = session.get(BASE_URL, params=params)
    if resp.status_code != 200:
        raise RetryableFetchError("Error when accessing website")
    return resp.text


def fetch_page_with_retries(
    session: requests.Session, page: int, retries: int = MAX_RETRIES
):
    for attempt in range(1, retries + 1):
        try:
            return fetch_page(session, page)
        except RetryableFetchError as e:
            err = e
        except Exception as e:
            err = RetryableFetchError(f"Unexpected error: {e}")

        print(f"[Attempt {attempt}] {err}")

        if attempt < retries:
            wait_time = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        else:
            print(f"Failed after {retries} attempts")
            return None


def find_tables(soup: BeautifulSoup):
    tables = soup.find_all("table")
    if not tables:
        return None
    return tables[2:]


def add_physio() -> None:
    pass


def save_physio() -> None:
    with open("physio.pkl", "wb") as f:
        pickle.dump(physio_name, f)
    with open("physio.txt", "w", encoding="utf-8") as f:
        for name in physio_name:
            f.write(name + "\n")


def load_physio_set() -> set[str]:
    with open("physio.pkl", "rb") as f:
        return pickle.load(f)


def fetch_physio():
    pass


def main():
    pass


if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()

    print(f"Time took: {toc - tic:0.4f}s")


# 1. we need to fetch the page, may be add some retries
# 2. now search all the table with beautiful soup
# 3. after find all the names, we don't even need to do anything to the names, we can just add it into the set
# 4. store it somewhere,

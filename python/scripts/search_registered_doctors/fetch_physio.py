from bs4 import BeautifulSoup
import threading
import time
import requests
import pickle
import random


class RetryableFetchError(Exception):
    """Signals a transient error that may succeed on retry."""


physio_name: dict[int] = {"name_repeated_count": 0}
BASE_URL = "https://www.smp-council.org.hk/hkifd/browse.php"
MAX_RETRIES = 5


def fetch_page(session: requests.Session) -> str | None:
    resp = session.get(BASE_URL, timeout=30)
    if resp.status_code != 200:
        raise RetryableFetchError("Error when accessing website")
    return resp.text


def fetch_page_with_retries(
    session: requests.Session, retries: int = MAX_RETRIES
) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            return fetch_page(session)

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
    if not tables or len(tables) < 2:
        return None
    return tables[2:]


def add_physio(table) -> None:
    if table is None:
        return

    rows = table.find_all("tr")
    if not rows:
        return

    for row in rows:
        tds = row.find_all("td")
        if len(tds) != 3:
            continue

        reg_no = tds[0].get_text(strip=True)
        eng_name = tds[1].get_text(strip=True)
        chi_name = tds[2].get_text(strip=True)

        if eng_name in physio_name:
            physio_name["name_repeated_count"] += 1

        physio_name[eng_name] = physio_name.get(eng_name, 0) + 1
        physio_name[chi_name] = physio_name.get(chi_name, 0) + 1

        print(reg_no, eng_name, chi_name)


def save_physio() -> None:
    with open("physio.pkl", "wb") as f:
        pickle.dump(physio_name, f)
    with open("physio.txt", "w", encoding="utf-8") as f:
        for name in physio_name:
            f.write(f"{name}\n")


def load_physio_set() -> dict[int]:
    with open("physio.pkl", "rb") as f:
        return pickle.load(f)


def fetch_physio():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.smp-council.org.hk/pt/en/content.php?page=reg_reg",
        }
    )

    # while True:
    html: str | None = fetch_page_with_retries(session, MAX_RETRIES)
    if html is None:
        return

    soup = BeautifulSoup(html, "lxml")
    tables = find_tables(soup)
    if tables is None:
        return

    for table in tables:
        add_physio(table)

    return


def main():
    # this will populate physio_name
    fetch_physio()

    print(physio_name["name_repeated_count"])
    # save the physio name locally
    save_physio()


if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()

    print(f"Time took: {toc - tic:0.4f}s")

from bs4 import BeautifulSoup, Tag
import threading
import time
import asyncio
import aiohttp
from typing import Optional, List, Set
import pickle
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RetryableFetchError(Exception):
    """Signals a transient error that may succeed on retry."""


doctors_name: Set[str] = set()
ALPHABET_SET: Set[str] = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CONCURRECY_LIMIT = 10
MAX_RETRIES = 5
SEM = asyncio.Semaphore(CONCURRECY_LIMIT)

OUTPUT_FILE = "doctors.pkl"
BASE_URL = "https://www.mchk.org.hk/english/list_register/list.php"
DOCTOR_TYPE = ["O", "P", "N", "M"]


# helper functions
def load_doctors_set() -> Set[str]:
    """
    Load the scraped registered doctors from doctors.pkl.
    """
    with open(OUTPUT_FILE, "rb") as f:
        return pickle.load(f)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "AuditScraper/1.0"})
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def timer() -> None:
    counter = 1
    while True:
        time.sleep(1)
        print(counter)
        counter += 1


def save_doctors() -> None:
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(doctors_name, f)
    with open("doctors.txt", "w", encoding="utf-8") as f:
        for name in doctors_name:
            f.write(name + "\n")


def expo_probing(session) -> int:
    page = 1
    while True:
        params = {"page": str(page), "ipp": "20", "type": "L"}
        try:
            resp = session.get(BASE_URL, params=params, timeout=15)
            if resp.status_code != 200:
                return page
        except requests.RequestException:
            return None

        soup: BeautifulSoup = BeautifulSoup(resp.text, "lxml")
        table: Optional[Tag] = find_table(soup)
        if table is None:
            return page

        rows: List[Tag] = table.find_all("tr")[2:]
        if not rows:
            return page
        if rows and "沒有相關搜尋結果" in str(rows[0]):
            return page

        if page > 1_000_000:
            return None

        page *= 2


def valid_page(page: int, session) -> bool:
    if page < 1:
        return False

    params = {"page": str(page), "ipp": "20", "type": "L"}

    try:
        response = session.get(BASE_URL, params=params, timeout=5)
        if response.status_code != 200:
            return False

        html = response.text
        soup = BeautifulSoup(html, "lxml")

        table: Optional[Tag] = find_table(soup)
        if table is None:
            return False

        rows: List[Tag] = table.find_all("tr")[2:]
        if not rows:
            return False
        first_text = rows[0].get_text(strip=True)
        if "沒有相關搜尋結果" in first_text:
            return False

        return True
    except requests.RequestException:
        return False


def search_last_page() -> int:
    session = make_session()
    start_page = 1
    end_page = expo_probing(session)
    if end_page is None:
        return None

    while start_page < end_page:
        mid = start_page + (end_page - start_page) // 2
        if valid_page(mid, session):
            start_page = mid + 1
        else:
            end_page = mid - 1

    return start_page


async def fetch_page(
    session: aiohttp.ClientSession, doctor_type: str, page: int
) -> str:
    params = {"page": str(page), "ipp": "20", "type": doctor_type}
    async with session.get(BASE_URL, params=params, ssl=False) as resp:
        if resp.status != 200:
            raise RetryableFetchError(f"HTTP {resp.status}")
        return await resp.text()


async def fetch_page_with_retry(
    session: aiohttp.ClientSession,
    doctor_type: str,
    page: int,
    retries: int = MAX_RETRIES,
) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await fetch_page(session, doctor_type, page)

        except aiohttp.ClientError as e:
            err = RetryableFetchError(f"Network error: {e}")
        except RetryableFetchError as e:
            err = e
        except Exception as e:
            err = RetryableFetchError(f"Unexpected error: {e}")

        print(f"[Attempt {attempt}] {err}")

        if attempt < retries:
            wait_time = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        else:
            print(f"Failed after {retries} attempts")
            return None


def find_table(soup: BeautifulSoup) -> Optional[Tag]:
    tables = soup.find_all("table")
    if not tables or len(tables) < 4:
        return None
    table = tables[3]
    return table


def add_doctors(rows: List[Tag]) -> bool:
    found_doctor = False

    for row in rows:
        tds = row.find_all("td")
        if tds is None:
            found_doctor = False
            break

        found_doctor = True

        name_td = tds[1]
        lines = name_td.get_text("\n", strip=True).split("\n")

        for name in lines:
            if name and name[0] in ALPHABET_SET:
                res_name = name.replace(",", "").upper()
                if res_name not in doctors_name:
                    doctors_name.add(res_name)
                else:
                    print(f"{res_name} already in the list")

    return found_doctor


async def worker(
    doctor_type: str, session: aiohttp.ClientSession, queue: asyncio.Queue[int]
) -> None:

    while True:
        page: int = await queue.get()
        found_doctor = True
        try:
            # here need to fetch the page
            html: str = await fetch_page(session, doctor_type, page)
            soup: BeautifulSoup = BeautifulSoup(html, "lxml")

            # try to find the target table on the website
            table: Optional[Tag] = find_table(soup)
            if table is None:
                queue.task_done()
                break

            # skipping the headers
            rows: List[Tag] = table.find_all("tr")[2:]
            if not rows:
                queue.task_done()
                found_doctor = False
                break

            # flag to see if we should continue search for the next page
            found_doctor: bool = add_doctors(rows)

            if found_doctor:
                await queue.put(page + 1)
            else:
                queue.task_done()
                break

            queue.task_done()
        except Exception as e:
            print(f"Error on page {page}: {e}")
            queue.task_done()
            break


async def fetch_doctors(doctor_type: str) -> None:
    queue: asyncio.Queue[int] = asyncio.Queue()
    await queue.put(1)

    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
    connector = aiohttp.TCPConnector(limit=CONCURRECY_LIMIT, force_close=False)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        # spawn workers to process pages concurrently
        workers: List[asyncio.Task] = [
            asyncio.create_task(worker(doctor_type, session, queue))
            for _ in range(CONCURRECY_LIMIT)
        ]

        # wait until scraping is done
        await queue.join()

        # cancel the workers because workers run forever in a while True loop
        for w in workers:
            w.cancel()


async def fetch_type_L(session: aiohttp.ClientSession, param: dict):
    async with session.get(BASE_URL, params=param, ssl=False) as response:
        if response.status != 200:
            raise RetryableFetchError(f"HTTP {response.status}")

        html: str = await response.text()
        soup: BeautifulSoup = BeautifulSoup(html, "lxml")

        table: Optional[Tag] = find_table(soup)
        if table is None:
            return

        rows: List[Tag] = table.find_all("tr")[2:]
        if not rows:
            return

        add_doctors(rows)


async def fetch_type_L_retry(
    session: aiohttp.ClientSession, param: dict, retries: int = MAX_RETRIES
):
    for attempt in range(1, retries + 1):
        try:
            async with SEM:
                return await fetch_type_L(session, param)

        except aiohttp.ClientError as e:
            err = RetryableFetchError(f"Network error: {e}")
        except RetryableFetchError as e:
            err = e
        except Exception as e:
            err = RetryableFetchError(f"Unexpected error: {e}")

        print(f"[Attempt {attempt}] {err}")

        if attempt < retries:
            wait_time = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        else:
            print(f"Failed after {retries} attempts for {param}")
            return None


async def fetch_doctors_type_L(params: list[dict]):
    if not params:
        return

    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
    connector = aiohttp.TCPConnector(limit=CONCURRECY_LIMIT, force_close=False)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch_type_L_retry(session, param) for param in params]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            print(f"Task failed: {res}")
        elif res:
            continue


async def main() -> Set[str]:
    # for each doctor type, we scrape them separately and concurrently
    # this is only for doctor type "O", "P", "N", "M", will do separetly with type "L"
    tasks = [fetch_doctors(doc_type) for doc_type in DOCTOR_TYPE]
    await asyncio.gather(*tasks)

    # first get all the params so we can do async all together for doctor type "L"
    last_page = search_last_page()
    if not last_page:
        raise RuntimeError("Could not determine last page for type L")

    params = [{"page": str(p), "ipp": "20", "type": "L"} for p in range(1, last_page)]
    await fetch_doctors_type_L(params)

    print(len(doctors_name))
    # save_doctors()


if __name__ == "__main__":
    # timer for the program
    threading.Thread(target=timer, daemon=True).start()

    tic = time.perf_counter()
    asyncio.run(main())
    toc = time.perf_counter()

    print(f"Time took: {toc - tic:0.4f}s")


# L: 16466, 15974, I think for this doctor type, some names are repeated, that's why numbers don't match
# O: 438, 438
# P: 555, 555
# M: 367, 367
# N: 155, 155

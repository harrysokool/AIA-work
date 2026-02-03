from bs4 import BeautifulSoup, Tag
import threading
import time
import asyncio
import aiohttp
from typing import Optional, List, Set
import pickle
import random


class RetryableFetchError(Exception):
    """Signals a transient error that may succeed on retry."""


doctors_name: Set[str] = set()
set_lock = asyncio.Lock()
ALPHABET_SET: Set[str] = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CONCURRENCY_LIMIT = 20
SEM = asyncio.Semaphore(CONCURRENCY_LIMIT)
MAX_RETRIES = 5
REQUEST_PAUSE = (0.2, 0.6)

OUTPUT_FILE = "doctors.pkl"
BASE_URL = "https://www.mchk.org.hk/english/list_register/list.php"
DOCTOR_TYPE = ["L", "O", "P", "N", "M"]


# helper functions
def save_doctors() -> None:
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(doctors_name, f)
    with open("doctors.txt", "w", encoding="utf-8") as f:
        for name in doctors_name:
            f.write(name + "\n")


def load_doctors_set() -> Set[str]:
    """
    Load the scraped registered doctors from doctors.pkl.
    """
    with open(OUTPUT_FILE, "rb") as f:
        return pickle.load(f)


def timer() -> None:
    counter = 1
    while True:
        time.sleep(1)
        print(counter)
        counter += 1


# functions
async def fetch_page(
    session: aiohttp.ClientSession, doctor_type: str, page: int
) -> str:
    params = {"page": str(page), "ipp": "20", "type": doctor_type}
    await asyncio.sleep(random.uniform(*REQUEST_PAUSE))
    async with SEM:
        async with session.get(BASE_URL, params=params) as resp:
            if resp.status == 429:
                raise RetryableFetchError("HTTP 429 Too Many Requests")
            if 500 <= resp.status < 600:
                raise RetryableFetchError(f"HTTP {resp.status} Server Error")
            if resp.status != 200:
                # Non-retryable (likely)
                text = await resp.text()
                raise Exception(f"HTTP {resp.status}: {text[:200]}")
            return await resp.text()


async def fetch_page_with_retry(
    session: aiohttp.ClientSession,
    doctor_type: str,
    page: int,
    retries: int = MAX_RETRIES,
) -> str | None:
    for attempt in range(1, retries + 1):
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


async def add_doctors(rows: List[Tag]) -> None:
    for row in rows:
        tds = row.find_all("td")
        if not tds or len(tds) < 2:
            continue

        name_td = tds[1]
        lines = name_td.get_text("\n", strip=True).split("\n")
        for name in lines:
            if not name:
                continue
            if name[0] in ALPHABET_SET:
                res_name = name.replace(",", "").upper()
                if res_name not in doctors_name:
                    async with set_lock:
                        doctors_name.add(res_name)


async def worker(
    doctor_type: str, session: aiohttp.ClientSession, queue: asyncio.Queue[int]
) -> None:
    while True:
        page: int = await queue.get()
        try:
            # here need to fetch the page
            html: str | None = await fetch_page_with_retry(session, doctor_type, page)
            if not html:
                await asyncio.sleep(random.uniform(0.2, 0.6))
                continue

            # try to find the target table on the website
            soup = BeautifulSoup(html, "lxml")
            table = find_table(soup)
            if table is None:
                break

            rows = table.find_all("tr")[2:] if table else []
            if not rows or "沒有相關搜尋結果" in rows[0].get_text(strip=True):
                break

            await add_doctors(rows)

            await queue.put(page + CONCURRENCY_LIMIT)
        except Exception as e:
            print(f"Error on page {page}: {e}")
        finally:
            queue.task_done()


async def fetch_doctors(doctor_type: str) -> None:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for p in range(1, CONCURRENCY_LIMIT + 1):
        await queue.put(p)

    timeout = aiohttp.ClientTimeout(total=30, connect=15, sock_read=45)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)
    headers = {"User-Agent": "AuditScraper/1.0 (+you@example.com)"}
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        # spawn workers to process pages concurrently
        workers: List[asyncio.Task] = [
            asyncio.create_task(worker(doctor_type, session, queue))
            for _ in range(CONCURRENCY_LIMIT)
        ]

        # wait until scraping is done
        await queue.join()

        # cancel the workers because workers run forever in a while True loop
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


async def main() -> Set[str]:
    # for each doctor type, we scrape them separately and concurrently
    tasks = [fetch_doctors(doc_type) for doc_type in DOCTOR_TYPE]
    await asyncio.gather(*tasks)

    print(len(doctors_name))
    save_doctors()


if __name__ == "__main__":
    # timer for the program
    threading.Thread(target=timer, daemon=True).start()

    tic = time.perf_counter()
    asyncio.run(main())
    toc = time.perf_counter()

    print(f"Time took: {toc - tic:0.4f}s")

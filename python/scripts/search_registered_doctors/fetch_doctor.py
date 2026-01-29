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
ALPHABET_SET: Set[str] = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CONCURRECY_LIMIT = 1
MAX_RETRIES = 5
OUTPUT_FILE = "doctors.pkl"
BASE_URL = "https://www.mchk.org.hk/english/list_register/list.php"
BASE_PARAMS = {"page": "", "ipp": "20", "type": ""}
DOCTOR_TYPE = ["L", "O", "P", "N", "M"]


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
            wait_time = (2 ** (attempt - 1)) + random.unifrom(0, 0.5)
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


async def main() -> Set[str]:
    # for each doctor type, we scrape them separately and concurrently
    tasks = [fetch_doctors(doc_type) for doc_type in DOCTOR_TYPE]
    await asyncio.gather(*tasks)
    save_doctors()


def load_doctors_set() -> Set[str]:
    """
    Load the scraped registered doctors from doctors.pkl.
    """
    with open(OUTPUT_FILE, "rb") as f:
        return pickle.load(f)


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

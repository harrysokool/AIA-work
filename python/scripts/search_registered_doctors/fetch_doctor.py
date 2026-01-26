from bs4 import BeautifulSoup, Tag
import threading
import time
import asyncio
import aiohttp
from typing import Optional, List, Set
import pickle


doctors_name: Set[str] = set()
ALPHABET_SET: Set[str] = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CONCURRECY_LIMIT = 10
OUTPUT_FILE = "doctors.pkl"
BASE_URL = "https://www.mchk.org.hk/english/list_register/list.php"
BASE_PARAMS = {"page": "", "ipp": "20", "type": ""}
DOCTOR_TYPE = ["N", "M"]


def timer() -> None:
    counter = 1
    while True:
        time.sleep(1)
        print(counter)
        counter += 1


def save_doctors() -> None:
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(doctors_name, f)


async def fetch_page(
    session: aiohttp.ClientSession, doctor_type: str, page: int
) -> str:
    params = {"page": str(page), "ipp": "20", "type": doctor_type}
    async with session.get(BASE_URL, params=params) as resp:
        return await resp.text()


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
        if len(tds) == 2:  # no more doctors in the table
            found_doctor = False
            break

        found_doctor = True

        name_td = tds[1]
        lines = name_td.get_text("\n", strip=True).split("\n")

        for name in lines:
            if name and name[0] in ALPHABET_SET:
                doctors_name.add(name.replace(",", ""))

    return found_doctor


async def worker(
    doctor_type: str, session: aiohttp.ClientSession, queue: asyncio.Queue[int]
) -> None:

    while True:
        page: int = await queue.get()

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

    connector = aiohttp.TCPConnector(limit=CONCURRECY_LIMIT, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:

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

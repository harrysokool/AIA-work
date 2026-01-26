from bs4 import BeautifulSoup
import requests
import threading
import time
import asyncio
import aiohttp

doctors_name = set()
ALPHABET_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CONCURRECY_LIMIT = 10
OUTPUT_FILE = "doctors.txt"
BASE_URL = "https://www.mchk.org.hk/english/list_register/list.php"
BASE_PARAMS = {"page": "", "ipp": "20", "type": ""}
DOCTOR_TYPE = ["L", "O", "P", "M", "N"]


def timer() -> None:
    counter = 1
    while True:
        time.sleep(1)
        print(counter)
        counter += 1


def save_doctors():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name in doctors_name:
            f.write(name + "\n")


async def fetch_page(session, doctor_type, page):
    params = {"page": str(page), "ipp": "20", "type": doctor_type}
    async with session.get(BASE_URL, params=params) as resp:
        return await resp.text()


async def worker(doctor_type, session, queue):
    while True:
        page = await queue.get()

        try:
            # here need to fetch the page
            html = await fetch_page(session, doctor_type, page)
            soup = BeautifulSoup(html, "lxml")

            # try to find the target table on the website
            tables = soup.find_all("table")
            if not tables:
                queue.task_done()
                break
            tables = tables[3]

            # skipping the headers
            rows = tables.find_all("tr")[2:]
            if not rows:
                queue.task_done()
                break

            # flag to see if we should continue search for the next page
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


async def fetch_doctors(doctor_type) -> None:
    queue = asyncio.Queue()
    await queue.put(1)

    connector = aiohttp.TCPConnector(limit=CONCURRECY_LIMIT, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [
            asyncio.create_task(worker(doctor_type, session, queue))
            for _ in range(CONCURRECY_LIMIT)
        ]

        # wait until scraping is done
        await queue.join()

        # cancel the workers because workers run forever in a while True loop
        for w in workers:
            w.cancel()


async def main():
    # for each doctor type, we scrape them separately and concurrently
    tasks = [fetch_doctors(doc_type) for doc_type in DOCTOR_TYPE]
    await asyncio.gather(*tasks)

    save_doctors()


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

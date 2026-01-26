from bs4 import BeautifulSoup
import requests
import threading
import time

doctors_name = set()
ALPHABET_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

OUTPUT_FILE = "doctors.txt"
BASE_URL = "https://www.mchk.org.hk/english/list_register/list.php"
BASE_PARAMS = {"page": "", "ipp": "20", "type": ""}
DOCTOR_TYPE = ["L", "O", "P", "M", "N"]
DOCTOR_TYPE = ["M", "N"]


def check_doctor_type(doctor_type: str) -> bool:
    return (
        isinstance(doctor_type, str) and len(doctor_type) == 1 and doctor_type.isalpha()
    )


def timer() -> None:
    counter = 1
    while True:
        time.sleep(1)
        print(counter)
        counter += 1


def fetch():
    pass


def get_doctors_from_table(soup):
    table = soup.find_all("table")
    if not table:
        raise Exception("Could not find table")

    table = table[3]
    rows = table.find_all("tr")[2:]
    if not rows:
        raise Exception("Could not find rows")

    for row in rows:
        tds = row.find_all("td")
        if len(tds) == 2:  # no more doctors in the table
            break

        name_td = tds[1]
        lines = name_td.get_text("\n", strip=True).split("\n")

        for name in lines:
            if name and name[0] in ALPHABET_SET:
                doctors_name.add(name.replace(",", ""))


def save_doctors(doctors_name):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name in doctors_name:
            f.write(name + "\n")


def fetch_doctors() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for doctor_type in DOCTOR_TYPE:
        page = 1

        while True:
            time.sleep(0.3)
            try:
                params = {"page": str(page), "ipp": "20", "type": doctor_type}
                html = session.get(BASE_URL, params=params).text
                soup = BeautifulSoup(html, "lxml")

                # Select the table with the doctors info
                get_doctors_from_table(soup)

                page += 1
            except Exception as e:
                print(f"Error on type {doctor_type}, page {page}: {e}")
                break

    save_doctors(doctors_name)

    print(f"Saved {len(doctors_name)}")


if __name__ == "__main__":
    # timer for the program
    threading.Thread(target=timer, daemon=True).start()

    tic = time.perf_counter()
    fetch_doctors()
    toc = time.perf_counter()
    print(f"Time took: {toc - tic:0.4f}s")

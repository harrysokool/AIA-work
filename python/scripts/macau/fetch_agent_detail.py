import json
from pathlib import Path
import time
import csv
import asyncio
import aiohttp
import random
import threading
import sys
from typing import Any, Dict, List, Optional, Set


class RetryableFetchError(Exception):
    pass


DETAIL_URL = "https://iiep.amcm.gov.mo/platform-enquiry-service/public/api/v1/web/enquiry/licenses/detail"
CONCURRECY_LIMIT = 10
MAX_RETRIES = 5
SEM = asyncio.Semaphore(CONCURRECY_LIMIT)

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://iiep.amcm.gov.mo",
    "Referer": "https://iiep.amcm.gov.mo/",
}


def timer():
    counter = 1
    while True:
        time.sleep(1)
        print(counter)
        counter += 1


def check_category(category: str) -> bool:
    return category in {"aps", "ang"}


def load_raw_data(raw_file: Path):
    try:
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            agents = data.get("content", [])
            return agents if not None else None
    except Exception as e:
        print("Error reading raw file:", e)
        return None


def agent_has_company(detail_agent: dict, company_list: set) -> bool:
    for corp in detail_agent.get("corporates") or []:
        for item in corp.get("items") or []:
            nameEn = item.get("nameEn") or ""
            if nameEn in company_list:
                return True

    return False


def extract_detail_params(agents):
    if not agents:
        return None

    detail_params = []
    for agent in agents:
        if not agent:
            continue

        licenseCategory = agent.get("licenseCategory", "")
        license_no = agent.get("licenseNo", "")

        if not licenseCategory or not license_no:
            continue

        detail_params.append({"category": licenseCategory, "no": license_no})

    return detail_params


async def fetch(session, param, company_list):
    try:
        async with session.get(DETAIL_URL, params=param, headers=HEADERS) as response:
            if response.status != 200:
                raise RetryableFetchError(f"HTTP {response.status}")

            if "application/json" not in response.headers.get("Content-Type", ""):
                raise RetryableFetchError(
                    f"Unexpected content type: {response.headers.get('Content-Type')}"
                )

            detail = await response.json()

            if agent_has_company(detail, company_list):
                return detail
            else:
                return False

    except aiohttp.ClientError as e:
        raise RetryableFetchError(f"Network error: {e}")
    except Exception as e:
        raise RetryableFetchError(f"Unexpected error: {e}")


async def fetch_with_retry(session, param, company_list, retries=MAX_RETRIES):
    async with SEM:
        for attempt in range(1, retries + 1):
            try:
                result = await fetch(session, param, company_list)

                if result is False:
                    return None

                return result

            except RetryableFetchError as e:
                print(f"[Attempt {attempt}] {e}")
                if attempt < retries:
                    wait_time = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    print(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed after {retries} attempts for {param}")
                    return None


def flatten_agents_for_excel(detail_agents):
    """
    One row per agent-company relationship (per corporate->item).
    """
    rows = []
    for agent in detail_agents:
        base = {
            "licenseNo": agent.get("licenseNo"),
            "licenseCategory": agent.get("licenseCategory"),
            "namePt": agent.get("namePt"),
            "nameTc": agent.get("nameTc"),
            "status": agent.get("status"),
            "publicYear": agent.get("publicYear"),
            "publicMonth": agent.get("publicMonth"),
            "publishTime": agent.get("publishTime"),
        }

        for corp in agent.get("corporates") or []:
            items = corp.get("items") or []

            # If some corp types have empty items, still output a row for visibility
            if not items:
                rows.append(
                    {
                        **base,
                        "permitType": corp.get("type"),
                        "corpStatus": corp.get("status"),
                        "corpStartDate": corp.get("startDate"),
                        "corpEndDate": corp.get("endDate"),
                        "companyNameEn": "",
                        "companyNameTc": "",
                        "associationDate": "",
                    }
                )
                continue

            for item in items:
                rows.append(
                    {
                        **base,
                        "permitType": corp.get("type"),
                        "corpStatus": corp.get("status"),
                        "corpStartDate": corp.get("startDate"),
                        "corpEndDate": corp.get("endDate"),
                        "companyNameEn": item.get("nameEn"),
                        "companyNameTc": item.get("nameTc"),
                        "associationDate": item.get("aDate"),
                    }
                )
    return rows


def write_csv(rows, filepath: Path):
    if not rows:
        print("No rows to write.")
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig so Excel opens Chinese correctly
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written: {filepath}")


async def fetch_agent_detail_and_export(category: str, company_list: List[str]) -> None:
    category: str = category.strip().lower()
    if not check_category(category):
        print("Invalid category input")
        return

    # need to do some check on the company list
    company_list: Set[str] = set(company_list)

    print(f"Fetching agent details (category={category}, company={company_list})")

    # Directories and Files
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MACAU_DATA_DIR = BASE_DIR / "data" / "agents" / "macau"
    RAW_DATA_DIR = MACAU_DATA_DIR / category / "raw_agent_data"
    PROCESSED_DATA_DIR = MACAU_DATA_DIR / category / "processed_agent_data"
    EXPORT_DIR = MACAU_DATA_DIR / category / "exports"
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = RAW_DATA_DIR / "all.json"

    agents = load_raw_data(raw_file)
    if agents is None:
        return

    detail_agents = []
    detail_params = extract_detail_params(agents)

    connector = aiohttp.TCPConnector(limit=CONCURRECY_LIMIT, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_with_retry(session, param, company_list) for param in detail_params
        ]
        results = await asyncio.gather(*tasks)

    detail_agents = [r for r in results if r is not None]

    # Save processed JSON (detail objects)
    processed_file = PROCESSED_DATA_DIR / f"all_{category}.json"
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(detail_agents, f, ensure_ascii=False, indent=2)
    print(f"Processed JSON written: {processed_file}")

    # Export CSV for Excel
    rows = flatten_agents_for_excel(detail_agents)
    csv_file = EXPORT_DIR / f"all_{category}.csv"
    write_csv(rows, csv_file)

    print(f"Done. Matched {len(detail_agents)} agents, exported {len(rows)} rows")


if __name__ == "__main__":
    category: str = input("Enter category number (1: aps, 2: ang): ")
    if category == "1":
        category = "aps"
    elif category == "2":
        category = "ang"
    else:
        sys.exit(1)

    company_input: str = (
        input("Enter company names separated by commas (AIA INTERNATIONAL LIMITED): ")
        or "AIA INTERNATIONAL LIMITED"
    )

    company_list: List[str] = [
        company.strip().upper()
        for company in company_input.split(",")
        if company_input.split()
    ]

    threading.Thread(target=timer, daemon=True).start()

    tic = time.perf_counter()
    asyncio.run(fetch_agent_detail_and_export(category, company_list))
    toc = time.perf_counter()
    print(f"Time took: {toc - tic:0.4f}s")


# some company name to test
# ASIA INSURANCE COMPANY LIMITED
# AIA INTERNATIONAL LIMITED

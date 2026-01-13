import json
import requests
from pathlib import Path
import time
import csv
import asyncio
import aiohttp
import random

DETAIL_URL = "https://iiep.amcm.gov.mo/platform-enquiry-service/public/api/v1/web/enquiry/licenses/detail"
CONCURRECY_LIMIT = 20
MAX_RETRIES = 5
SEM = asyncio.Semaphore(CONCURRECY_LIMIT)

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://iiep.amcm.gov.mo",
    "Referer": "https://iiep.amcm.gov.mo/",
}


def check_category(category: str) -> bool:
    return category in {"aps", "ang"}


def load_raw_data(raw_file):
    try:
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Error reading raw file:", e)
        return None
    return data


def agent_has_company(detail_agent: dict, company: str) -> bool:
    for corp in detail_agent.get("corporates") or []:
        for item in corp.get("items") or []:
            nameEn = item.get("nameEn") or ""
            if nameEn.startswith(company):
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


async def fetch(session, param, company):
    try:
        print("getting detail for:", param)
        async with session.get(DETAIL_URL, params=param, headers=HEADERS) as response:
            if response.status != 200:
                text = await response.text()
                print(f"Error {response.status} for {param}: {text[:200]}...")
                return None

            if "application/json" not in response.headers.get("Content-Type", ""):
                text = await response.text()
                print(
                    f"Unexpected content type for {param}: {response.headers.get('Content-Type')}"
                )
                print(f"Response preview: {text[:200]}...")
                return None

            detail = await response.json()

            if agent_has_company(detail, company):
                return detail
            else:
                return None

    except Exception as e:
        print(f"Error fetching {param}: {e}")
        return None


async def fetch_with_retry(session, param, company, retries=MAX_RETRIES):
    async with SEM:
        await asyncio.sleep(0.3 + random.random() * 0.7)
        for attempt in range(1, retries + 1):
            try:
                await asyncio.sleep(random.uniform(0.5, 1.5))
                return await fetch(session, param, company)
            except (
                aiohttp.ClientOSError,
                aiohttp.ClientResponseError,
                ConnectionResetError,
            ) as e:
                print(f"[Attempt {attempt}] Error fetching {param}: {e}")
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


async def fetch_agent_detail_and_export(category: str, company: str = "AIA"):
    category = category.strip().lower()
    company = company.strip()

    if not check_category(category):
        print("Invalid category input")
        return

    print(f"Fetching agent details (category={category}, company={company})")

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MACAU_DATA_DIR = BASE_DIR / "data" / "agents" / "macau"
    RAW_DATA_DIR = MACAU_DATA_DIR / category / "raw_agent_data"
    PROCESSED_DATA_DIR = MACAU_DATA_DIR / category / "processed_agent_data"
    EXPORT_DIR = MACAU_DATA_DIR / category / "exports"

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = RAW_DATA_DIR / "all.json"

    data = load_raw_data(raw_file)
    if data is None:
        return

    agents = data.get("content", [])
    detail_agents = []
    detail_params = extract_detail_params(agents)

    connector = aiohttp.TCPConnector(limit=5, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch(session, param, company) for param in detail_params]
        results = await asyncio.gather(*tasks)

    detail_agents = [r for r in results if r is not None]

    # Save processed JSON (detail objects)
    processed_file = PROCESSED_DATA_DIR / f"all_{company}.json"
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(detail_agents, f, ensure_ascii=False, indent=2)
    print(f"Processed JSON written: {processed_file}")

    # Export CSV for Excel
    rows = flatten_agents_for_excel(detail_agents)
    csv_file = EXPORT_DIR / f"all_{company}.csv"
    write_csv(rows, csv_file)

    print(f"Done. Matched {len(detail_agents)} agents, exported {len(rows)} rows")


if __name__ == "__main__":
    category = input("Enter category (aps, ang): ")
    company = input("Enter company prefix (e.g., AIA): ").strip() or "AIA"

    tic = time.perf_counter()
    asyncio.run(fetch_agent_detail_and_export(category, company))
    toc = time.perf_counter()
    print(f"Time took: {toc - tic:0.4f}s")

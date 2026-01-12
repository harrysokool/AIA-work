import json
import requests
from pathlib import Path
import time
import csv

# URL
DETAIL_URL = "https://iiep.amcm.gov.mo/platform-enquiry-service/public/api/v1/web/enquiry/licenses/detail"


def check_category(category: str) -> bool:
    return category in {"aps", "ang"}


def agent_has_company(detail_agent: dict, company: str) -> bool:
    for corp in detail_agent.get("corporates") or []:
        for item in corp.get("items") or []:
            nameEn = item.get("nameEn") or ""
            if nameEn.startswith(company):
                return True
    return False


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


def fetch_agent_detail_and_export(category: str, company: str = "AIA"):
    category = category.strip().lower()
    company = company.strip()

    if not check_category(category):
        print("Invalid category input")
        return

    print(f"Fetching agent details (category={category}, company={company})")

    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "MyApp/1.0",
            "Accept": "application/json",
        }
    )

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MACAU_DATA_DIR = BASE_DIR / "data" / "agents" / "macau"
    RAW_DATA_DIR = MACAU_DATA_DIR / category / "raw_agent_data_by_letter"
    PROCESSED_DATA_DIR = MACAU_DATA_DIR / category / "processed_agent_data_by_letter"
    EXPORT_DIR = MACAU_DATA_DIR / category / "exports"

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = RAW_DATA_DIR / "all.json"
    try:
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Error reading raw file:", e)
        return

    agents = data.get("content", [])
    agent_skipped = 0

    detail_agents = []
    seen_license = set()

    for agent in agents:
        print(f"fetchin {agent.get("namePt")} detail")
        licenseCategory = agent.get("licenseCategory", "")
        license_no = agent.get("licenseNo", "")

        if not licenseCategory or not license_no:
            agent_skipped += 1
            continue

        # avoid duplicate detail fetch
        key = (licenseCategory, license_no)
        if key in seen_license:
            continue
        seen_license.add(key)

        detail_params = {"category": licenseCategory, "no": license_no}

        try:
            resp = s.get(DETAIL_URL, params=detail_params, timeout=10)
            if resp.status_code != 200:
                agent_skipped += 1
                continue

            detail = resp.json()

            # keep only agents matching company
            if agent_has_company(detail, company):
                detail_agents.append(detail)
            else:
                agent_skipped += 1

        except Exception as e:
            print("Error fetching detail:", e)
            agent_skipped += 1
            continue

        time.sleep(0.5)

    # Save processed JSON (detail objects)
    processed_file = PROCESSED_DATA_DIR / f"all_{company}.json"
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(detail_agents, f, ensure_ascii=False, indent=2)
    print(f"Processed JSON written: {processed_file}")

    # Export CSV for Excel
    rows = flatten_agents_for_excel(detail_agents)
    csv_file = EXPORT_DIR / f"all_{company}.csv"
    write_csv(rows, csv_file)

    print(
        f"Done. Matched {len(detail_agents)} agents, exported {len(rows)} rows, skipped {agent_skipped}."
    )


if __name__ == "__main__":
    category = input("Enter category (aps, ang): ")
    company = input("Enter company prefix (e.g., AIA): ").strip() or "AIA"

    tic = time.perf_counter()
    fetch_agent_detail_and_export(category, company)
    toc = time.perf_counter()
    print(f"Time took: {toc - tic:0.4f}s")

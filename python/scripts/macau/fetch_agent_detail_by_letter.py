import json
import sys
import requests
from pathlib import Path
import time


def filter_agent_by_letter(agents, letter):
    if not agents:
        pass

    filtered_agents = []
    for agent in agents:
        lastname = agent.get("namePt", "").strip().lower()
        if lastname and lastname[0] == letter:
            filtered_agents.append(agent)

    return filtered_agents


def filter_agent_by_company(agents, company):
    if not agents:
        return

    filtered_agents = []
    for agent in agents:
        corp = agent.get("corporates", [])
        corp_items = corp.get("items", [])
        for item in corp_items:
            corp_item_name = item.get("nameEn", "")
            if corp_item_name[:3] == company:
                filtered_agents.append(agent)

    return filtered_agents


def fetch_agent_detail(letter, category):
    print(f"Fetching agent details for letter: {letter}")

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
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = RAW_DATA_DIR / f"{letter}.json"
    try:
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Error reading raw file:", e)
        return

    agents = data.get("content", [])
    DETAIL_URL = "https://iiep.amcm.gov.mo/platform-enquiry-service/public/api/v1/web/enquiry/licenses/detail"

    valid_agents = filter_agent_by_letter(agents, letter)
    agent_detail = []
    for agent in valid_agents:
        licenseCategory = agent.get("licenseCategory", "")
        license_no = agent.get("licenseNo", "")

        if not licenseCategory or not license_no:
            continue

        detail_params = {"category": licenseCategory, "no": license_no}

        try:
            resp = s.get(DETAIL_URL, params=detail_params, timeout=10)
            if resp.status_code != 200:
                continue

            detail_obj = resp.json()
        except Exception as e:
            print("Error fetching detail:", e)
            continue

        agent_detail = filter_agent_by_company(detail_obj, "AIA")

        time.sleep(0.5)

    output_file = PROCESSED_DATA_DIR / f"{letter}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(agent_detail, f, ensure_ascii=False, indent=2)

    print(f"Letter {letter} saved {len(agent_detail)} records")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <letter> <licenseCategory>")
        sys.exit(1)
    fetch_agent_detail(sys.argv[1], sys.argv[2])

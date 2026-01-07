import json
import sys
import requests
from pathlib import Path
import time

def fetch_agent_detail(letter):
    print(f"Fetching agent details for letter: {letter}")

    s = requests.Session()
    s.headers.update({
        "User-Agent": "MyApp/1.0",
        "Accept": "application/json",
    })

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "agents" /"raw_agent_data_by_letter_macau"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "agents" / "processed_agent_data_by_letter_macau"
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

    agent_detail = []
    for agent in agents:
        name_pt = (agent.get("namePt") or "").strip()
        if not name_pt.lower().startswith(letter.lower()):
            print(f"Skipping agent {name_pt} due name does not start with letter {letter}")
            continue

        category = agent.get("licenseCategory", "")
        license_no = agent.get("licenseNo", "")

        if not category or not license_no:
            print(f"Skipping agent {name_pt} due to missing license info.")
            continue

        detail_params = {
            "category": category, 
            "no": license_no
        }

        try:
            resp = s.get(DETAIL_URL, params=detail_params, timeout=10)
            if resp.status_code != 200:
                print(f"Request Error {resp.status_code} for {name_pt}")
                continue
            agent_detail.append(resp.json())
        except Exception as e:
            print("Error fetching detail:", e)
            continue

        time.sleep(0.5)

    output_file = PROCESSED_DATA_DIR / f"{letter}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(agent_detail, f, ensure_ascii=False, indent=2)

    print(f"Letter {letter} saved {len(agent_detail)} records")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <letter>")
        sys.exit(1)
    fetch_agent_detail(sys.argv[1])

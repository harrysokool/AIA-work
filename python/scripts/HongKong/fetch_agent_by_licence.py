import requests
from pathlib import Path
import json
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
EMPTY_RESPONSE = '{"data":[],"itemsCount":"0","errorCode":null,"errorMsg":null}'


def fetch_agent(sessionToken, licence):
    s = requests.Session()
    print("session created!")

    # url for searching the agents
    BASE_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search"

    # variables
    token = sessionToken

    # need to loop through the alphabet letters for the first name
    # surNameValue, token both need update
    # other params are fixed
    params = {
        "seachIndicator": "licNo",
        "searchValue": "",
        "status": "all",
        "page": 1,
        "pagesize": 10,
        "token": token,
    }

    headers = {
        "Accept": "application/json",
        "Referer": "https://iir.ia.org.hk/",
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # directory/file paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "agents" / "hongkong"
    ALL_FILE = DATA_DIR / "agents_all_count.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    totalActiveAgentCount = 0
    agentDict = {}

    licence = licence.lower()

    for i in range(10000):
        print(i)
        num = f"{i:04d}"
        id_value = licence + num
        params["searchValue"] = id_value
        # params["searchValue"] = "ia9999"

        try:
            resp = s.get(
                f"{BASE_URL}/individual",
                params=params,
                headers=headers,
                timeout=30,
                verify=False,
            )
        except Exception as e:
            print(f"Request error for {num}: {e}")
            continue

        if not resp.ok:
            print(f"Failed for {num}: {resp.status_code}")
            continue

        if resp.text.strip() == EMPTY_RESPONSE:
            continue

        try:
            obj = resp.json()
        except ValueError:
            print(f"Invalid JSON for {num}")
            continue

        print(obj)

    # print(
    #     f"{licence} saved {agentCount} records, skipped {agentSkipped} due to mismatch letter with last name"
    # )

    agentDict["totalAgentCount"] = totalActiveAgentCount
    with open(ALL_FILE, "w", encoding="utf-8") as f:
        json.dump(agentDict, f, ensure_ascii=False, indent=2)

    print(f"total active agent count: {totalActiveAgentCount}")
    print("Done counting agents.")


if __name__ == "__main__":
    token = input("Enter Token: ")
    licenseLetter = input("Enter Licence Letter: ")

    tic = time.perf_counter()
    fetch_agent(
        token,
        licenseLetter,
    )
    toc = time.perf_counter()
    print(f"total time took: {toc - tic:0.4f}")

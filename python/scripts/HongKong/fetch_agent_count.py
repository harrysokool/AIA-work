import requests
from pathlib import Path
import json
import sys
import time


def fetch_agent(sessionToken):
    s = requests.Session()
    print("session created!")

    # url for searching the agents
    BASE_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search"

    # variables
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    token = sessionToken

    # need to loop through the alphabet letters for the first name
    # surNameValue, token both need update
    # other params are fixed
    params = {
        "seachIndicator": "engName",
        "status": "A",
        "surNameValue": "",
        "givenNameValue": "",
        "page": 1,
        "pagesize": 1000,
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
    for letter in alphabets:
        params["surNameValue"] = letter

        try:
            resp = s.get(
                f"{BASE_URL}/individual", params=params, headers=headers, timeout=30
            )
            if resp.status_code != 200:
                print(f"Failed for {letter}: {resp.status_code}")
                continue
            obj = resp.json()
        except Exception as e:
            print("Error", e)
            continue

        agentCount = 0
        agentSkipped = 0
        for agent in obj.get("data", []):
            name = agent.get("engName", "").lower()
            if name and name[0] == letter:
                agentCount += 1
            else:
                agentSkipped += 1

        agentDict[letter] = agentCount
        totalActiveAgentCount += agentCount

        print(
            f"Letter {letter} saved {agentCount} records, skipped {agentSkipped} due to mismatch letter with last name"
        )

    agentDict["totalAgentCount"] = totalActiveAgentCount
    with open(ALL_FILE, "w", encoding="utf-8") as f:
        json.dump(agentDict, f, ensure_ascii=False, indent=2)

    print(f"total active agent count: {totalActiveAgentCount}")
    print("Done counting agents.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_agents.py <token>")
        sys.exit(1)
    tic = time.perf_counter()
    fetch_agent(sys.argv[1])
    toc = time.perf_counter()
    print(f"total time took: {toc - tic:0.4f}")

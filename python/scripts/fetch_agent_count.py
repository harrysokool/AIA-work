import requests
from pathlib import Path
import json
import sys

def fetch_agent(sessionToken):
    s = requests.Session()
    print('session created!')

    # clear and set up cookies
    s.cookies.clear()

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
        "token": token
    }
    headers = {
        "Accept": "application/json",
        "Referer": "https://iir.ia.org.hk/",
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }

    # directory/file paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    AGENTS_DIR = BASE_DIR / "data" / "agents"
    BY_LETTER_DIR = AGENTS_DIR / "agents_by_letter"
    BY_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    ALL_FILE = AGENTS_DIR / "agents_all_count.json"

    totalActiveAgentCount = 0
    agentDict = {}
    for letter in alphabets:
        params["surNameValue"] = letter

        resp = s.get(f"{BASE_URL}/individual", params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"Failed for {letter}: {resp.status_code}")
            continue
        
        # dict_keys(['data', 'itemsCount', 'errorCode', 'errorMsg']), this will return these fields
        obj = resp.json()

        agentCount = 0
        for agent in obj.get("data", []):
            if agent.get("engName", "").lower().startswith(letter):
                agentCount += 1
                
        agentDict[f"{letter}_count"] = agentCount
        totalActiveAgentCount += agentCount

        print(f"Letter {letter} saved {agentCount} records")

    agentDict["totalAgentCount"] = totalActiveAgentCount
    with open(ALL_FILE, "w", encoding="utf-8") as f:
        json.dump(agentDict, f, ensure_ascii=False, indent=2)

    print("Done counting agents.")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_agents.py <token>")
        sys.exit(1)

    fetch_agent(sys.argv[1])
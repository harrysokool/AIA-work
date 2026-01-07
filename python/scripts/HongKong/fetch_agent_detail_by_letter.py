import requests
from pathlib import Path
import json
import sys

def fetch_agent(sessionToken, letter):
    s = requests.Session()
    print('session created!')

    # url for searching the agents
    BASE_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search"

    # variables
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

    # directory paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    BY_LETTER_DIR = BASE_DIR / "data" / "agents" / "hongkong" / "agents_by_letter"
    BY_LETTER_DIR.mkdir(parents=True, exist_ok=True)

    params["surNameValue"] = letter.strip().lower()

    try:
        resp = s.get(f"{BASE_URL}/individual", params=params, headers=headers, timeout=30)
    except Exception as e:
        print("Error", e)
        sys.exit()
        
    if resp.status_code != 200:
        print(f"Failed for {letter}: {resp.status_code}")
        sys.exit()
        
    # dict_keys(['data', 'itemsCount', 'errorCode', 'errorMsg']), this will return these fields
    try:
        obj = resp.json()
    except Exception as e:
        print("Error", e)
        sys.exit()

    # filter the agents' surname that does not start with the letter
    agentCount = 0
    letter_agent = []
    for agent in obj.get("data", []):
        if agent.get("engName", "").lower().startswith(letter.lower()):
            key = agent.get("key", "")
            licStatus = agent.get("licenseStatus", "")
            detail_params = {
                "key": key,
                "licStatus": licStatus,
                "token": token
            }
            try:
                detail_resp = s.get(f"{BASE_URL}/individualDetail", params=detail_params)
            except Exception as e:
                print("Error", e)
                sys.exit()
            agentDetail = detail_resp.json()
            letter_agent.append(agentDetail)
            agentCount += 1
    
    with open(BY_LETTER_DIR / f"{letter}.json", "w", encoding="utf-8") as f:
        json.dump(letter_agent, f, ensure_ascii=False, indent=2)

    print(f"Letter {letter} saved {agentCount} records")
    
    
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_agents.py <token> <letter>")
        sys.exit(1)

    fetch_agent(sys.argv[1], sys.argv[2])
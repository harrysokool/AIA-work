import requests
from pathlib import Path
import json
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, Union

def ms_to_human_time(
    ms: Optional[Union[int, float]],
    tz: str = "UTC",
    fmt: str = "%Y-%m-%d %H:%M:%S"
) -> Optional[str]:
    if ms is None:
        return None

    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return None

    if ms <= 0:
        return None

    seconds = ms / 1000

    if tz.upper() == "HK":
        tzinfo = timezone(timedelta(hours=8))
        dt = datetime.fromtimestamp(seconds, tz=tzinfo)
    else:
        dt = datetime.utcfromtimestamp(seconds).replace(tzinfo=timezone.utc)

    return dt.strftime(fmt)


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

    # directory paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    AGENTS_DIR = BASE_DIR / "data" / "agents"
    BY_LETTER_DIR = AGENTS_DIR / "agents_by_letter"
    BY_LETTER_DIR.mkdir(parents=True, exist_ok=True)

    totalActiveAgentCount = 0
    all_agent = []
    for letter in alphabets:
        params["surNameValue"] = letter

        resp = s.get(f"{BASE_URL}/individual", params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"Failed for {letter}: {resp.status_code}")
            continue
        
        # dict_keys(['data', 'itemsCount', 'errorCode', 'errorMsg']), this will return these fields
        obj = resp.json()

        # filter the agents' surname that does not start with the letter
        letter_agent = []
        agentCount = 0
        for agent in obj.get("data", []):
            if agent.get("engName", "").lower().startswith(letter):
                # going to get the specific agent detail here
                key = agent.get("key", "")
                licStatus = agent.get("licenseStatus", "")
                detail_params = {
                    "key": key,
                    "licStatus": licStatus,
                    "token": token
                }
                detail_resp = s.get(f"{BASE_URL}/individualDetail", params=detail_params)
                agentDetail = detail_resp.json()
                agentDetail["licenseStartDateHKT"] = ms_to_human_time(agentDetail["licenseStartDate"])
                agentDetail["licenseEndDateHKT"] = ms_to_human_time(agentDetail["licenseEndDate"])
                letter_agent.append(agentDetail)
                agentCount += 1
            
        all_agent.extend(letter_agent)
        totalActiveAgentCount += agentCount

        with open(BY_LETTER_DIR / f"{letter}.json", "w", encoding="utf-8") as f:
            json.dump(letter_agent, f, ensure_ascii=False, indent=2)

        print(f"Letter {letter} saved {agentCount} records")

    # save all agents into single json file
    ALL_FILE = AGENTS_DIR / "agents_all.json"
    with open(ALL_FILE, "w", encoding="utf-8") as f:
        json.dump(all_agent, f, ensure_ascii=False, indent=2)

    print("saved", len(all_agent), "agent records")
    print("Done saving all agents locally.")
    
    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_agents.py <token>")
        sys.exit(1)

    fetch_agent(sys.argv[1])
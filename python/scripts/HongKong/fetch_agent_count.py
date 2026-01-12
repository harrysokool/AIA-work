import requests
from pathlib import Path
import json
import time


# status must be either a, i, all, no other option
def check_status(status):
    if status not in {"a", "i", "all"}:
        return False
    return True


# this will count agent where their lastname prefix with prefix
def count_agent(obj, prefix):
    if not obj:
        return (0, 0)

    prefixLength = len(prefix)
    agent_count = 0
    agent_skipped = 0
    agents = obj.get("data", [])

    for agent in agents:
        lastname = agent.get("engName", "").lower()
        if lastname and lastname[:prefixLength] == prefix:
            agent_count += 1
        else:
            agent_skipped += 1

    return (agent_count, agent_skipped)


def fetch_agent(sessionToken, status):
    s = requests.Session()
    print("session created!")

    status = status.strip().lower()
    if not check_status(status):
        print("Invalid status input")
        return
    if len(status) == 1:
        status = status.upper()
    else:
        status = status.lower()

    # url for searching the agents
    BASE_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search"

    # variables
    alphabets = "abcdefghijklmnopqrstuvwxyz".upper()
    token = sessionToken

    # need to loop through the alphabet letters for the first name
    # surNameValue, token both need update
    # other params are fixed
    params = {
        "seachIndicator": "engName",
        "status": status,
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
                f"{BASE_URL}/individual",
                params=params,
                headers=headers,
                timeout=30,
                verify=False,
            )

            if resp.status_code != 200:
                print(f"Failed for {letter}: {resp.status_code}")
                continue

            obj = resp.json()
        except Exception as e:
            print("Error", e)
            continue

        agentCount, agentSkipped = count_agent(obj, letter)
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
    token = input("Enter Token: ")
    status = input("Enter Licence Status (a: Active, i: inactive, all: all): ")

    tic = time.perf_counter()
    fetch_agent(token, status)
    toc = time.perf_counter()
    print(f"total time took: {toc - tic:0.4f}")

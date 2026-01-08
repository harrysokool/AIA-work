from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from pathlib import Path
import json
import sys

# directory/file paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ALL_FILE = BASE_DIR / "data" / "agents" / "hongkong" / "agents_all_count.json"

# url for searching the agents
BASE_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search"

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
    "token": "",
}
headers = {
    "Accept": "application/json",
    "Referer": "https://iir.ia.org.hk/",
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}


def count_letter(token, letter):
    session = requests.Session()
    print(f"session created for letter {letter}")
    local_params = params.copy()
    local_params["token"] = token
    local_params["surNameValue"] = letter

    resp = None
    try:
        resp = session.get(
            f"{BASE_URL}/individual", params=local_params, headers=headers, timeout=30
        )
        resp.raise_for_status()
        obj = resp.json()
    except Exception as e:
        print(f"Error fetching for {letter}:", e)
        return letter, 0

    agentCount = 0
    agentSkipped = 0
    for agent in obj.get("data", []):
        name = agent.get("engName", "")
        if name and name[0].lower() == letter:
            agentCount += 1
        else:
            agentSkipped += 1

    print(f"Letter {letter} saved {agentCount} records, skipped {agentSkipped}")
    return letter, agentCount


def fetch_agent(sessionToken):
    agentDict = {}
    totalActiveAgentCount = 0

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(count_letter, sessionToken, letter)
            for letter in "abcdefghijklmnopqrstuvwxyz"
        ]
        for f in as_completed(futures):
            letter, count = f.result()
            agentDict[letter] = count
            totalActiveAgentCount += count
            print(f"{letter}", count)

    agentDict["totalAgentCount"] = totalActiveAgentCount
    with open(ALL_FILE, "w", encoding="utf-8") as f:
        json.dump(agentDict, f, ensure_ascii=False, indent=2)

    print(f"total active agent count: {totalActiveAgentCount}")
    print("Done counting agents.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_agents.py <token>")
        sys.exit(1)
    fetch_agent(sys.argv[1])

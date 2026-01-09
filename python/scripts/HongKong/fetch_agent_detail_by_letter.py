import requests
from pathlib import Path
import json
import sys


# letter must be length of 1 and must be a single char from the alphabet
def check_letter(letter):
    if len(letter) != 1:
        return False
    if len(letter) == 1 and letter not in "abcdefghijklmnopqrstuvwxyz":
        return False
    return True


# status must be either a, i, all, no other option
def check_status(status):
    if status not in {"a", "i", "all"}:
        return False
    return True


def fetch_agent(sessionToken, letter, status):
    s = requests.Session()
    print("session created!")

    headers = {
        "Accept": "application/json",
        "Referer": "https://iir.ia.org.hk/",
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    s.headers.update(headers)

    # variables
    letter = letter.strip().lower()
    if not check_letter(letter):
        print("Invalid letter input")
        return

    status = status.strip().lower()
    if not check_status(status):
        print("Invalid status input")
        return
    if len(status) == 1:
        status = status.upper()
    else:
        status = status.lower()

    params = {
        "seachIndicator": "engName",
        "status": status,
        "surNameValue": letter,
        "givenNameValue": "",
        "page": 1,
        "pagesize": 1000,
        "token": sessionToken,
    }

    # url for searching the agents
    BASE_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search"

    # directory paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    BY_LETTER_DIR = BASE_DIR / "data" / "agents" / "hongkong" / "agents_by_letter"
    BY_LETTER_DIR.mkdir(parents=True, exist_ok=True)

    try:
        resp = s.get(f"{BASE_URL}/individual", params=params, timeout=30)
        resp.raise_for_status()
        obj = resp.json()
    except Exception as e:
        print("Error", e)
        return

    agentCount = 0
    letter_agent = []

    for agent in obj.get("data", []):
        name = agent.get("engName", "").lower()
        if name and name[0] == letter:
            key = agent.get("key", "")
            licStatus = agent.get("licenseStatus", "")
            detail_params = {"key": key, "licStatus": licStatus, "token": sessionToken}

            try:
                detail_resp = s.get(
                    f"{BASE_URL}/individualDetail",
                    params=detail_params,
                )
                detail_resp.raise_for_status()
                agentDetail = detail_resp.json()
                letter_agent.append(agentDetail)
                agentCount += 1
            except Exception as e:
                print("Error", e)
                sys.exit()

    OUTPUT_FILE = ""
    if status == "A":
        OUTPUT_FILE = BY_LETTER_DIR / f"{letter}_active.json"
    elif status == "I":
        OUTPUT_FILE = BY_LETTER_DIR / f"{letter}_inactive.json"
    elif status == "all":
        OUTPUT_FILE = BY_LETTER_DIR / f"{letter}_all.json"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(letter_agent, f, ensure_ascii=False, indent=2)

    print(f"Letter {letter} saved {agentCount} records")


if __name__ == "__main__":
    token = input("Enter Token: ")
    letter = input("Enter Letter: ")
    status = input("Enter Status (a: active, i: inactive, all: all): ")
    fetch_agent(token, letter, status)

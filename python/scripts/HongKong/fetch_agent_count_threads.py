from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import random
from pathlib import Path
import json
import sys
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "agents" / "hongkong"
ALL_FILE = DATA_DIR / "agents_all_count.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search"

BASE_PARAMS = {
    "seachIndicator": "engName",
    "status": "",
    "surNameValue": "",
    "givenNameValue": "",
    "page": 1,
    "pagesize": 1000,
    "token": "",
}

HEADERS = {
    "Accept": "application/json",
    "Referer": "https://iir.ia.org.hk/",
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}


def check_status(status):
    if status not in {"a", "i", "all"}:
        return False
    return True


def make_session():
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount(
        "https://", HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    )
    return s


def safe_json(resp, letter):
    try:
        return resp.json()
    except Exception:
        # Print useful debug info
        ct = resp.headers.get("Content-Type", "")
        snippet = (resp.text or "")[:200].replace("\n", " ")
        print(
            f"[{letter}] JSON parse failed. status={resp.status_code} content-type={ct} body_snippet={snippet!r}"
        )
        raise


def count_agent(obj, letter):
    if not obj:
        return (0, 0)

    agent_count = 0
    agent_skipped = 0
    agents = obj.get("data", [])
    if len(agents) >= 1000:
        print(letter, "is more than 1000")

    for agent in agents:
        lastname = agent.get("engName", "").lower()
        if lastname and lastname[0] == letter:
            agent_count += 1
        else:
            agent_skipped += 1

    return (agent_count, agent_skipped)


def count_letter(session, token, letter, status, attempts=6):
    params = dict(BASE_PARAMS)
    params["token"] = token
    params["surNameValue"] = letter
    params["status"] = status

    last_err = None
    for i in range(attempts):
        # jitter + exponential backoff
        time.sleep(random.uniform(0.05, 0.25) + (0.4 * (2**i)) * 0.1)

        try:
            resp = session.get(
                f"{BASE_URL}/individual",
                params=params,
                headers=HEADERS,
                timeout=30,
                verify=False,
            )

            if resp.status_code in (401, 403):
                print(f"[{letter}] auth error {resp.status_code}")
                return letter, 0

            # If 200 but HTML/blank, treat as retryable
            ct = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" not in ct:
                snippet = (resp.text or "")[:200].replace("\n", " ")
                raise ValueError(
                    f"non-json response ct={ct} status={resp.status_code} snippet={snippet!r}"
                )

            resp.raise_for_status()
            obj = resp.json()  # may raise -> retry

            agentCount, agentSkipped = count_agent(obj, letter)

            print(f"Letter {letter} saved {agentCount} records, skipped {agentSkipped}")
            return letter, agentCount

        except Exception as e:
            last_err = e
            print(f"[{letter}] attempt {i + 1}/{attempts} failed: {e}")

    print(f"[{letter}] giving up after {attempts} attempts: {last_err}")
    return letter, 0


def fetch_agent(token, status, max_workers=6):
    status = status.strip().lower()
    if not check_status(status):
        print("Invalid status input")
        return
    if len(status) == 1:
        status = status.upper()
    else:
        status = status.lower()

    total = 0
    agentDict = {}

    # one shared session per worker thread
    sessions = [make_session() for _ in range(max_workers)]

    letters = "abcdefghijklmnopqrstuvwxyz"
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, letter in enumerate(letters):
            sess = sessions[i % max_workers]
            futures.append(pool.submit(count_letter, sess, token, letter, status))

        for f in as_completed(futures):
            try:
                letter, count = f.result()
            except Exception as e:
                print("future failed:", e)
                continue
            agentDict[letter] = count
            total += count

    agentDict["totalAgentCount"] = total
    with open(ALL_FILE, "w", encoding="utf-8") as f:
        json.dump(agentDict, f, ensure_ascii=False, indent=2)

    print("total active agent count:", total)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_agent_count_threads.py <token> <status>")
        sys.exit(1)

    tic = time.perf_counter()
    fetch_agent(sys.argv[1], sys.argv[2], max_workers=10)
    toc = time.perf_counter()
    print(f"total time took: {toc - tic:0.4f}")

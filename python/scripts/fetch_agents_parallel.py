import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LIST_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search/individual"
DETAIL_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search/individualDetail"


def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def safe_json(resp: requests.Response) -> Optional[Dict[str, Any]]:
    try:
        return resp.json()
    except ValueError:
        return None


def get_json(
    s: requests.Session,
    url: str,
    params: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float = 30.0
) -> Optional[Dict[str, Any]]:
    try:
        r = s.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException:
        return None

    if r.status_code != 200:
        return None

    return safe_json(r)


def load_progress(progress_file: Path) -> Dict[str, Any]:
    if progress_file.exists():
        return json.loads(progress_file.read_text(encoding="utf-8"))
    return {}


def save_progress(progress_file: Path, progress: Dict[str, Any]) -> None:
    progress_file.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_one_letter(
    letter: str,
    token: str,
    out_dir: Path,
    headers: Dict[str, str],
    pagesize: int = 1000,
    per_detail_sleep: float = 0.02
) -> int:
    """
    Fetch list+detail for a single starting-letter partition.
    Writes output to out_dir/<letter>.json and maintains out_dir/progress.json.
    Returns number of saved agent records for this letter.
    """
    s = build_session()

    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.json"
    progress = load_progress(progress_file)

    letter_state = progress.get(letter, {})
    start_page = int(letter_state.get("next_page", 1))
    done = bool(letter_state.get("done", False))
    if done:
        return int(letter_state.get("saved_count", 0))

    letter_agents: List[Dict[str, Any]] = []
    letter_file = out_dir / f"{letter}.json"
    if letter_file.exists() and start_page > 1:
        try:
            letter_agents = json.loads(letter_file.read_text(encoding="utf-8"))
        except Exception:
            letter_agents = []

    saved_count = len(letter_agents)
    page = start_page

    while True:
        params = {
            "seachIndicator": "engName",
            "status": "A",
            "surNameValue": letter,
            "givenNameValue": "",
            "page": page,
            "pagesize": pagesize,
            "token": token,
        }

        obj = get_json(s, LIST_URL, params=params, headers=headers, timeout=30.0)
        if obj is None:
            progress[letter] = {"next_page": page, "done": False, "saved_count": saved_count}
            save_progress(progress_file, progress)
            return saved_count

        data = obj.get("data") or []
        if not data:
            progress[letter] = {"next_page": page, "done": True, "saved_count": saved_count}
            save_progress(progress_file, progress)
            letter_file.write_text(json.dumps(letter_agents, ensure_ascii=False, indent=2), encoding="utf-8")
            return saved_count

        for agent in data:
            eng_name = (agent.get("engName") or "").strip().lower()
            if not eng_name.startswith(letter):
                continue

            key = agent.get("key") or ""
            lic_status = agent.get("licenseStatus") or ""

            detail_params = {"key": key, "licStatus": lic_status, "token": token}
            detail_obj = get_json(s, DETAIL_URL, params=detail_params, headers=headers, timeout=30.0)

            agent["agentDetail"] = detail_obj
            letter_agents.append(agent)
            saved_count += 1

            if per_detail_sleep:
                time.sleep(per_detail_sleep)

        letter_file.write_text(json.dumps(letter_agents, ensure_ascii=False, indent=2), encoding="utf-8")
        progress[letter] = {"next_page": page + 1, "done": False, "saved_count": saved_count}
        save_progress(progress_file, progress)

        items_count = obj.get("itemsCount")
        if isinstance(items_count, int):
            max_pages = (items_count + pagesize - 1) // pagesize
            if page >= max_pages:
                progress[letter] = {"next_page": page, "done": True, "saved_count": saved_count}
                save_progress(progress_file, progress)
                return saved_count
        else:
            if len(data) < pagesize:
                progress[letter] = {"next_page": page, "done": True, "saved_count": saved_count}
                save_progress(progress_file, progress)
                return saved_count

        page += 1


def fetch_all_letters_parallel(token: str) -> None:
    BASE_DIR = Path(__file__).resolve().parent.parent
    AGENTS_DIR = BASE_DIR / "data" / "agents"
    BY_LETTER_DIR = AGENTS_DIR / "agents_by_letter_parallel"
    BY_LETTER_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "Accept": "application/json",
        "Referer": "https://iir.ia.org.hk/",
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    alphabets = "abcdefghijklmnopqrstuvwxyz"

    max_workers = 100  # tune this; start low to avoid 429
    total = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(fetch_one_letter, letter, token, BY_LETTER_DIR, headers)
            for letter in alphabets
        ]
        for f in as_completed(futures):
            total += f.result()

    print("Total saved records across letters:", total)
    print("Progress file lets you rerun with a new token and continue safely.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fetch_agents_parallel.py <token>")
        raise SystemExit(1)
    fetch_all_letters_parallel(sys.argv[1])

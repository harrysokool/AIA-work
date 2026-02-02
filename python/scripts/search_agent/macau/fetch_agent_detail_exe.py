import sys
import json
from pathlib import Path
import time
import csv
import asyncio
import aiohttp
import random
from typing import Any, Optional


class RetryableFetchError(Exception):
    """Signals a transient error that may succeed on retry."""


DETAIL_URL = "https://iiep.amcm.gov.mo/platform-enquiry-service/public/api/v1/web/enquiry/licenses/detail"
CONCURRENCY_LIMIT = 10
MAX_RETRIES = 5
SEM = asyncio.Semaphore(CONCURRENCY_LIMIT)

HEADERS: dict[str, str] = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://iiep.amcm.gov.mo",
    "Referer": "https://iiep.amcm.gov.mo/",
}

# Fixed company list as requested
COMPANY_LIST: set[str] = {"AIA INTERNATIONAL LIMITED"}


# ---- Helpers ----
def _norm_company_name(s: str) -> str:
    return (s or "").strip().upper()


def load_raw_data(raw_file: Path) -> list[dict] | None:
    try:
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        agents = data.get("content", [])
        if agents is None:
            print(f"[WARN] {raw_file}: 'content' is null")
            return None
        if not isinstance(agents, list):
            print(f"[WARN] {raw_file}: 'content' is not a list")
            return None
        return agents
    except Exception as e:
        print(f"[ERROR] Reading raw file {raw_file}: {e}")
        return None


def agent_has_company(detail_agent: dict, company_list: set[str]) -> bool:
    for corp in detail_agent.get("corporates") or []:
        for item in corp.get("items") or []:
            name_en = _norm_company_name(item.get("nameEn") or "")
            if name_en and name_en in company_list:
                return True
    return False


def extract_detail_params(agents: list[dict]) -> list[dict] | None:
    """
    Build params for detail calls using each agent's own licenseCategory + licenseNo.
    """
    if not agents:
        return None

    detail_params: list[dict] = []
    skipped = 0
    categories_seen: set[str] = set()

    for agent in agents:
        if not agent:
            continue

        cat = (agent.get("licenseCategory") or "").strip()
        license_no = (agent.get("licenseNo") or "").strip()

        if cat and license_no:
            categories_seen.add(cat)
            detail_params.append({"category": cat, "no": license_no})
        else:
            skipped += 1

    if skipped:
        print(f"[INFO] Skipped {skipped} agent(s) missing licenseCategory or licenseNo")

    if categories_seen:
        cats = ", ".join(sorted(categories_seen))
        print(f"[INFO] Categories found in file: {cats}")

    return detail_params or None


async def fetch(
    session: aiohttp.ClientSession, param: dict, company_list: set[str]
) -> dict | None:
    async with session.get(
        DETAIL_URL, params=param, headers=HEADERS, ssl=False
    ) as response:
        if response.status != 200:
            raise RetryableFetchError(f"HTTP {response.status}")

        if "application/json" not in (response.headers.get("Content-Type") or ""):
            raise RetryableFetchError(
                f"Unexpected content type: {response.headers.get('Content-Type')}"
            )

        detail = await response.json()

        if agent_has_company(detail, company_list):
            return detail
        return None


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    param: dict,
    company_list: set[str],
    retries: int = MAX_RETRIES,
) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            async with SEM:
                return await fetch(session, param, company_list)

        except aiohttp.ClientError as e:
            err = RetryableFetchError(f"Network error: {e}")
        except RetryableFetchError as e:
            err = e
        except Exception as e:
            err = RetryableFetchError(f"Unexpected error: {e}")

        print(f"[Attempt {attempt}] {err}")

        if attempt < retries:
            wait_time = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        else:
            print(f"Failed after {retries} attempts for {param}")
            return None


def flatten_agents_for_excel(detail_agents: list[dict]) -> list[dict]:
    """
    One row per agent-company relationship (per corporate->item).
    """
    rows: list[dict[str, Any]] = []
    for agent in detail_agents:
        base = {
            "licenseNo": agent.get("licenseNo"),
            "licenseCategory": agent.get("licenseCategory"),
            "namePt": agent.get("namePt"),
            "nameTc": agent.get("nameTc"),
            "status": agent.get("status"),
            "publicYear": agent.get("publicYear"),
            "publicMonth": agent.get("publicMonth"),
            "publishTime": agent.get("publishTime"),
        }

        for corp in agent.get("corporates") or []:
            items = corp.get("items") or []

            if not items:
                rows.append(
                    {
                        **base,
                        "permitType": corp.get("type"),
                        "corpStatus": corp.get("status"),
                        "corpStartDate": corp.get("startDate"),
                        "corpEndDate": corp.get("endDate"),
                        "companyNameEn": "",
                        "companyNameTc": "",
                        "associationDate": "",
                    }
                )
                continue

            for item in items:
                rows.append(
                    {
                        **base,
                        "permitType": corp.get("type"),
                        "corpStatus": corp.get("status"),
                        "corpStartDate": corp.get("startDate"),
                        "corpEndDate": corp.get("endDate"),
                        "companyNameEn": item.get("nameEn"),
                        "companyNameTc": item.get("nameTc"),
                        "associationDate": item.get("aDate"),
                    }
                )
    return rows


def write_csv(rows: list[dict], filepath: Path) -> None:
    if not rows:
        print("No rows to write.")
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig so Excel opens Chinese correctly
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written: {filepath}")


def write_excel(rows: list[dict], filepath: Path, sheet_name: str = "Agents") -> None:
    """
    Write rows (list[dict]) into an Excel .xlsx file using pandas + openpyxl.
    Auto-sets column widths and keeps column order from the first row.
    """
    if not rows:
        print("No rows to write.")
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Import locally to avoid import cost when not needed
    import pandas as pd

    # Preserve order from the first row's keys
    columns = list(rows[0].keys())
    df = pd.DataFrame(rows, columns=columns)

    # Write with openpyxl engine
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        # Auto-fit: approximate by using max length in column (works for ASCII/English; CJK may need tweaks)
        ws = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns, start=1):
            # Convert non-strings when measuring width
            series_as_str = df[col].astype(str)
            max_len = max([len(col)] + series_as_str.map(len).tolist())
            # Add padding
            ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = min(
                max_len + 2, 60
            )

    print(f"Excel written: {filepath}")


async def process_input_file(input_json: Path) -> None:
    """
    Reads a raw JSON (expects top-level {"content": [...]}) and fetches detail records,
    filtering by COMPANY_LIST. Uses each agent's own licenseCategory.
    Outputs written to <input_dir>/<input_stem>_out/.
    """
    t_start = time.perf_counter()
    print(f"\n=== Processing: {input_json} ===")

    agents = load_raw_data(input_json)
    if agents is None:
        print("[ERROR] No agents loaded; skipping.")
        return

    detail_params = extract_detail_params(agents)
    if detail_params is None:
        print("[WARN] No valid detail params found; skipping.")
        return

    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT, force_close=False)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            fetch_with_retry(session, param, COMPANY_LIST) for param in detail_params
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    detail_agents: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[TASK ERROR] {r}")
        elif r:
            detail_agents.append(r)

    out_dir = input_json.parent / f"{input_json.stem}_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_file = out_dir / f"{input_json.stem}_detail.json"
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(detail_agents, f, ensure_ascii=False, indent=2)
    print(f"Processed JSON written: {processed_file}")

    rows = flatten_agents_for_excel(detail_agents)
    csv_file = out_dir / f"{input_json.stem}.csv"
    write_csv(rows, csv_file)

    xlsx_file = out_dir / f"{input_json.stem}.xlsx"
    write_excel(rows, xlsx_file, sheet_name="Agents")

    print(f"Done. Matched {len(detail_agents)} agents, exported {len(rows)} rows")
    print(f"Time took: {time.perf_counter() - t_start:0.4f}s")


def resolve_inputs_from_argv() -> list[Path]:
    """
    Supports drag & drop onto .exe (paths in sys.argv[1:]) or running from terminal.
    Accepts any .json file; others are skipped.
    """
    if len(sys.argv) < 2:
        print(
            "Usage: Drag one or more JSON files onto the .exe, or run:\n"
            "  tool.exe <file1.json> <file2.json> ..."
        )
        return []

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".json":
            paths.append(p)
        else:
            print(f"[WARN] Skipping non-JSON or missing path: {arg}")
    return paths


async def main_async():
    inputs = resolve_inputs_from_argv()
    if not inputs:
        return

    # Process sequentially to keep logs tidy; switch to gather for parallel file-level processing if needed.
    for input_json in inputs:
        try:
            await process_input_file(input_json)
        except Exception as e:
            print(f"[FATAL] Failed on {input_json}: {e}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()

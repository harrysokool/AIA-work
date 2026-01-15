import requests
from pathlib import Path
import json
import time
import urllib3
import asyncio
import aiohttp
import random
import threading
import sys
from typing import Any, Dict, List, Optional, Set
import ssl


class RetryableFetchError(Exception):
    pass


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# url
# https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search/individualDetail?key=p8nH5v5Fj3I%3D&licStatus=all&token=hWA%252BY64JCYfHRKkqBSCZreZTNBjXM%252FkkUdSbf2%252FkJ%252FQ%253D
BASE_URL = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search/individual"
EMPTY_RESPONSE = '{"data":[],"itemsCount":"0","errorCode":null,"errorMsg":null}'

# directory/file paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "agents" / "hongkong"
ALL_FILE = DATA_DIR / "agents_keys.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "Accept": "application/json",
    "Referer": "https://iir.ia.org.hk/",
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}

PARAMS = {
    "seachIndicator": "licNo",
    "searchValue": "",
    "status": "all",
    "page": 1,
    "pagesize": 10,
    "token": "sessionToken",
}


def timer() -> None:
    counter = 1
    while True:
        time.sleep(1)
        print(counter)
        counter += 1


def checkLetter(letter: str) -> bool:
    if not letter or letter not in "abcdefghijklmnopqrstuvwxyz":
        return False
    return True


async def fetch(session, param):
    print("start fetching for ", param)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    try:
        async with session.get(
            BASE_URL, params=param, headers=HEADERS, ssl=ssl_context
        ) as response:
            if response.status != 200:
                raise RetryableFetchError(f"HTTP {response.status}")

            if "application/json" not in response.headers.get("Content-Type", ""):
                raise RetryableFetchError(
                    f"Unexpected content type: {response.headers.get('Content-Type')}"
                )

            detail = await response.json()

            if detail.get("itemsCount", 0) == 1:
                return detail
            else:
                return False

    except aiohttp.ClientError as e:
        raise RetryableFetchError(f"Network error: {e}")
    except Exception as e:
        raise RetryableFetchError(f"Unexpected error: {e}")


async def fetch_agent(sessionToken, licenseLetter):
    params = PARAMS.copy()
    params["token"] = sessionToken

    agent_params = []
    licenseLetter = licenseLetter.lower()
    test_param = params.copy()
    test_param["searchValue"] = "IA2145"
    agent_params.append(test_param)

    # for letter in "abcdefghijklmnopqrstuvwxyz":
    #     for i in range(10000):
    #         param = params.copy()
    #         num = f"{i:04d}"
    #         licence_number = licenseLetter + letter + num
    #         param["searchValue"] = licence_number
    #         agent_params.append(param)

    connector = aiohttp.TCPConnector(limit=10, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch(session, param) for param in agent_params]
        results = await asyncio.gather(*tasks)

    valid_agents = [r for r in results if r is not None]
    print(valid_agents)


if __name__ == "__main__":
    sessionToken = input("Enter Session Token: ")
    licenseLetter = input("Enter Licence Letter: ")

    tic = time.perf_counter()
    asyncio.run(
        fetch_agent(
            sessionToken,
            licenseLetter,
        )
    )
    toc = time.perf_counter()
    print(f"total time took: {toc - tic:0.4f}")

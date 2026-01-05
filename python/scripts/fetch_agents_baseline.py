import requests

s = requests.Session()
print('session created!')

# clear and set up cookies
s.cookies.clear()

# url for searching the agents
url = "https://iir.ia.org.hk/IISPublicRegisterRestfulAPI/v1/search/individual"

# need to udpate the token manually after solving the captchas
# also need to loop through the alphabet letters for the first name
# surNameValue, token both need update
# other params are fixed
token = "D3X25Y5JE10PkcuENFWXX89EzawhGSDbcTk6b7%2BjiiM%3D"
surName = "a"

params = {
    "seachIndicator": "engName",
    "status": "A",
    "surNameValue": surName,
    "givenNameValue": "",
    "page": 1,
    "pagesize": 10,
    "token": token
}

headers = {
    "Accept": "application/json",
    "Referer": "https://iir.ia.org.hk/",
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9"
}

resp = s.get(url, params=params, headers=headers, timeout=30)
obj = resp.json()
print(obj["itemsCount"])
print(obj["data"][0])

# AIA Automation

Automation utilities for:

- Insurance agent registry data (Hong Kong + Macau)
- HK registered doctor verification from receipt images (OCR + matching)

## Repo layout

- `python/` main scripts and data
- `python/data/` cached agent datasets and exports
- `python/scripts/search_agent/` agent registry tools
- `python/scripts/search_registered_doctors/` doctor OCR + verification pipeline

## Setup

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additional dependencies for the OCR pipeline (not listed in `requirements.txt` yet):

```bash
pip install aiohttp beautifulsoup4 lxml numpy opencv-python pytesseract
```

You also need the system Tesseract binary installed and available on your PATH.

## Scripts

### Hong Kong agent registry

- `python/scripts/search_agent/HongKong/fetch_agent_detail_by_letter.py`  
  Pull agent details by surname initial and status using a session token.
  Writes JSON to `python/data/agents/hongkong/agents_by_letter/`.

- `python/scripts/search_agent/HongKong/fetch_agent_key.py`  
  Probe for individual keys by license prefix and fetch details via the public API.

### Macau agent registry

- `python/scripts/search_agent/macau/fetch_agent_detail.py`  
  Reads raw list data from  
  `python/data/agents/macau/<category>/raw_agent_data/<category>.json`  
  Filters by company, saves processed JSON, and exports a CSV for Excel.

### HK registered doctors (OCR pipeline)

- `python/scripts/search_registered_doctors/fetch_doctor.py`  
  Scrapes the HKMC register and writes `doctors.pkl` + `doctors.txt`.

- `python/scripts/search_registered_doctors/image_processing.py`  
  Preprocesses receipt scans into OCR-friendly images.

- `python/scripts/search_registered_doctors/extract_doctors.py`  
  Extracts doctor names from a single preprocessed image.

- `python/scripts/search_registered_doctors/run_pipeline.py`  
  End-to-end pipeline: preprocess `doctor_receipts/`, run OCR, and verify
  against the HKMC list.

## Quick start (OCR pipeline)

```bash
cd python/scripts/search_registered_doctors
python fetch_doctor.py
python run_pipeline.py
```

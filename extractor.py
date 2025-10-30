import json
import pandas as pd
import ollama
from pathlib import Path

OCR_FILE = "ocr_output.json"  
OUTPUT_DIR = "output"
LLM_MODEL = "deepseek-r1:8b"  
PROMPT = """You are an AP automation system. Extract key invoice fields from OCR text.

OCR TEXT:
{ocr_text}

Extract these EXACT fields and return ONLY valid JSON:
{{
  "vendor": "string (seller/supplier company name)",
  "invoice_no": "string (invoice number)",
  "date": "YYYY-MM-DD (invoice date)",
  "amount": number (total gross amount including tax)",
  "tax": number (VAT/tax amount only)",
  "po_no": "string (PO number if mentioned, else null)"
}}
CRITICAL EXTRACTION RULES:
1. VENDOR = Seller/Supplier name (NOT Client/Customer)
   - Look for "Seller:" section
   - Usually listed BEFORE "Client:" section
   - Example: "Seller: Andrews, Kirby and Valdez" → vendor = "Andrews, Kirby and Valdez"
2. INVOICE_NO = Invoice number at the top
   - Look for "Invoice no:" or "Invoice #"
   - Remove any spaces/special chars
   - Example: "Invoice no: 51109338" → invoice_no = "51109338"
3. DATE = Invoice date in YYYY-MM-DD format
   - Look for "Date of issue:" or "Date:"
   - Convert from MM/DD/YYYY to YYYY-MM-DD
   - Example: "04/13/2013" → date = "2013-04-13"
4. AMOUNT = Total Gross Worth (including tax)
   - PRIORITY 1: Look for "SUMMARY" section → "Gross worth" or "Total"
   - PRIORITY 2: Sum all items "Gross worth" column
   - Must be HIGHER than net worth
   - Example: "Gross worth: $ 6 204,19" → amount = 6204.19
5. TAX = VAT/Tax amount ONLY (not rate %)
   - PRIORITY 1: Look for "SUMMARY" section → "VAT" column (not "VAT [%]")
   - PRIORITY 2: Calculate from items if SUMMARY missing
   - Example: "VAT: $ 564,02" → tax = 564.02
6. PO_NO = Purchase Order number (if exists)
   - Look for "PO:", "P.O.", "Purchase Order", "PO #", "PO No"
   - If not found → po_no = null
7. NUMBER FORMATTING:
   - Remove ALL spaces: "6 204,19" → "6204.19"
   - Convert comma to dot: "564,02" → "564.02"
   - Remove currency symbols: "$ 6 204,19" → "6204.19"
VALIDATION CHECKS:
- amount > tax
- vendor ≠ client

JSON OUTPUT:
"""

def process_invoice(filename: str, ocr_text: str) -> dict:
    print(f"Processing {filename}... ", end="", flush=True)
    try:
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=PROMPT.format(ocr_text=ocr_text[:3000]),
            format="json",
            options={"temperature": 0.1}
        )
        data = json.loads(response["response"])
        data["filename"] = filename
        data["status"] = "SUCCESS"
        print("")
        return data
    except Exception as e:
        print(f"{e}")
        return {"filename": filename, "status": "FAILED", "error": str(e)}
    
def main():
    print(f" Loading {OCR_FILE}...")
    with open(OCR_FILE, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    seen = {}
    for item in ocr_data:
        seen[item["filename"]] = item
    ocr_data = list(seen.values())

    print(f"Found {len(ocr_data)} unique invoices\n")

    results = []
    for item in ocr_data:
        result = process_invoice(item["filename"], item["ocr_text"])
        results.append(result)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    invoices = []
    for r in results:
        invoices.append({
            "po_no": r.get("po_no") or "",
            "vendor": r.get("vendor"),
            "invoice_no": r.get("invoice_no"),
            "date": r.get("date"),
            "tax": r.get("tax"),
            "amount": r.get("amount"),
        })

    df_invoices = pd.DataFrame(invoices)
    df_invoices.to_csv(f"{OUTPUT_DIR}/invoices.csv", index=False)

    print(f"\n Exported: {OUTPUT_DIR}/invoices.csv")
    print("\nPreview:")
    print(df_invoices.head(10).to_string(index=False))

    print("\n SUMMARY:")
    print(f"Total processed: {len(results)}")
    print(f"Success: {sum(1 for r in results if r['status'] == 'SUCCESS')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'FAILED')}")

    print("\n Sample extracted data:")
    print(df_invoices.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

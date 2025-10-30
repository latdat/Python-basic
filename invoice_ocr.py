import os
import logging
from paddleocr import PaddleOCR
import json
from datetime import datetime
from pathlib import Path

IMAGE_FOLDER = r"D:\Gia_su\BA\VLM\Image"
OUTPUT_FILE = "ocr_output.json"  
MIN_TEXT_LEN = 50                
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", "pdf")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logging.getLogger("ppocr").setLevel(logging.ERROR)

ocr = PaddleOCR(use_textline_orientation=True, lang="en")

def extract_full_text_from_image(image_path: str):
    try:
        result = ocr.predict(image_path)
        if not result or len(result) == 0:
            return "", "Empty OCR result"
        
        page = result[0]
        if hasattr(page, "json") and page.json:
            res = page.json.get("res", {})
            texts = res.get("rec_texts", [])
            if isinstance(texts, list) and texts:
                return "\n".join(texts), None
        return "", 

    except Exception as e:
        return "", f"OCR Error: {str(e)}"

def process_invoice(image_path: Path):
    filename = image_path.name
    print(f"Processing: {filename}...", end=" ", flush=True)

    text, error = extract_full_text_from_image(str(image_path))
    if error:
        print(f" {error}")
        return None, error

    if len(text) < MIN_TEXT_LEN:
        print(f" Too short ({len(text)} chars)")
        return None, "Extracted text too short"

    print(f"{len(text)} chars")

    return {
        "filename": filename,
        "ocr_text": text,
        "text_length": len(text),
        "num_lines": len(text.splitlines()),
        "timestamp": datetime.now().isoformat(),
    }, None

def main():
    image_files = sorted({
        p for p in Path(IMAGE_FOLDER).iterdir() if p.suffix.lower() in ALLOWED_EXTS
    })

    if not image_files:
        print(f" No images found in {IMAGE_FOLDER}")
        return

    print(f" Found {len(image_files)} unique images\n{'=' * 60}\n")

    results, errors = [], []

    for img_path in image_files:
        data, err = process_invoice(img_path)
        if data:
            results.append(data)
        if err:
            errors.append({"file": img_path.name, "error": err})

    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / OUTPUT_FILE
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f" Saved {len(results)} invoices → {json_path}")

    if results:
        total_chars = sum(r["text_length"] for r in results)
        avg_chars = total_chars / len(results)
        print("\n Statistics:")
        print(f"   Avg length: {avg_chars:.0f} chars")
        print(f"   Shortest: {min(r['text_length'] for r in results)}")
        print(f"   Longest: {max(r['text_length'] for r in results)}")

        first = results[0]
        print(f"\n Preview: {first['filename']}")
        print(f"   {first['text_length']} chars, {first['num_lines']} lines")
        print(f"\n   {first['ocr_text'][:300]}...")

    if errors:
        print(f"\n {len(errors)} errors (first 5 shown):")
        for err in errors[:5]:
            print(f"   {err['file']}: {err['error'][:100]}")

        error_path = base_dir / "ocr_errors.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"Saved error log → {error_path}")

    print(f"\n{'='*60}")
    print(f" Done! {len(results)}/{len(image_files)} successful")
    if results:
        print(f"\nNext step → Run LLM extractor on {OUTPUT_FILE}")
if __name__ == "__main__":
    main()

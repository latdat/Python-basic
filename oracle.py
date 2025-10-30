import json
import csv
import base64
from pathlib import Path
from datetime import datetime
import requests
from tqdm import tqdm

# ============ CẤU HÌNH ============
IMAGE_FOLDER = r"D:\Gia_su\BA\VLM\Image"
OUTPUT_CSV = "quiz_questions.csv"
OUTPUT_JSON = "quiz_raw_output.json"
ALLOWED_EXTS = (".jpg", ".jpeg", ".png")

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"

# ============ HÀM ENCODE ẢNH ============
def encode_image(image_path: Path) -> str:
    """Encode ảnh thành base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ============ HÀM EXTRACT QUIZ ============
def extract_quiz_from_image(image_path: Path) -> dict:
    """Gửi ảnh cho Ollama và extract thông tin quiz"""
    try:
        image_base64 = encode_image(image_path)
        
        prompt = """Extract quiz information from this image into JSON format.

Rules:
1. Identify type: "Multiple Choice" (○/● circles) or "Checkboxes" (☐/☑)
2. Extract complete question text
3. Extract all answer options
4. Find marked correct answers (● filled circles or ☑ checked boxes)
5. If none marked, use empty array []

Output ONLY this JSON (no markdown):
{
  "question": "complete question text",
  "type": "Multiple Choice",
  "options": ["A text", "B text", "C text", "D text"],
  "correct_answers": [0, 2]
}"""

        # ✅ GỌI API BẰNG REQUESTS
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 1000
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        response_text = response_data.get("response", "").strip()
        
        # Xử lý markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        # Validate
        required = ["question", "type", "options", "correct_answers"]
        if not all(k in result for k in required):
            raise ValueError(f"Missing keys. Got: {list(result.keys())}")
        
        return {
            "filename": image_path.name,
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except json.JSONDecodeError as e:
        return {
            "filename": image_path.name,
            "status": "error",
            "error": f"JSON decode error: {str(e)}",
            "raw_response": response_text if 'response_text' in locals() else ""
        }
    except requests.exceptions.RequestException as e:
        return {
            "filename": image_path.name,
            "status": "error",
            "error": f"API error: {str(e)}"
        }
    except Exception as e:
        return {
            "filename": image_path.name,
            "status": "error",
            "error": str(e),
            "raw_response": response_text if 'response_text' in locals() else ""
        }

# ============ CSV CONVERSION ============
def results_to_csv(results: list, csv_path: Path):
    """Chuyển JSON sang CSV với số cột động"""
    csv_rows = []
    max_options = max(
        (len(r["data"]["options"]) for r in results if r["status"] == "success"),
        default=0
    )
    
    if max_options == 0:
        return 0, 0
    
    fieldnames = ["Question", "Type"]
    fieldnames += [f"Option_{chr(65+i)}" for i in range(max_options)]
    fieldnames.append("Correct_Answers")
    
    for item in results:
        if item["status"] != "success":
            continue
        
        data = item["data"]
        row = {
            "Question": data.get("question", ""),
            "Type": data.get("type", "")
        }
        
        for i in range(max_options):
            col = f"Option_{chr(65+i)}"
            row[col] = data["options"][i] if i < len(data["options"]) else ""
        
        correct = data.get("correct_answers", [])
        row["Correct_Answers"] = ", ".join(chr(65+i) for i in correct) if correct else ""
        
        csv_rows.append(row)
    
    if csv_rows:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    
    return len(csv_rows), max_options

# ============ MAIN ============
def main():
    print("=" * 70)
    print(f"  QUIZ EXTRACTOR - {MODEL_NAME}")
    print("=" * 70)
    
    # Test Ollama connection
    try:
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        test_response.raise_for_status()
        
        # ✅ Kiểm tra model có tồn tại
        models = test_response.json().get("models", [])
        model_names = [m["name"] for m in models]
        
        if MODEL_NAME not in model_names and f"{MODEL_NAME}:latest" not in model_names:
            print(f"❌ Model '{MODEL_NAME}' not found!")
            print(f"   Available models: {', '.join(model_names[:5])}")
            print(f"   Run: ollama pull {MODEL_NAME}")
            return
        
        print("✓ Ollama connected")
        print(f"✓ Model '{MODEL_NAME}' found")
        
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure: ollama serve")
        return
    
    # Tìm ảnh
    image_files = sorted([
        p for p in Path(IMAGE_FOLDER).iterdir() 
        if p.suffix.lower() in ALLOWED_EXTS
    ])
    
    if not image_files:
        print(f"❌ No images in {IMAGE_FOLDER}")
        return
    
    print(f"\n📁 Found {len(image_files)} images\n")
    
    # Xử lý từng ảnh
    results = []
    for img in tqdm(image_files, desc="Processing"):
        results.append(extract_quiz_from_image(img))
    
    # Lưu JSON
    json_path = Path(OUTPUT_JSON)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Lưu CSV
    csv_path = Path(OUTPUT_CSV)
    successful = [r for r in results if r["status"] == "success"]
    csv_count, max_opts = results_to_csv(successful, csv_path)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"✓ Success: {len(successful)}/{len(results)}")
    print(f"✗ Errors: {len(results) - len(successful)}")
    print(f"\n📄 {json_path}")
    print(f"📊 {csv_path} ({csv_count} rows, {max_opts} options)")
    
    if len(successful) < len(results):
        print("\n⚠️  Errors:")
        for r in results:
            if r["status"] == "error":
                print(f"  • {r['filename']}: {r['error'][:60]}")
    
    if successful:
        preview = successful[0]["data"]
        print(f"\n📋 Preview: {preview['type']}")
        print(f"  Q: {preview['question'][:60]}...")
        print(f"  Options: {len(preview['options'])}")
        print(f"  Correct: {preview['correct_answers']}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
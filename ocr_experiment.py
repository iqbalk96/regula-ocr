
from pathlib import Path
import csv
import re
import time

import cv2
import easyocr
import numpy as np
import pymupdf

# ==============================
# KONFIGURASI
# ==============================
PDF_PATH = Path("data/dataset/2007_Peraturan_BPK_02.pdf")
PAGE_NUMBER = 1
OUTPUT_DIR = Path("output/experiment")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF tidak ditemukan: {PDF_PATH.resolve()}")

# ==============================
# FUNGSI EVALUASI
# ==============================
def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def edit_distance(a, b):
    # Levenshtein distance dengan memori O(min(len(a), len(b)))
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b)
            ))

        previous = current

    return previous[-1]


def cer(reference, hypothesis):
    ref = normalize(reference).replace(" ", "")
    hyp = normalize(hypothesis).replace(" ", "")

    if not ref:
        return None

    return edit_distance(ref, hyp) / len(ref)


def wer(reference, hypothesis):
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()

    if not ref:
        return None

    return edit_distance(ref, hyp) / len(ref)


# ==============================
# 1. BACA DAN VALIDASI PDF
# ==============================
doc = pymupdf.open(str(PDF_PATH))

if not 1 <= PAGE_NUMBER <= len(doc):
    doc.close()
    raise ValueError("Nomor halaman tidak valid.")

page = doc[PAGE_NUMBER - 1]
reference = page.get_text("text").strip()

print("PDF:", PDF_PATH.name)
print("Halaman:", PAGE_NUMBER)
print("Karakter referensi:", len(reference))

if len(reference) < 100:
    print("PERINGATAN: teks referensi sedikit.")
    print("Pastikan halaman ini sesuai untuk pengujian OCR.")

# Render halaman ke gambar resolusi 2x
pix = page.get_pixmap(
    matrix=pymupdf.Matrix(2, 2),
    alpha=False
)

image_path = OUTPUT_DIR / "01_original.png"
pix.save(str(image_path))
doc.close()

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError("Gambar PDF gagal dibaca.")

# Simpan referensi untuk pemeriksaan manual
(OUTPUT_DIR / "reference.txt").write_text(
    reference, encoding="utf-8"
)

# ==============================
# 2. PREPROCESSING OPENCV
# ==============================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# CLAHE: meningkatkan kontras lokal
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)
contrast = clahe.apply(gray)

# Adaptive thresholding
binary = cv2.adaptiveThreshold(
    contrast,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    11
)

images = {
    "original": image,
    "clahe": cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR),
    "adaptive_threshold": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
}

cv2.imwrite(str(OUTPUT_DIR / "02_clahe.png"), contrast)
cv2.imwrite(str(OUTPUT_DIR / "03_threshold.png"), binary)

# ==============================
# 3. JALANKAN OCR
# ==============================
print("\nMemuat EasyOCR...")
reader = easyocr.Reader(["id", "en"], gpu=False)

rows = []

for method, img in images.items():
    print(f"\nMenjalankan OCR: {method}")
    start = time.perf_counter()

    detections = reader.readtext(img, detail=1)

    elapsed = time.perf_counter() - start

    # EasyOCR mengembalikan bounding box, teks, confidence.
    # Urutan deteksi digunakan sebagai urutan pembacaan awal.
    texts = [item[1] for item in detections]
    result_text = "\n".join(texts)

    text_path = OUTPUT_DIR / f"result_{method}.txt"
    text_path.write_text(result_text, encoding="utf-8")

    char_error = cer(reference, result_text)
    word_error = wer(reference, result_text)

    rows.append({
        "method": method,
        "reference_chars": len(normalize(reference)),
        "recognized_chars": len(normalize(result_text)),
        "detected_lines": len(texts),
        "CER": round(char_error, 4) if char_error is not None else "",
        "WER": round(word_error, 4) if word_error is not None else "",
        "runtime_seconds": round(elapsed, 2),
    })

    print("Jumlah baris:", len(texts))
    print("CER:", rows[-1]["CER"])
    print("WER:", rows[-1]["WER"])
    print("Waktu:", round(elapsed, 2), "detik")

# ==============================
# 4. SIMPAN HASIL EVALUASI
# ==============================
csv_path = OUTPUT_DIR / "evaluation.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n========== RINGKASAN ==========")
for row in rows:
    print(
        f"{row['method']:20} "
        f"CER={row['CER']} "
        f"WER={row['WER']} "
        f"time={row['runtime_seconds']}s"
    )

print("\nOutput tersimpan di:", OUTPUT_DIR.resolve())
print("Evaluasi CSV:", csv_path.resolve())

from pathlib import Path

import cv2
import pymupdf
import easyocr

PDF_PATH = Path("data/dataset/2007_Peraturan_BPK_02.pdf")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Render halaman pertama PDF menjadi gambar
doc = pymupdf.open(str(PDF_PATH))
page = doc[0]

pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)

image_path = OUTPUT_DIR / "halaman_01.png"
pix.save(str(image_path))
doc.close()

# Inisialisasi OCR bahasa Indonesia dan Inggris
reader = easyocr.Reader(["id", "en"], gpu=False)

# Jalankan OCR
image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError("Gambar PDF gagal dibaca.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = reader.readtext(image_rgb, detail=1)

# Gabungkan teks hasil pengenalan
lines = [text for _, text, confidence in results]
recognized_text = "\n".join(lines)

# Simpan hasil
output_path = OUTPUT_DIR / "hasil_ocr_easy.txt"
output_path.write_text(recognized_text, encoding="utf-8")

print("Jumlah baris:", len(lines))
print("\n--- HASIL OCR ---\n")
print(recognized_text[:3000])
print("\nHasil tersimpan:", output_path.resolve())

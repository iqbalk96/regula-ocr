from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("data/dataset/2007_Peraturan_BPK_02.pdf")
reader = PdfReader(str(pdf_path))

total_chars = 0
pages_with_images = 0

print(f"File: {pdf_path.name}")
print(f"Jumlah halaman: {len(reader.pages)}\n")

for i, page in enumerate(reader.pages, start=1):
    text = (page.extract_text() or "").strip()

    try:
        image_count = len(page.images)
    except Exception:
        image_count = 0

    total_chars += len(text)

    if image_count > 0:
        pages_with_images += 1

    print(f"Halaman {i:02d} | Teks: {len(text):4d} karakter | Gambar: {image_count}")

print("\n--- Ringkasan ---")
print("Total karakter teks:", total_chars)
print("Halaman dengan objek gambar:", pages_with_images)
print("Total halaman:", len(reader.pages))

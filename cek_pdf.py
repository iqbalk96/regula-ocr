
from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("data/dataset/2007_Peraturan_BPK_02.pdf")

if not pdf_path.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {pdf_path.resolve()}")

reader = PdfReader(str(pdf_path))

print("Nama file:", pdf_path.name)
print("Jumlah halaman:", len(reader.pages))

for i, page in enumerate(reader.pages[:3], start=1):
    text = (page.extract_text() or "").strip()

    print(f"\n--- Halaman {i} ---")
    print("Jumlah karakter:", len(text))
    print(text[:300] or "[Tidak ada teks yang berhasil diekstrak]")

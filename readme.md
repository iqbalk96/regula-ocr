# Implementasi Computer Vision untuk Peningkatan Akurasi OCR pada Dokumen Peraturan Hukum Indonesia


## Abstrak

Dokumen peraturan hukum Indonesia umumnya memiliki struktur teks yang panjang dan memerlukan proses digitalisasi agar informasi di dalamnya lebih mudah dicari dan diolah. Salah satu pendekatan yang dapat digunakan untuk mengekstraksi teks dari dokumen berbasis gambar adalah *Optical Character Recognition* (OCR). Penelitian ini mengimplementasikan OCR menggunakan EasyOCR pada dokumen peraturan hukum Indonesia dan membandingkan hasil pengenalan teks dari tiga kondisi, yaitu gambar asli, peningkatan kontras menggunakan CLAHE, serta pemrosesan *adaptive thresholding*. Evaluasi dilakukan menggunakan *Character Error Rate* (CER) dan *Word Error Rate* (WER), dengan teks hasil ekstraksi PDF sebagai referensi awal. Pada halaman yang diuji, gambar asli menghasilkan CER 11,60% dan WER 23,97%. CLAHE menghasilkan CER 10,81% dan WER 20,55%, sedangkan adaptive thresholding menghasilkan CER 6,08% dan WER 21,23%. Hasil awal menunjukkan bahwa adaptive thresholding memperoleh CER terendah, sementara CLAHE memperoleh WER terendah. Namun, evaluasi masih terbatas pada satu halaman dan teks referensi belum diverifikasi secara manual, sehingga hasil belum dapat digeneralisasi ke seluruh dokumen.

**Kata kunci:** Computer Vision, OCR, EasyOCR, CLAHE, adaptive thresholding, CER, WER.

---

## 1. Pendahuluan

### 1.1 Latar Belakang

Digitalisasi dokumen merupakan salah satu cara untuk meningkatkan aksesibilitas informasi. Dokumen peraturan hukum sering kali tersedia dalam format PDF yang dapat berisi teks digital, hasil pemindaian, atau gabungan keduanya. Pada dokumen yang berupa gambar, isi teks tidak selalu dapat langsung dicari atau diproses oleh komputer. Teknologi *Optical Character Recognition* (OCR) dapat digunakan untuk mengenali karakter pada gambar dan mengubahnya menjadi teks digital.

Akurasi OCR dapat dipengaruhi oleh kualitas gambar, kontras, ketajaman karakter, serta kondisi latar belakang. Oleh karena itu, preprocessing citra dapat diterapkan sebelum proses OCR. Penelitian ini membandingkan hasil OCR pada gambar asli dengan dua variasi preprocessing, yaitu *Contrast Limited Adaptive Histogram Equalization* (CLAHE) dan *adaptive thresholding*.

### 1.2 Rumusan Masalah

1. Bagaimana menerapkan OCR pada halaman dokumen peraturan hukum Indonesia menggunakan EasyOCR?
2. Bagaimana pengaruh CLAHE dan adaptive thresholding terhadap hasil OCR?
3. Metode preprocessing mana yang menghasilkan nilai CER dan WER lebih rendah pada halaman yang diuji?

### 1.3 Tujuan

1. Mengimplementasikan pipeline OCR menggunakan Python dan EasyOCR.
2. Menerapkan preprocessing citra menggunakan CLAHE dan adaptive thresholding.
3. Membandingkan hasil OCR berdasarkan CER, WER, dan waktu pemrosesan.
4. Mengidentifikasi keterbatasan eksperimen sebagai dasar pengembangan berikutnya.

### 1.4 Manfaat

Eksperimen ini diharapkan dapat menjadi contoh penerapan computer vision untuk membantu digitalisasi dokumen hukum. Teks hasil OCR dapat menjadi tahap awal untuk pencarian isi peraturan, pengindeksan arsip, dan pengolahan dokumen secara otomatis. Hasil OCR tetap perlu diperiksa, terutama apabila digunakan untuk kebutuhan hukum yang menuntut ketepatan tinggi.

---

## 2. Dasar Teori

### 2.1 Optical Character Recognition

*Optical Character Recognition* (OCR) merupakan teknologi untuk mengenali karakter dari gambar dan mengubahnya menjadi teks digital. Pada penelitian ini, EasyOCR digunakan untuk mendeteksi dan mengenali teks pada citra halaman PDF. EasyOCR mendukung beberapa bahasa, termasuk bahasa Indonesia melalui konfigurasi bahasa yang digunakan pada program.

### 2.2 Preprocessing Citra

Preprocessing dilakukan untuk menyiapkan citra sebelum OCR. Penelitian ini menggunakan tiga kondisi masukan:

1. **Gambar asli (*original*):** citra halaman yang dirender dari PDF tanpa preprocessing tambahan.
2. **CLAHE:** teknik peningkatan kontras lokal yang membatasi penguatan histogram untuk mengurangi amplifikasi kontras yang berlebihan.
3. **Adaptive thresholding:** teknik mengubah citra menjadi hitam-putih dengan ambang yang dihitung berdasarkan wilayah lokal. Teknik ini dapat membantu memisahkan teks dari latar belakang ketika pencahayaan atau intensitas citra tidak merata.

### 2.3 Evaluasi OCR

Kinerja OCR dievaluasi menggunakan dua metrik:

- **Character Error Rate (CER):** proporsi operasi penyisipan, penghapusan, dan penggantian karakter dibandingkan dengan jumlah karakter pada teks referensi.
- **Word Error Rate (WER):** proporsi operasi penyisipan, penghapusan, dan penggantian kata dibandingkan dengan jumlah kata pada teks referensi.

Nilai CER dan WER yang lebih rendah menunjukkan hasil yang lebih dekat dengan teks referensi. Nilai metrik sangat bergantung pada kualitas dan ketepatan teks referensi. Dalam eksperimen ini, referensi diambil dari ekstraksi teks PDF dan belum diverifikasi secara menyeluruh secara manual.

---

## 3. Metodologi

### 3.1 Dataset

Dataset berasal dari koleksi `Azzindani/ID_REG` pada Hugging Face. Salah satu dokumen yang digunakan adalah `2007_Peraturan_BPK_02.pdf`, yaitu dokumen peraturan BPK tahun 2007. Dokumen tersebut terdiri dari 16 halaman. Pemeriksaan awal menunjukkan bahwa sebagian besar halaman memiliki teks yang dapat diekstraksi langsung dari PDF, sedangkan setidaknya satu halaman tidak menghasilkan teks ekstraksi dan memuat gambar. Eksperimen preprocessing dan OCR yang dilaporkan di sini dilakukan pada **halaman pertama** dokumen.

Karena dokumen memiliki teks digital sekaligus kemungkinan elemen berbasis gambar, dataset ini tidak boleh dianggap sebagai dokumen hasil pemindaian penuh. Penelitian ini berfokus pada pengujian pipeline OCR pada citra halaman PDF dan perbandingan preprocessing, bukan pada klaim bahwa seluruh dokumen merupakan hasil scan.

### 3.2 Alat dan Pustaka

Implementasi dilakukan menggunakan Python dengan pustaka berikut:

| Pustaka | Kegunaan |
|---|---|
| PyMuPDF (`fitz`) | Merender halaman PDF menjadi citra dan mengekstraksi teks referensi awal |
| OpenCV (`cv2`) | Melakukan preprocessing citra |
| EasyOCR | Mengenali teks pada citra |
| Pillow / NumPy | Mendukung pemrosesan citra sesuai kebutuhan lingkungan |
| `time` dan `csv` | Mengukur waktu proses dan menyimpan hasil evaluasi |

### 3.3 Alur Eksperimen

Tahapan penelitian adalah sebagai berikut:

1. Memuat dokumen PDF dan memilih halaman pertama.
2. Merender halaman PDF menjadi citra menggunakan PyMuPDF.
3. Menyimpan citra asli sebagai baseline.
4. Menghasilkan citra preprocessing menggunakan CLAHE.
5. Menghasilkan citra preprocessing menggunakan adaptive thresholding.
6. Menjalankan EasyOCR pada ketiga citra dengan konfigurasi bahasa Indonesia dan Inggris.
7. Menyimpan hasil pengenalan teks ke berkas teks.
8. Membandingkan hasil OCR dengan teks referensi awal menggunakan CER dan WER.
9. Mencatat waktu pemrosesan dan menyimpan hasil dalam CSV.

### 3.4 Skenario Pengujian

| Skenario | Input | Perlakuan |
|---|---|---|
| Original | Citra halaman PDF | Tanpa preprocessing tambahan |
| CLAHE | Citra halaman PDF | Peningkatan kontras lokal |
| Adaptive thresholding | Citra halaman PDF | Konversi ke citra biner berdasarkan ambang lokal |

Variabel yang dibandingkan adalah nilai CER, WER, dan waktu pemrosesan. Pengujian menggunakan halaman dan konfigurasi OCR yang sama agar hasil antar-skenario dapat dibandingkan secara lebih adil.

### 3.5 Lingkungan Pengujian

Eksperimen dijalankan pada lingkungan Python 3.12 dengan EasyOCR dan OpenCV. OCR dijalankan menggunakan CPU (`gpu=False`). Spesifikasi prosesor, RAM, sistem operasi, dan versi pasti seluruh pustaka perlu ditambahkan apabila diwajibkan oleh format laporan.

---

## 4. Implementasi

Pipeline diimplementasikan sebagai skrip Python. PyMuPDF digunakan untuk merender halaman PDF menjadi gambar. Selanjutnya, OpenCV membuat dua variasi citra preprocessing: CLAHE dan adaptive thresholding. Ketiga citra kemudian diproses dengan EasyOCR.

Hasil OCR disimpan secara terpisah agar teks dari setiap skenario dapat diperiksa. Program juga membuat teks referensi dari ekstraksi teks PDF dan menghitung CER serta WER. Ringkasan hasil disimpan ke `evaluation.csv`, sedangkan gambar dan teks hasil OCR disimpan di direktori `output/experiment`.

Struktur keluaran eksperimen adalah sebagai berikut:

```text
output/
└── experiment/
    ├── 01_original.png
    ├── 02_clahe.png
    ├── 03_threshold.png
    ├── reference.txt
    ├── result_original.txt
    ├── result_clahe.txt
    ├── result_adaptive_threshold.txt
    └── evaluation.csv
```

Nama berkas di atas menggambarkan keluaran yang diharapkan dari skrip eksperimen. Pastikan nama dan keberadaan setiap berkas sesuai dengan folder hasil aktual sebelum laporan dikumpulkan.

---

## 5. Hasil dan Pembahasan

### 5.1 Hasil Evaluasi

Berikut ringkasan hasil yang diperoleh dari satu halaman yang diuji.

| Metode | CER | WER | Waktu pemrosesan |
|---|---:|---:|---:|
| Original | 11,60% | 23,97% | 11,13 detik |
| CLAHE | 10,81% | **20,55%** | 12,87 detik |
| Adaptive thresholding | **6,08%** | 21,23% | 10,33 detik |

**Catatan:** nilai di atas merupakan hasil eksperimen awal. Waktu pemrosesan dapat berubah tergantung perangkat dan kondisi eksekusi. Referensi evaluasi berasal dari teks yang diekstraksi langsung dari PDF dan belum diverifikasi secara menyeluruh.

### 5.2 Analisis CER

Metode original memperoleh CER sebesar 11,60%, sedangkan CLAHE menghasilkan 10,81%. Adaptive thresholding memperoleh CER terendah, yaitu 6,08%. Dibandingkan baseline original, nilai CER adaptive thresholding turun sekitar 47,6% secara relatif. Hal ini menunjukkan bahwa pada halaman yang diuji, pemisahan teks dan latar belakang melalui ambang lokal membantu mengurangi kesalahan karakter.

### 5.3 Analisis WER

Metode original memperoleh WER sebesar 23,97%. CLAHE menghasilkan WER terendah, yaitu 20,55%, atau penurunan relatif sekitar 14,3% dibandingkan original. Adaptive thresholding menghasilkan WER sebesar 21,23%, lebih baik daripada original tetapi sedikit lebih tinggi daripada CLAHE.

Perbedaan peringkat CER dan WER dapat terjadi karena kesalahan karakter tidak selalu menghasilkan perubahan jumlah kata yang sebanding. Sebuah kata yang salah satu karakternya berubah dapat dihitung sebagai satu kesalahan kata pada WER, sementara CER memperhitungkan jumlah operasi karakter.

### 5.4 Analisis Waktu Pemrosesan

CLAHE membutuhkan waktu 12,87 detik, lebih lama dibandingkan original (11,13 detik) dan adaptive thresholding (10,33 detik). Pada pengujian ini, adaptive thresholding memperoleh CER terendah sekaligus waktu pemrosesan terpendek. Namun, hasil waktu dari satu kali pengujian belum cukup untuk menyimpulkan bahwa metode tersebut selalu paling cepat. Waktu dapat dipengaruhi oleh beban sistem, pemuatan model, dan variasi proses.

### 5.5 Pembahasan Umum

Tidak ada satu metode yang unggul pada seluruh metrik. Adaptive thresholding paling baik berdasarkan CER, sedangkan CLAHE paling baik berdasarkan WER. Pemilihan preprocessing perlu disesuaikan dengan kebutuhan aplikasi: apabila kesalahan karakter menjadi perhatian utama, hasil eksperimen ini mengarah pada adaptive thresholding; apabila kesalahan kata menjadi metrik utama, CLAHE memberikan hasil lebih baik.

Kesimpulan ini masih bersifat awal karena hanya satu halaman yang diuji dan referensi teks belum divalidasi manual. Ekstraksi teks PDF juga dapat memiliki artefak, misalnya pemenggalan kata atau susunan karakter yang tidak sesuai tampilan visual. Oleh sebab itu, sebagian nilai CER/WER dapat mencerminkan kesalahan pada teks referensi, bukan hanya kesalahan OCR.

---

## 6. Kesimpulan dan Saran

### 6.1 Kesimpulan

Berdasarkan eksperimen awal, OCR menggunakan EasyOCR dapat diterapkan pada citra halaman dokumen peraturan hukum Indonesia. Preprocessing citra menghasilkan perbedaan kinerja dibandingkan penggunaan citra original. Adaptive thresholding menghasilkan CER terendah sebesar 6,08%, sedangkan CLAHE menghasilkan WER terendah sebesar 20,55%. Dibandingkan citra original, kedua metode preprocessing memperbaiki metrik yang diuji, meskipun metode terbaik berbeda menurut metrik evaluasinya.

Hasil ini belum dapat digeneralisasi ke semua dokumen peraturan karena pengujian baru dilakukan pada satu halaman. Teks referensi juga masih berasal dari ekstraksi PDF dan perlu diperiksa sebelum dianggap sebagai *ground truth*.

### 6.2 Saran Pengembangan

1. Menguji lebih banyak halaman dan beberapa dokumen dengan kualitas citra berbeda.
2. Memeriksa dan memperbaiki teks referensi secara manual agar evaluasi CER dan WER lebih dapat dipercaya.
3. Mengulangi pengukuran waktu beberapa kali dan melaporkan rata-ratanya.
4. Menguji variasi resolusi render, grayscale, denoising, atau metode thresholding lain.
5. Menambahkan pemeriksaan manual pada hasil OCR sebelum digunakan untuk pengarsipan atau pencarian dokumen hukum.

---

## Daftar Pustaka

1. JaidedAI. (n.d.). *EasyOCR*. GitHub repository. https://github.com/JaidedAI/EasyOCR
2. OpenCV. (n.d.). *OpenCV Documentation*. https://docs.opencv.org/
3. PyMuPDF. (n.d.). *PyMuPDF Documentation*. https://pymupdf.readthedocs.io/
4. Hugging Face. (n.d.). *Azzindani/ID_REG dataset*. https://huggingface.co/datasets/Azzindani/ID_REG

> **Checklist sebelum pengumpulan**
>
> - Ganti seluruh identitas dalam tanda kurung siku.
> - Masukkan screenshot atau gambar hasil preprocessing bila diminta dosen.
> - Pastikan tabel hasil sama dengan `evaluation.csv`.
> - Verifikasi `reference.txt` dengan tampilan visual PDF dan jelaskan bahwa evaluasi awal memakai hasil ekstraksi otomatis.
> - Tambahkan spesifikasi perangkat jika disyaratkan.
> - Sesuaikan struktur dan gaya sitasi dengan pedoman kampus.

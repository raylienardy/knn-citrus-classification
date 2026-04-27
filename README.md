# 🍊 Citrus Classification using K-Nearest Neighbor (KNN)

Proyek ini merupakan tugas UAS mata kuliah Pengolahan Citra Digital yang bertujuan untuk mengklasifikasikan jenis jeruk berdasarkan fitur warna dan tekstur menggunakan algoritma K-Nearest Neighbor (KNN).

## 📌 Dataset

Dataset terdiri dari 3 kelas:

- Jeruk Lemon
- Jeruk Manis
- Jeruk Nipis

Eksperimen dilakukan dengan variasi jumlah data:

- 500 data per kelas
- 750 data per kelas
- 1000 data per kelas

Setiap dataset dibagi menjadi:

- Training (60%)
- Validation (20%)
- Testing (20%)

## ⚙️ Metode

1. Ekstraksi fitur:
   - Mean dan standar deviasi warna (RGB & HSV)
   - Fitur tekstur (grayscale, Laplacian, edge density)

2. Klasifikasi:
   - Algoritma K-Nearest Neighbor (KNN)

3. Evaluasi:
   - Confusion Matrix
   - Accuracy
   - Precision
   - Recall
   - Specificity
   - F1-Score

## 🔍 Pemilihan Parameter

Nilai K ditentukan menggunakan data validation dengan mencoba beberapa kandidat:

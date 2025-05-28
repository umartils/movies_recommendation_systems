# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Pada bagian ini, Kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Dalam merancang sistem rekomendasi film, pemahaman terhadap kebutuhan bisnis merupakan aspek krusial agar solusi yang dihasilkan tidak hanya unggul secara teknis, tetapi juga mampu menjawab tantangan nyata yang dihadapi pengguna. Tujuan utama dari sistem rekomendasi ini adalah untuk memperkaya pengalaman pengguna melalui rekomendasi film yang lebih tepat dan personal, serta mendukung platform streaming dalam meningkatkan tingkat retensi dan keterlibatan pengguna.

### Problem Statements

- Bagaimana langkah-langkah dalam mengenali serta mengumpulkan informasi dari data yang digunakan untuk membangun model sistem rekomendasi?
- Bagaimana proses perancangan model sistem rekomendasi dengan pendekatan *content-based filtering*?
- Bagaimana metode yang digunakan untuk membangun model rekomendasi menggunakan pendekatan *collaborative filtering*?
- Bagaimana cara mengevaluasi performa dari model sistem rekomendasi yang telah dibuat?

### Goals

- Melakukan analisis awal terhadap data serta visualisasi untuk memperoleh pemahaman mengenai struktur dan sifat dari dataset yang digunakan.
- Merancang sistem rekomendasi film dengan memanfaatkan pendekatan content-based filtering.
- Mengembangkan sistem rekomendasi film menggunakan metode collaborative filtering.
- Melakukan evaluasi terhadap performa model rekomendasi yang telah dibangun dengan menggunakan metrik evaluasi yang tepat.

### Solution Approach

Untuk menjawab rumusan masalah dan mencapai target yang telah ditetapkan, diperlukan strategi penyelesaian yang sistematis dan terorganisir, yang meliputi langkah-langkah berikut:

**Eksplorasi Data Awal:**
Tahapan pertama dari proyek ini adalah memahami karakteristik dataset melalui proses exploratory data analysis (EDA). Aktivitas ini mencakup analisis terhadap jumlah baris dan kolom, jenis data, distribusi nilai, serta visualisasi grafik yang dapat membantu dalam mengenali pola umum maupun anomali dalam data.

**Pengembangan Sistem Rekomendasi dengan Content-Based Filtering:**
Langkah ini diawali dengan proses pembersihan data (data cleaning) yang mencakup:

- Menghapus data duplikat dan *missing values* atau nilai kosong (NaN)
- Mengeliminasi kolom yang tidak relevan
- Melakukan pemrosesan terhadap data teks

Selanjutnya, dilakukan transformasi data seperti vektorisasi teks dengan metode TF-IDF dan perhitungan kesamaan antar item menggunakan cosine similarity. Tahap akhir adalah merancang fungsi rekomendasi serta melakukan pengujian prediksi.

**Pengembangan Sistem Rekomendasi dengan Collaborative Filtering:**
Proses ini juga dimulai dari pembersihan dan persiapan data, yang meliputi:

- Menghapus data yang duplikat dan tidak valid
- Melakukan encoding pada variabel kategorikal
- Membagi data menjadi set pelatihan dan pengujian (train-test split)

**Evaluasi Kinerja Model:**
Penilaian terhadap performa model dilakukan untuk memastikan efektivitas hasil yang diperoleh:

- Pada model content-based filtering, digunakan metrik seperti Precision untuk menilai akurasi dari rekomendasi yang dihasilkan.
- Sementara pada model collaborative filtering, digunakan metrik Root Mean Squared Error (RMSE) untuk mengukur tingkat kesalahan dalam prediksi rating pengguna.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

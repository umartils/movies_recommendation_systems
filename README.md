# Laporan Proyek Machine Learning - Umar Tilmisani

## Project Overview

Dalam era digital saat ini, jumlah konten hiburan seperti film dan serial televisi meningkat secara eksponensial. Platform penyedia layanan streaming seperti Netflix, Disney+, dan Amazon Prime Video menawarkan ribuan pilihan kepada pengguna. Namun, keberlimpahan pilihan ini justru dapat menimbulkan tantangan tersendiri bagi pengguna dalam memilih film yang sesuai dengan preferensi mereka. Oleh karena itu, dibutuhkan sistem yang mampu membantu pengguna untuk menemukan film yang relevan dan menarik secara otomatis dan personal. Sistem inilah yang dikenal sebagai sistem rekomendasi film (*movie recommendation system*).

Sistem rekomendasi merupakan cabang dari bidang *Information Retrieval* dan *Machine Learning* yang dirancang untuk memprediksi preferensi pengguna terhadap suatu item berdasarkan data historis. Dalam konteks film, sistem rekomendasi dapat menggunakan informasi seperti histori penilaian (rating), genre film, atau kesamaan perilaku antar pengguna. Dua pendekatan utama yang banyak digunakan dalam sistem rekomendasi adalah *content-based filtering* dan *collaborative filtering*. Pendekatan *content-based* memanfaatkan fitur dari film seperti genre dan deskripsi, sedangkan *collaborative filtering* menggunakan data perilaku pengguna lain yang memiliki kesamaan preferensi [1].

Netflix, misalnya, mengklaim bahwa lebih dari 75% film yang ditonton pengguna berasal dari sistem rekomendasi mereka [2]. Hal ini menunjukkan betapa pentingnya sistem ini dalam meningkatkan kepuasan pengguna, waktu tayang (watch time), serta loyalitas terhadap platform. Selain itu, sistem rekomendasi juga memainkan peran besar dalam strategi bisnis karena mampu meningkatkan engagement dan mengurangi tingkat berhenti berlangganan (*churn*).

Seiring perkembangan teknologi, algoritma yang digunakan dalam sistem rekomendasi pun semakin canggih, mulai dari pendekatan sederhana berbasis *k-Nearest Neighbors (k-NN)* hingga pendekatan berbasis *deep learning* seperti *neural collaborative filtering*. Oleh karena itu, membangun proyek sistem rekomendasi film menjadi salah satu tantangan sekaligus peluang yang menarik dalam bidang data science dan kecerdasan buatan.

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

*Dataset* yang digunakan pada proyek ini adalah data **"*MovieLens Latest Dataset*"** yang dapat diakses secara publik atau *open-source* pada situs web GroupLens. Untuk dataset sendiri dapat diakses dan diunduh melalui link [MovieLens Dataset by GroupLens](https://grouplens.org/datasets/movielens/). 
Terdapat beberapa file pada dataset ini seperti file `links.csv`, `movies.csv`, `ratings.csv`, dan `tags.csv`. Namun dataset yang digunakan pada proyek ini hanya dataset `movies.csv` dan `ratings.csv` dengan rincian dataset sebagai berikut.

- `movies.csv`: Berisi data film seperti judul film, identitas film dan genre film. Dataset ini akan digunakan untuk membangun model sistem rekomendasi dengan pendekatan *content-based filtering*
- `ratings.csv`: Berisi data rating film yang diberikan oleh pengguna seperti identitas pengguna, identitas film, dan rating film. Dataset ini akan digunakan untuk membangun model sistem rekomendasi dengan pendekatan *collaborative filtering*.

Selanjutnya, dilakukan tahap *Exploratory Data Analysis* (EDA)  untuk meningkatkan pemahaman mengenai data yang digunakan.

### Exploratory Data Analysis (EDA)

#### Fitur-fitur Pada Dataset

##### File: `movies.csv`

Dataset ini terdiri dari beberapa fitur sebagai berikut:

```py
RangeIndex: 9742 entries, 0 to 9741
Data columns (total 3 columns):
#    Column                     Non-Null Count  Dtype 
---  ------                     --------------  ----- 
0   movieId                     9742 non-null   int64 
1   title                       9742 non-null   object 
2   genres                      9742 non-null   object
dtypes: int64(1), object(2)
memory usage: 228.5+ KB
```

Dataset pada file `movies.csv` terdiri dari 3 kolom dan 9742 baris data. Dataset ini berisi informasi mengenai film seperti informasi id film, judul film dan juga genre film. Penjelasan setiap kolom dijelaskan sebagai berikut.

- `movieId`: Berisi data identitas dari film memiliki tipe data int64 (numerikal).
- `title`: Berisi data judul dari film memiliki tipe data object (kategorikal).
- `genres`: Berisi data genre dari film memiliki tipe data object (kategorikal).

##### File: `ratings.csv`

Dataset ini terdiri dari beberapa fitur sebagai berikut:

```py
RangeIndex: 100836 entries, 0 to 100835
Data columns (total 4 columns):
#    Column                     Non-Null Count  Dtype 
---  ------                     --------------  ----- 
0   userId                      100836 non-null  int64
1   movieId                     100836 non-null  int64
2   rating                      100836 non-null  float64
3   timestamp                   100836 non-null  int64
dtypes: int64(3), float64(1)
memory usage: 3.1 MB
```

Dataset pada file `ratings.csv` terdiri dari 4 kolom dan 100835 baris data. Dataset ini berisi informasi mengenai rating yang diberikan pada film seperti informasi id pengguna yang memberi rating, id film yang diberi rating, informasi rating yang diberikan pengguna, dan juga kolom waktu. Penjelasan setiap kolom dijelaskan sebagai berikut.

- `userId`: Berisi data identitas dari pengguna yang memberi rating film memiliki tipe data int64 (numerikal).
- `movieId`: Berisi data identitas dari film memiliki tipe data int64 (numerikal).
- `rating`: Berisi data rating dari film memiliki tipe data float64 (numerikal).
- `timestamp`: Berisi data waktu pada saat pengguna memberikan rating memiliki tipe data int64 (numerikal).

#### Melihat Ringkasan Data

##### File: `movies.csv`

| movieId | title                              | genres                                          |
|---------|------------------------------------|------------------------------------------------ |
| 1       | Toy Story (1995)                   | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 2       | Jumanji (1995)                     | Adventure\|Children\|Fantasy                    |
| 3       | Grumpier Old Men (1995)            | Comedy\|Romance                                 |
| 4       | Waiting to Exhale (1995)           | Comedy\|Drama\|Romance                          |
| 5       | Father of the Bride Part II (1995) | Comedy                                          |

Tabel ini berisi data mengenai film yang tersedia dalam sistem. Penjelasan tiap kolom:

- `movieId` adalah ID unik yang digunakan untuk mengidentifikasi setiap film. Nilai ini menjadi acuan utama dan sering digunakan untuk menghubungkan data antar tabel.
- `title` berisi judul film lengkap dengan tahun rilisnya dalam tanda kurung. Contohnya: *Toy Story (1995)*.
- `genres` adalah daftar kategori atau genre dari film tersebut. Genre dipisahkan dengan tanda "|" (pipe). Contoh: *Adventure|Animation|Children|Comedy|Fantasy*.

##### File: `ratings.csv`

| userId | movieId | rating | timestamp  |
|--------|---------|--------|------------|
| 1      | 1       | 4.0    | 964982703  |
| 1      | 3       | 4.0    | 964981247  |
| 1      | 6       | 4.0    | 964982224  |
| 1      | 47      | 5.0    | 964983815  |
| 1      | 50      | 5.0    | 964982931  |

Tabel ini menyimpan informasi mengenai rating (penilaian) yang diberikan oleh pengguna terhadap film. Penjelasan tiap kolom:

- `userId` menunjukkan ID unik dari pengguna yang memberikan rating.
- `movieId` merujuk pada film yang dinilai oleh pengguna, dan nilainya cocok dengan movieId di tabel film.
- `rating` menunjukkan nilai penilaian dari pengguna terhadap film, biasanya dalam skala 0.5 hingga 5.0.
- `timestamp` adalah waktu ketika rating diberikan, dalam format UNIX timestamp (jumlah detik sejak 1 Januari 1970).

#### Memeriksa Nilai Kosong dan Data Duplikat

Pada tahap ini dilakukan proses pemeriksaan nilai kosong pada data dan juga memeriksa data yang duplikat. Hal ini dilakukan agar dataset yang digunakan untuk modeling memiliki kualitas yang baik. Berikut proses dari tahap memeriksa nilai kosong dan data duplikat.

##### File: `movies.csv`

- Kode
  
    ```py
    
    movies = pd.read_csv("movies.csv")
    null_val = movies.isna().sum()
    duplicated_data = movies.duplicated().sum()
    print("=============================================")
    print(f"Jumlah Missing Values Pada Dataset Movies: \n{null_val}")
    print(f"Jumlah Data Duplikat Pada Dataset Movies: {duplicated_data}")
    print("=============================================")
    ```
- Output

    ```py
    =============================================
    Jumlah Missing Values Pada Dataset Movies: 
    movieId    0
    title      0
    genres     0
    dtype: int64
    Jumlah Data Duplikat Pada Dataset Movies: 0
    =============================================
    ```

##### File: `ratings.csv`

- Kode
  
    ```py
    
    ratings = pd.read_csv("ratings.csv")
    null_val = ratings.isna().sum()
    duplicated_data = ratings.duplicated().sum()
    print("=============================================")
    print(f"Jumlah Missing Values Pada Dataset Ratings: \n{null_val}")
    print(f"Jumlah Data Duplikat Pada Dataset Ratings: {duplicated_data}")
    print("=============================================")
    ```
- Output

    ```py
    =============================================
    Jumlah Missing Values Pada Dataset Ratings: 
    userId       0
    movieId      0
    rating       0
    timestamp    0
    dtype: int64
    Jumlah Data Duplikat Pada Dataset Ratings: 0
    =============================================
    ```

#### Analisis Distribusi Data

Pada tahap ini dilakukan visualisasi data untuk melihat distribusi nilai dari fitur yang akan digunakan untuk pemodelan pada setiap dataset. 

- Distribusi data **genres** pada file `movies.csv`
  
    <p align="center">
    <img src="https://github.com/user-attachments/assets/96884440-8397-4c6f-9412-556b65da4496" alt="genres" />
    </p><div align="center">Gambar 1 - Distribusi Data Genre</div>

    Berdasarkan visulisasi pada `Gambar 1`, terdapat beberapa informasi yang dijelaskan sebagai berikut.

    -  **Genre Terpopuler:**

        * Genre *Comedy* mendominasi dataset dengan total **2.779 film**, diikuti oleh *Drama* (**2.226 film**) dan *Action* (**1.828 film**).
        * Hal ini menunjukkan bahwa film bergenre komedi, drama, dan aksi adalah yang paling umum dalam dataset.

  -  **Genre Kurang Umum:**

        * Genre seperti *War*, *Film-Noir*, *Musical*, dan *Western* memiliki jumlah film yang sangat sedikit (di bawah 30 film).
        * Ini bisa berarti bahwa data yang tersedia untuk genre-genre ini terlalu sedikit untuk digunakan dalam pelatihan model sistem rekomendasi secara efektif.

  -  **Kehadiran Nilai Tidak Valid:**

        * Terlihat masih ada kategori `"(no genres listed)"` dengan **23 film** Sehingga perlu dilakukan proses pemilihan fitur agar menghapus genre yang tidak relevan dengan pembuatan model.
  
- Distribusi data **rating** pada file `ratings.csv`
    <p align="center">
    <img src="https://github.com/user-attachments/assets/cf61716f-87ef-4500-b05f-cc3eb8b7acde" alt="rating" />
    </p><div align="center">Gambar 2 - Distribusi Data Rating</div>

    Berdasarkan visulisasi pada `Gambar 2`, terdapat beberapa informasi yang dijelaskan sebagai berikut.



##### File: `movies.csv`

##### File: `ratings.csv`

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

### Data Cleaning

Pada tahap ini, dilakukan pembersihan data agar data yang digunakan memiliki kualitas yang baik. Pada tahap ini dilakukan tahap penghapusan nilai kosong dan data duplikat. Berikut penerapan dari tahap *data cleaning*.

- Menangani Missing Values dan Duplicated Data pada Dataset Movies
  
    **Kode:**
    ```py
    clean_movies = movies.dropna().drop_duplicates()
    clean_movies.info()
    ```

    **Output:**
    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9742 entries, 0 to 9741
    Data columns (total 3 columns):
    #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
    0   movieId  9742 non-null   int64 
    1   title    9742 non-null   object
    2   genres   9742 non-null   object
    dtypes: int64(1), object(2)
    memory usage: 228.5+ KB
    ```
- Menangani Missing Values dan Duplicated Data pada Dataset Ratings
  
    **Kode:**
    ```py
    clean_ratings = ratings.dropna().drop_duplicates()
    clean_ratings.info()
    ```
    **Output:**
    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100836 entries, 0 to 100835
    Data columns (total 4 columns):
    #   Column     Non-Null Count   Dtype  
    ---  ------     --------------   -----  
    0   userId     100836 non-null  int64  
    1   movieId    100836 non-null  int64  
    2   rating     100836 non-null  float64
    3   timestamp  100836 non-null  int64  
    dtypes: float64(1), int64(3)
    memory usage: 3.1 MB
    ```
### Pemilihan Fitur

Pada tahap ini dilakukan proses pemilihan fitur untuk membangun model sistem rekomendasi pada setiap dataset. Tahap ini dilakukan agar model yang dibangun memiliki performa baik. Karena tidak semua fitur pada data berisi informasi yang relevan dengan model sistem rekomendasi yang akan dibangun. Berikut proses pemilihan fitur pada setiap dataset.

- Pemilihan fitur pada dataset `movies.csv`
  
    **Kode:**
    ```py
    clean_movies = clean_movies[clean_movies['genres'] != '(no genres listed)'] 
    clean_movies['genres'].unique()
    ```
    **Output:**
    ```py
    array(['Adventure', 'Comedy', 'Action', 'Drama', 'Crime', 'Children',
    'Mystery', 'Animation', 'Documentary', 'Thriller', 'Horror',
    'Fantasy', 'Western', 'Film-Noir', 'Romance', 'Sci-Fi', 'Musical',
    'War'], dtype=object)
    ```
    Penjelasan:

    Pada dataset `movies.csv` dilakukan proses pemilihan fitur dengan menghapus nilai ***no_genres_listed*** pada kolom `genres`. Hal ini dilakukan untuk menghilangkan data yang tidak memiliki informasi genre yang berguna, sehingga analisis atau model yang dibangun nantinya tidak terpengaruh oleh entri yang tidak mengandung kategori genre yang valid. Dengan membersihkan data dari nilai seperti ***(no genres listed)***, kita memastikan bahwa setiap entri dalam kolom `genres` merepresentasikan minimal satu kategori genre yang dapat digunakan untuk keperluan klasifikasi, analisis statistik, atau visualisasi data.

    
### Text Processing

Pada tahap ini dilakukan proses pembersihan teks pada kolom `genres`. Hal ini dilakukan untuk menghilangkan karakter lain selain huruf. Sehingga setiap nilai pada kolom `genres` memiliki format yang seragam sehingga memudahkan model dalam membaca data. Berikut penerapan dari tahap *text processing*.

**Kode**
```py
clean_movies['genres'] = clean_movies['genres'].replace({'Sci-Fi':'Scifi', 'Film-Noir':'Filmnoir'})
print("Data untuk genre SciFi:")
print(clean_movies[clean_movies['genres'] == 'Scifi'].head())

print("\nData untuk genre Filmnoir:")
print(clean_movies[clean_movies['genres'] == 'Filmnoir'].head())
```
**Output**
```py
Data untuk genre SciFi:
      movieId                                  title genres
668       880       Island of Dr. Moreau, The (1996)  Scifi
1320     1779                          Sphere (1998)  Scifi
1719     2311  2010: The Year We Make Contact (1984)  Scifi
1902     2526                          Meteor (1979)  Scifi
2000     2661        It Came from Outer Space (1953)  Scifi

Data untuk genre Filmnoir:
      movieId                       title    genres
279       320               Suture (1993)  Filmnoir
695       913  Maltese Falcon, The (1941)  Filmnoir
711       930            Notorious (1946)  Filmnoir
913      1212       Third Man, The (1949)  Filmnoir
1531     2066      Out of the Past (1947)  Filmnoir
```
### Data Transformation
### Data Splitting

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


## Referensi

\[1] F. Ricci, L. Rokach, dan B. Shapira, *Recommender Systems Handbook*. Springer, 2011.

\[2] C. A. Gómez-Uribe dan N. Hunt, “The Netflix Recommender System: Algorithms, Business Value, and Innovation,” *ACM Trans. Manage. Inf. Syst.*, vol. 6, no. 4, pp. 1–19, Dec. 2015, doi: 10.1145/2843948.

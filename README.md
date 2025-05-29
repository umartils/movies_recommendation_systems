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

Selanjutnya, dilakukan transformasi data seperti vektorisasi teks dengan metode TF-IDF dan perhitungan kesamaan antar item menggunakan *cosine similarity*. Tahap akhir adalah merancang fungsi rekomendasi serta melakukan pengujian prediksi.

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

Dataset pada file `movies.csv` terdiri dari 3 kolom dan **9.742** baris data. Dataset ini berisi informasi mengenai film seperti informasi id film, judul film dan juga genre film. Penjelasan setiap kolom dijelaskan sebagai berikut.

- `movieId`: Berisi data identitas dari film memiliki tipe data int64 (numerikal).
- `title`: Berisi data judul dari film memiliki tipe data object (kategorikal).
- `genres`: Berisi data genre dari film memiliki tipe data object (kategorikal).

##### File: `ratings.csv`

Dataset ini terdiri dari beberapa fitur sebagai berikut:

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

Dataset pada file `ratings.csv` terdiri dari 4 kolom dan **100.836** baris data. Dataset ini berisi informasi mengenai rating yang diberikan pada film seperti informasi id pengguna yang memberi rating, id film yang diberi rating, informasi rating yang diberikan pengguna, dan juga kolom waktu. Penjelasan setiap kolom dijelaskan sebagai berikut.

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

| userId | movieId   | rating | timestamp  |
|--------|-----------|--------|------------|
| 1      | 1         | 4.0    | 964981247  |
| 1      | 110       | 4.0    | 964981247  |
| 1      | 158       | 4.0    | 964982224  |
| 1      | 260       | 4.5    | 964983815  |
| 1      | 356       | 5.0    | 964982931  |

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
    <img src="https://github.com/user-attachments/assets/d5cd5066-5168-46ae-aa70-ef94a995eea8" alt="genres" />
    </p><div align="center">Gambar 1 - Distribusi Data Genre</div>

    Berdasarkan visulisasi pada `Gambar 1`, terdapat beberapa informasi yang dijelaskan sebagai berikut.

    -  **Genre Terpopuler:**

        * Genre *Comedy* mendominasi dataset dengan total 2.779 film, diikuti oleh *Drama* (2.226 film) dan *Action* (1.828 film).
        * Hal ini menunjukkan bahwa film bergenre komedi, drama, dan aksi adalah yang paling umum dalam dataset.

  -  **Genre Kurang Umum:**

        * Genre seperti *War*, *Film-Noir*, *Musical*, dan *Western* memiliki jumlah film yang sangat sedikit (di bawah 30 film).
        * Ini bisa berarti bahwa data yang tersedia untuk genre-genre ini terlalu sedikit untuk digunakan dalam pelatihan model sistem rekomendasi secara efektif.

  -  **Kehadiran Nilai Tidak Valid:**

        * Terlihat masih ada kategori `"(no genres listed)"` dengan **23 film** Sehingga perlu dilakukan proses pemilihan fitur agar menghapus genre yang tidak relevan dengan pembuatan model.
  
- Distribusi data **rating** pada file `ratings.csv`
    <p align="center">
    <img src="https://github.com/user-attachments/assets/d3d9d25d-53c3-4cc9-b3b8-49f8149b4956" alt="rating" />
    </p><div align="center">Gambar 2 - Distribusi Data Rating</div>

    Berdasarkan visulisasi pada `Gambar 2`, terdapat beberapa informasi yang dijelaskan sebagai berikut.

    - **Rating 4.0 adalah yang paling umum diberikan**

        * Dengan jumlah lebih dari **25 ribu**, rating ini menunjukkan bahwa banyak pengguna cenderung memberikan penilaian yang tinggi.
        * Ini bisa menunjukkan bahwa sebagian besar film dinilai cukup baik oleh pengguna.

     - **Distribusi condong ke arah rating tinggi**

        * Setelah 4.0, rating 3.0 dan 5.0 juga mendapatkan jumlah yang sangat tinggi, masing-masing lebih dari 20 ribu dan 13 ribu.
        * Ini menunjukkan bahwa pengguna lebih sering memberikan rating **tengah hingga tinggi** (3.0–5.0) dibanding rating rendah.

     - **Rating rendah sangat jarang diberikan**

        * Rating seperti **0.5, 1.0, dan 1.5** memiliki jumlah yang jauh lebih kecil (di bawah 2 ribu), menunjukkan bahwa:

          * Entah pengguna enggan memberikan rating rendah,
          * Atau film yang sangat buruk relatif jarang dalam dataset.

     - **Distribusi tidak simetris**

        * Terlihat bahwa rating tidak tersebar merata; jumlah rating menurun secara signifikan setelah 4.0, menandakan distribusi **positif skew** (condong ke kiri).
        * Rating 2.0 ke bawah memiliki distribusi yang relatif rendah.

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

- Pemilihan fitur pada dataset `ratings.csv`
  
  - Menghapus fitur `timestamp`:
    **Kode:**
    ```py
    clean_ratings = clean_ratings.drop(columns=['timestamp'], axis=1)
    clean_ratings.info()
    ```
    **Output:**
    ```py
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100836 entries, 0 to 100835
    Data columns (total 3 columns):
    #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
    0   userId   100836 non-null  int64  
    1   movieId  100836 non-null  int64  
    2   rating   100836 non-null  float64
    dtypes: float64(1), int64(2)
    memory usage: 2.3 MB
    ```
  - Menghapus nilai **movieId** yang tidak ada pada file `movies.csv`

    **Kode:**
    ```py
    clean_ratings = clean_ratings[clean_ratings['movieId'].isin(clean_movies['movieId'])]
    print(f"Jumlah data rating setelah menghapus film tidak relevan: {len(clean_ratings)}\n")
    clean_ratings.info()
    ```
    **Output:**
    ```py
    Jumlah data rating setelah menghapus film tidak relevan: 100836

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100836 entries, 0 to 100835
    Data columns (total 3 columns):
    #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
    0   userId   100836 non-null  int64  
    1   movieId  100836 non-null  int64  
    2   rating   100836 non-null  float64
    dtypes: float64(1), int64(2)
    memory usage: 2.3 MB
    ```

    
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

Tahap transformasi data merupakan tahap untuk mengubah bentuk atau format data mentah menjadi data yang siap untuk digunakan pemodelan. Proses transformasi data sangat penting dilakukan agar model yang dibangun memiliki performa yang baik. Berikut beberapa tahapan yang akan dilakukan dalam proses transformasi data:

#### TF-IDF Vectorizer

Dalam membangun sistem *Content-Based Filtering*, salah satu tahap penting adalah merepresentasikan data fitur dari item dalam bentuk vektor numerik yang dapat dihitung kemiripannya. Pada dataset **`movies.csv`**, kolom `genres` merupakan fitur penting yang mencerminkan konten dari masing-masing **`movies.csv`**. Untuk mengubah informasi tekstual pada kolom `genres` menjadi representasi numerik yang bermakna, digunakan teknik TF-IDF (*Term Frequency* - *Inverse Document Frequency*).

Berikut rumus atau persamaan untuk mencari nilai dari TF-IDF

**1. Term Frequency (TF)**

Mengukur seberapa sering sebuah istilah $t$ muncul dalam sebuah dokumen $d$:

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{k} f_{k,d}}
$$

* $f_{t,d}$ = jumlah kemunculan term $t$ dalam dokumen $d$
* $\sum_{k} f_{k,d}$ = total semua term dalam dokumen $d$

**2. Inverse Document Frequency (IDF)**

Mengukur seberapa penting sebuah istilah $t$ secara global dalam seluruh dokumen:

$$
\text{IDF}(t) = \log \left( \frac{N}{n_t} \right)
$$

* $N$ = total jumlah dokumen
* $n_t$ = jumlah dokumen yang mengandung term $t$

> Catatan: Untuk menghindari pembagian dengan nol, sering digunakan versi modifikasi seperti:

$$
\text{IDF}(t) = \log \left( \frac{1 + N}{1 + n_t} \right) + 1
$$

**3. TF-IDF**

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

Dengan rumus ini, TF-IDF memberikan nilai tinggi untuk istilah yang sering muncul dalam satu dokumen tetapi jarang muncul di seluruh dokumen, menjadikannya ideal untuk mengenali kata-kata yang unik dan penting dalam konteks tertentu.

Berikut penerapan metode TF-IDF untuk mengubah nilai pada fitur `genres` menjadi vektor numerik.

**Kode:**
```py
tfidf = TfidfVectorizer()
genre_tfid_cbf = tfidf.fit_transform(clean_movies['genres']) 
genre_tfid_cbf.shape
```
**Output**
```py
(9708, 18)
```
Proses tersebut menghasilkan vektor numerik dengan ukuran (9708, 18) yang artinya terdapat 9708 baris film dengan 18 dimensi sesuai jumlah data unik dari genre

#### Cosine Similarity

*Cosine similarity* digunakan untuk mengukur kemiripan antar dua dokumen (atau *item*) berdasarkan nilai vektor TF-IDF mereka. Dalam sistem rekomendasi, *cosine similarity* akan mengukur seberapa mirip dua film (atau *item*) berdasarkan genre yang telah diubah menjadi vektor TF-IDF. Cara untuk mencari nilai *cosine similarity* dari setiap *item* adalah sebagai berikut.

Misalkan:

* $A$ = vektor TF-IDF dari dokumen pertama (film A)
* $B$ = vektor TF-IDF dari dokumen kedua (film B)

$$
\text{CosineSimilarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}
$$

* $A \cdot B$ = hasil dot product antara vektor A dan B
* $\|A\|$ = panjang (magnitudo) vektor A
* $\|B\|$ = panjang (magnitudo) vektor B

$$
\text{CosineSimilarity}(A, B) = \frac{\sum_{i=1}^{n} A_i \times B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

Setelah menghasilkan matrix TF-IDF untuk semua film berdasarkan genre, cosine similarity dapat digunakan untuk:

- Menentukan film mana yang paling mirip dengan film yang sedang ditonton pengguna.
- Membuat rekomendasi film berdasarkan kemiripan konten.

Berikut penerapan dari metode *cosine similarity*.

**Kode:**
```py
genre_cosin_cbf = cosine_similarity(genre_tfid_cbf)
genre_cosin_df = pd.DataFrame(genre_cosin_cbf, index=clean_movies['title'], columns=clean_movies['title'])
genre_cosin_df.sample(5, axis=1, random_state=42).sample(5, axis=0, random_state=42)
```

**Output:**
| title                              | Wrong Turn (2003) | Jewel of the Nile, The (1985) | Major Dundee (1965) | Vigilante Diaries (2016) | We Own the Night (2007) |
|------------------------------------|--------------------|-------------------------------|----------------------|---------------------------|--------------------------|
| Shattered (1991)                  | 0.0                | 0.0                           | 0.0                  | 0.0                       | 0.0                      |
| Dirty Work (1998)                 | 0.0                | 0.0                           | 0.0                  | 0.0                       | 0.0                      |
| Solaris (Solyaris) (1972)         | 0.0                | 0.0                           | 0.0                  | 0.0                       | 0.0                      |
| Mississippi Burning (1988)       | 0.0                | 0.0                           | 0.0                  | 0.0                       | 1.0                      |
| Late Marriage (Hatuna Meuheret) (2001) | 0.0           | 0.0                           | 0.0                  | 0.0                       | 0.0                      |

Setiap baris mewakili satu film, dan setiap kolom juga mewakili film lain. Nilai di dalam tabel menunjukkan tingkat kemiripan antara dua film berdasarkan genre mereka, dihitung menggunakan cosine similarity. Jika data bernilai 0, berarti tidak ada kemiripan antar film, dan jika data bernilai 1, terdapat kemiripan antar film dilihat dari genre kedua film tersebut

#### Data Encoding

Pada tahap ini dilakukan proses encoding data dengan mengubah data **userId** dan **movieId** pada dataset `ratings.csv` agar memiliki nilai yang berurutan untuk memudahkan dalam proses pemodelan. Berikut sampel data setelah dilakukan proses encoding.

|   | userId | movieId | rating | user | movies |
|---|--------|---------|--------|------|--------|
| 0 |   1    |    1    |  4.0   |  0   |   0    |
| 1 |   1    |    3    |  4.0   |  0   |   1    |
| 2 |   1    |    6    |  4.0   |  0   |   2    |
| 3 |   1    |   47    |  5.0   |  0   |   3    |
| 4 |   1    |   50    |  5.0   |  0   |   4    |

Berdasarkan tabel di atas, data yang dilakukan encoding adalah data **userId** dan **movieId**. Data **userId** diubah dan disimpan ke dalam kolom **user** lalu data **movieId** diubah dan disimpan ke dalam kolom **movies**. Kedua kolom hasil encoding tersebut nantinya akan digunakan sebagai fitur untuk melatih model sistem rekomendasi. Hal tersebut dilakukan untuk memudahkan mesin dalam membaca data sehingga model yang dibangun memiliki performa yang baik.

### Data Splitting

*Data splitting* adalah proses memisahkan dataset menjadi dua bagian utama, yaitu **data latih (*training set*)** dan **data uji (*testing set*)**. Tujuan dari proses ini adalah untuk mengevaluasi kinerja model secara objektif terhadap data yang belum pernah dilihat sebelumnya. Rasio yang umum digunakan dalam pemisahan data adalah **80:20**, artinya:

- **80%** dari total data digunakan sebagai **data latih** untuk melatih model machine learning.
- **20%** sisanya digunakan sebagai **data uji** untuk menguji performa model.

Proses ini sangat penting agar model tidak hanya mengingat data pelatihan (*overfitting*), tetapi juga mampu melakukan generalisasi dengan baik terhadap data baru.

## Modeling

Pada proyek ini, sistem rekomendasi dibangun dengan dua pendekatan utama, yaitu *Content-Based Filtering* dan *Collaborative Filtering*, masing-masing menggunakan dataset yang berbeda untuk menghasilkan rekomendasi yang relevan bagi pengguna.

### *Content-Based Filtering*

#### Penjelasan *Content-Based Filtering*
Pendekatan Content-Based Filtering difokuskan pada karakteristik atau fitur dari item yang direkomendasikan, dalam hal ini genre dari film. Dataset yang digunakan adalah `movies.csv`, yang berisi informasi mengenai judul film dan genre yang dimiliki.

Langkah pertama dalam pendekatan ini adalah melakukan pra-pemrosesan data genre, di mana genre yang semula disimpan dalam bentuk string dipisahkan dan diubah menjadi format vektor menggunakan teknik *Text Vectorization* seperti TF-IDF (*Term Frequency-Inverse Document Frequency*). Representasi ini memungkinkan sistem untuk memahami keterkaitan antar film berdasarkan kemiripan genre.

Selanjutnya, untuk menghasilkan rekomendasi, digunakan metode *cosine similarity* untuk mengukur tingkat kemiripan antar film berdasarkan vektor genre yang telah dibentuk. Ketika seorang pengguna menonton atau memberikan rating tinggi pada sebuah film, sistem akan mencari film lain yang memiliki kemiripan genre tertinggi dan menyarankannya kepada pengguna.

#### Penerapan *Content-Based Filtering*

Berikut penerapan untuk membuat sistem rekomendasi dengan pendekatan *Content-Based Filtering*.
```py
def movies_recommendations(nama_movies, similarity_data=genre_cosin_df, items=clean_movies[['title', 'genres']], k=5):
    index = similarity_data.loc[:,nama_movies].to_numpy().argpartition(
        range(-1, -k, -1))

    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(nama_movies, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)
```
Pada kode tersebut dibangun suatu sistem rekomendasi dengan mencari *item* yang memiliki nilai *cosine similarity* yang sama dengan inputan atau parameter `nama_movies` di dalam sebuah fungsi `movies_recommendation`. Selanjutnya sistem akan menghapus *item* yang memiliki nama yang sama dengan inputan, agar *output* dihasilkan oleh sistem hanya menampilkan item lain yang serupa saja.

### *Collaborative Filtering*

#### Penjelasan *Collaborative Filtering*
Pendekatan Collaborative Filtering dalam proyek ini menggunakan data histori interaksi pengguna terhadap film yang terdapat dalam dataset `ratings.csv`. Dataset ini mencakup informasi tentang ID pengguna, ID film, dan rating yang diberikan oleh pengguna terhadap film tersebut.

Alih-alih menggunakan teknik tradisional seperti *Matrix Factorization* atau *SVD*, sistem ini menerapkan pendekatan *deep learning* untuk mempelajari hubungan kompleks antara pengguna dan film berdasarkan histori rating mereka. Model yang digunakan merupakan arsitektur jaringan neural sederhana yang menerima ID pengguna dan ID film sebagai input, yang kemudian diubah menjadi representasi vektor melalui *embedding layer*.

Layer embedding ini bertugas memetakan ID pengguna dan ID film ke dalam ruang laten berdimensi lebih rendah yang dapat menangkap fitur-fitur tersembunyi dari preferensi pengguna dan karakteristik film. Vektor-vektor ini kemudian digabungkan dan diteruskan ke beberapa layer dense (*fully connected*) untuk mempelajari interaksi non-linear di antara fitur tersebut.

Model dilatih untuk memprediksi rating yang akan diberikan pengguna terhadap suatu film, menggunakan *binary crossentropy* sebagai *loss function*, dengan penyesuaian pada skala rating. Untuk evaluasi performa model, digunakan metrik *Root Mean Squared Error* (RMSE) untuk mengukur seberapa akurat prediksi dibandingkan dengan rating aktual.

Pendekatan berbasis *deep learning* ini memungkinkan sistem untuk menangkap hubungan yang lebih kompleks antara pengguna dan item, serta memiliki kemampuan generalisasi yang lebih baik dalam kondisi data yang sparse.

#### Penerapan *Collaborative Filtering*
Berikut proses pembuatan sistem rekomendasi dengan pendekatan *Collaborative Filtering* menggunakan algoritma deep learning
```py
class RecommenderNet(tf.keras.Model):

  def __init__(self, num_users, num_movies, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_movies = num_movies
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( 
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) 
    self.movies_embedding = layers.Embedding(
        num_movies,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.movies_bias = layers.Embedding(num_movies, 1) 

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) 
    user_bias = self.user_bias(inputs[:, 0])
    movies_vector = self.movies_embedding(inputs[:, 1]) 
    movies_bias = self.movies_bias(inputs[:, 1]) 

    dot_user_movies = tf.tensordot(user_vector, movies_vector, 2)

    x = dot_user_movies + user_bias + movies_bias

    return tf.nn.sigmoid(x)
```

Pembuatan sistem rekomendasi tersebut menggunakan model `RecommenderNet` yaitu model *deep learning* yang dirancang khusus untuk membangun sistem rekomendasi dengan pendekatan *collaborative filtering* dan juga merupakan *subclass* dari `tf.keras.Model` atau model kostum dari TensorFlow/Keras. Model ini dibangun menggunakan framework TensorFlow dan Keras, dengan pendekatan Embedding Layer untuk merepresentasikan pengguna dan film ke dalam ruang dimensi laten. Arsitektur ini memanfaatkan dot product untuk menghitung skor prediksi interaksi antara pengguna dan film, dengan fungsi aktivasi sigmoid untuk membatasi output antara 0 dan 1. Proses kerja secara singkat dari model ini adalah sebagai berikut.
- Input: pasangan [user_id, movie_id]
- Embedding: mengonversi user dan movie ID menjadi vektor
- Dot product: menggabungkan vektor user dan movie untuk menghasilkan skor prediksi
- Penyesuaian Bias: menambahkan bias pengguna dan film
- Output: skor prediksi dalam rentang 0–1 sebagai estimasi rating atau probabilitas user menyukai film

Selanjutnya dilakukan proses pelatihan (*training*) model dengan menggunakan arsitektur **RecommenderNet** yang telah didefinisikan sebelumnya:

```python
model = RecommenderNet(jumlah_users, jumlah_movies, 50)  # Inisialisasi model

# Kompilasi model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# Memulai proses training
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8,
    epochs=30,
    validation_data=(x_val, y_val)
)
```

Model dilatih selama **30 epoch** dengan **batch size sebesar 8**, menggunakan *loss function* **Binary Crossentropy** dan *optimizer* **Adam** dengan *learning rate* 0.001. Sebagai metrik evaluasi, digunakan **Root Mean Squared Error (RMSE)** untuk mengukur seberapa jauh prediksi model dari nilai sebenarnya. Selama proses pelatihan, model belajar memetakan hubungan antara pengguna dan film berdasarkan histori rating yang tersedia.
## Evaluation

Tahapan evaluasi bertujuan untuk mengukur kinerja model dalam memberikan rekomendasi kepada pengguna. Evaluasi dilakukan setelah model dibangun dan dilakukan pengujian untuk melihat performa dari model. Untuk mengukur kinerja dari sistem rekomendasi yang dikembangkan, digunakan dua metrik evaluasi yang berbeda, sesuai dengan pendekatan yang digunakan. Berikut metrik yang digunakan untuk melakukan evaluasi terhadap dua pendekatan yang digunakan untuk membangun sistem rekomendasi.

### Metrik Precision (*Content-Based Filtering*)

Pada pendekatan Content-Based Filtering, sistem merekomendasikan sejumlah film yang memiliki kesamaan konten (dalam hal ini genre) dengan film yang disukai oleh pengguna. Untuk mengevaluasi kualitas rekomendasi ini, digunakan metrik **Precision\@K**.

**Precision\@K** mengukur **proporsi item yang relevan** (misalnya, yang benar-benar disukai atau ditonton oleh pengguna) dari **K item yang direkomendasikan**. Berikut persamaan atau rumus untuk menghitung nilai **Precision\@K**

#### Rumus
$$
\text{Precision@K} = \frac{|\text{Relevant Items} \cap \text{Recommended Items}@K|}{K}
$$

* **Relevant Items**: Film yang benar-benar relevan (misalnya film yang diberi rating tinggi oleh pengguna).
* **Recommended Items\@K**: Daftar top-K film yang direkomendasikan oleh sistem.

Precision\@K bernilai antara 0 dan 1, di mana nilai yang lebih tinggi menunjukkan bahwa rekomendasi sistem lebih akurat dalam menyarankan item relevan.

#### Hasil Evaluasi
Berikut adalah hasil evaluasi dan perhitungan metrik **Precision\@K** menggunakan model *content-based filtering* yang telah dibangun.

Pertama, pengguna memasukkan film `Toy Story (1995)` sebagai input. Berdasarkan data, film ini memiliki genre *Adventure*:

```python
clean_movies[clean_movies['title'] == 'Toy Story (1995)']
```

|   | movieId | title            | genres    |
| - | ------- | ---------------- | --------- |
| 0 | 1       | Toy Story (1995) | Adventure |

Kemudian sistem menghasilkan 5 rekomendasi film yang memiliki genre serupa:

```python
movies_recommendations('Toy Story (1995)')
```

|   | title                    | genres    |
| - | ------------------------ | --------- |
| 0 | Touching the Void (2003) | Adventure |
| 1 | Over the Hedge (2006)    | Adventure |
| 2 | RV (2006)                | Adventure |
| 3 | Shaggy Dog, The (2006)   | Adventure |
| 4 | Pink Panther, The (2006) | Adventure |

Dari hasil rekomendasi di atas, dapat dilihat bahwa semua film yang direkomendasikan memiliki genre yang sama, yaitu *Adventure*, yang sesuai dengan genre film input. Oleh karena itu, nilai **Precision\@K** dapat dihitung sebagai berikut:

$$
\text{Precision@5} = \frac{\text{Jumlah film relevan}}{\text{Jumlah film yang direkomendasikan}} = \frac{5}{5} = 1
$$

Dengan demikian, model content-based filtering yang digunakan menunjukkan performa yang sangat baik dalam memberikan rekomendasi yang relevan, dengan nilai **Precision\@5** sebesar **1.0**, yang berarti seluruh film yang direkomendasikan sesuai dengan karakteristik genre dari film yang dijadikan referensi.

### Metrik RMSE (*Collaborative Filtering*)
Untuk pendekatan Collaborative Filtering berbasis deep learning, sistem bertugas memprediksi **rating** yang akan diberikan pengguna terhadap film. Oleh karena itu, metrik evaluasi yang digunakan adalah *Root Mean Squared Error* (RMSE), yang mengukur seberapa dekat prediksi sistem terhadap rating aktual. Berikut persamaan atau rumus untuk mencari nilai RMSE

#### Rumus

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (\hat{r}_i - r_i)^2 }
$$

* $\hat{r}_i$: Rating yang diprediksi oleh model
* $r_i$: Rating aktual dari pengguna
* $n$: Jumlah total prediksi

RMSE bernilai lebih kecil jika prediksi sistem semakin mendekati rating aktual. Nilai RMSE yang rendah menandakan bahwa sistem memiliki performa prediksi yang baik.

#### Hasil Evaluasi

Model *collaborative filtering* yang dikembangkan menggunakan pendekatan deep learning menghasilkan nilai evaluasi atau RMSE Sebagai berikut:

- **RMSE Training**: 0.1457
- **RMSE Test**: 0.2264

Nilai RMSE yang relatif rendah pada data training dan test menunjukkan bahwa model mampu mempelajari pola preferensi pengguna terhadap film dengan cukup baik, serta memiliki **generalisasi yang cukup baik** pada data yang tidak pernah dilihat sebelumnya (test set).


## Referensi

\[1] F. Ricci, L. Rokach, dan B. Shapira, *Recommender Systems Handbook*. Springer, 2011.

\[2] C. A. Gómez-Uribe dan N. Hunt, “The Netflix Recommender System: Algorithms, Business Value, and Innovation,” *ACM Trans. Manage. Inf. Syst.*, vol. 6, no. 4, pp. 1–19, Dec. 2015, doi: 10.1145/2843948.

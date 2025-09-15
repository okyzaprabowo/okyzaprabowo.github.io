# Materi Pembelajaran: Vektor dan Matriks untuk Machine Learning

## Daftar Isi
1. [Pengantar](#pengantar)
2. [Vektor](#vektor)
3. [Matriks](#matriks)
4. [Operasi Vektor](#operasi-vektor)
5. [Operasi Matriks](#operasi-matriks)
6. [Aplikasi dalam Machine Learning](#aplikasi-dalam-machine-learning)
7. [Latihan Soal dengan Python](#latihan-soal-dengan-python)

## Pengantar

Aljabar linear adalah cabang matematika yang mempelajari vektor, ruang vektor, transformasi linear, dan sistem persamaan linear. Dalam konteks machine learning, aljabar linear menjadi fondasi matematis yang sangat penting karena:

- Data direpresentasikan dalam bentuk vektor dan matriks
- Algoritma machine learning menggunakan operasi matriks untuk komputasi yang efisien
- Teknik optimasi berbasis gradien menggunakan konsep aljabar linear
- Reduksi dimensi dan transformasi data menggunakan dekomposisi matriks

## Vektor

### Definisi
Vektor adalah besaran yang memiliki magnitude (besar) dan direction (arah). Dalam konteks matematika, vektor dapat direpresentasikan sebagai array satu dimensi yang berisi sekumpulan bilangan real.

### Notasi Matematis
Vektor biasanya ditulis dengan huruf kecil dan dicetak tebal atau dengan tanda panah:
- **v** atau v⃗
- Dalam bentuk kolom: v = [v₁, v₂, ..., vₙ]ᵀ
- Dalam bentuk baris: v = [v₁, v₂, ..., vₙ]

### Jenis-jenis Vektor

#### 1. Vektor Baris (Row Vector)
```
v = [1, 2, 3, 4]
```

#### 2. Vektor Kolom (Column Vector)
```
v = [1]
    [2]
    [3]
    [4]
```

#### 3. Vektor Nol (Zero Vector)
```
0 = [0, 0, 0, ..., 0]
```

#### 4. Vektor Unit (Unit Vector)
Vektor dengan magnitude = 1
```
e₁ = [1, 0, 0]  # unit vector dalam arah x
e₂ = [0, 1, 0]  # unit vector dalam arah y
e₃ = [0, 0, 1]  # unit vector dalam arah z
```

### Properti Vektor

#### Magnitude (Norm)
Magnitude atau norm dari vektor v adalah:
```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

#### Normalisasi
Proses mengubah vektor menjadi unit vector:
```
v̂ = v / ||v||
```

## Matriks

### Definisi
Matriks adalah susunan bilangan-bilangan dalam bentuk persegi panjang yang terdiri dari baris dan kolom. Matriks berukuran m×n memiliki m baris dan n kolom.

### Notasi Matematis
Matriks biasanya ditulis dengan huruf kapital:
```
A = [a₁₁  a₁₂  ...  a₁ₙ]
    [a₂₁  a₂₂  ...  a₂ₙ]
    [⋮    ⋮    ⋱   ⋮  ]
    [aₘ₁  aₘ₂  ...  aₘₙ]
```

### Jenis-jenis Matriks

#### 1. Matriks Persegi (Square Matrix)
Matriks dengan jumlah baris = jumlah kolom (n×n)

#### 2. Matriks Identitas (Identity Matrix)
Matriks persegi dengan diagonal utama = 1 dan elemen lainnya = 0
```
I = [1  0  0]
    [0  1  0]
    [0  0  1]
```

#### 3. Matriks Nol (Zero Matrix)
Matriks dengan semua elemen = 0

#### 4. Matriks Diagonal
Matriks persegi dengan elemen non-nol hanya pada diagonal utama

#### 5. Matriks Simetris
Matriks persegi dimana A = Aᵀ (A sama dengan transposenya)

#### 6. Matriks Ortogonal
Matriks persegi dimana A·Aᵀ = I

## Operasi Vektor

### 1. Penjumlahan dan Pengurangan Vektor
```
u + v = [u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ]
u - v = [u₁ - v₁, u₂ - v₂, ..., uₙ - vₙ]
```

### 2. Perkalian Skalar
```
c·v = [c·v₁, c·v₂, ..., c·vₙ]
```

### 3. Dot Product (Perkalian Titik)
```
u·v = u₁v₁ + u₂v₂ + ... + uₙvₙ = Σᵢ uᵢvᵢ
```

Properti dot product:
- u·v = v·u (komutatif)
- u·(v + w) = u·v + u·w (distributif)
- (cu)·v = c(u·v)
- u·u = ||u||²

### 4. Cross Product (Perkalian Silang)
Hanya untuk vektor 3D:
```
u × v = [u₂v₃ - u₃v₂, u₃v₁ - u₁v₃, u₁v₂ - u₂v₁]
```

### 5. Proyeksi Vektor
Proyeksi vektor u pada vektor v:
```
proj_v(u) = (u·v / ||v||²) · v
```

## Operasi Matriks

### 1. Penjumlahan dan Pengurangan Matriks
```
(A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ
(A - B)ᵢⱼ = Aᵢⱼ - Bᵢⱼ
```

### 2. Perkalian Skalar
```
(cA)ᵢⱼ = c·Aᵢⱼ
```

### 3. Perkalian Matriks
Untuk matriks A(m×n) dan B(n×p):
```
(AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
```

Syarat: jumlah kolom A = jumlah baris B

### 4. Transpose Matriks
```
(Aᵀ)ᵢⱼ = Aⱼᵢ
```

Properti transpose:
- (Aᵀ)ᵀ = A
- (A + B)ᵀ = Aᵀ + Bᵀ
- (AB)ᵀ = BᵀAᵀ

### 5. Determinan
Untuk matriks 2×2:
```
det(A) = a₁₁a₂₂ - a₁₂a₂₁
```

Untuk matriks 3×3:
```
det(A) = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)
```

### 6. Invers Matriks
Untuk matriks A yang memiliki invers:
```
A·A⁻¹ = A⁻¹·A = I
```

Syarat: det(A) ≠ 0

Untuk matriks 2×2:
```
A⁻¹ = (1/det(A)) · [a₂₂  -a₁₂]
                    [-a₂₁  a₁₁]
```

### 7. Rank Matriks
Rank adalah jumlah maksimum baris atau kolom yang linear independent.

### 8. Trace Matriks
Trace adalah jumlah elemen diagonal utama:
```
tr(A) = Σᵢ Aᵢᵢ
```

## Aplikasi dalam Machine Learning

### 1. Representasi Data
- **Dataset**: Matriks X dengan baris = sampel, kolom = fitur
- **Label**: Vektor y dengan elemen = target value
- **Parameter Model**: Vektor w (weights) dan skalar b (bias)

### 2. Linear Regression
```
ŷ = Xw + b
```
Solusi optimal: w = (XᵀX)⁻¹Xᵀy

### 3. Principal Component Analysis (PCA)
- Hitung matriks kovarians: C = (1/n)XᵀX
- Cari eigenvalues dan eigenvectors dari C
- Proyeksikan data ke principal components

### 4. Neural Networks
- Forward propagation: z = Wx + b, a = σ(z)
- Backpropagation: menggunakan turunan matriks

### 5. Similarity Measures
- Cosine similarity: cos(θ) = (u·v)/(||u||·||v||)
- Euclidean distance: d = ||u - v||

### 6. Dimensionality Reduction
- SVD: A = UΣVᵀ
- Matrix factorization untuk recommendation systems




## Latihan Soal dengan Python

Berikut adalah 10 contoh soal latihan beserta solusinya menggunakan Python dan NumPy:

```python
#!/usr/bin/env python3
"""
Latihan Soal Vektor dan Matriks dengan Python
Menggunakan NumPy untuk operasi aljabar linear
"""

import numpy as np

print("=" * 60)
print("LATIHAN SOAL VEKTOR DAN MATRIKS DENGAN PYTHON")
print("=" * 60)

# ============================================================================
# SOAL 1: Operasi Dasar Vektor
# ============================================================================
print("\n" + "="*50)
print("SOAL 1: Operasi Dasar Vektor")
print("="*50)

print("Diberikan dua vektor:")
u = np.array([3, 4, 5])
v = np.array([1, 2, 3])
print(f"u = {u}")
print(f"v = {v}")

print("\nHitung:")
print("a) u + v")
hasil_a = u + v
print(f"   Hasil: {hasil_a}")

print("b) u - v")
hasil_b = u - v
print(f"   Hasil: {hasil_b}")

print("c) 3u")
hasil_c = 3 * u
print(f"   Hasil: {hasil_c}")

print("d) Dot product u·v")
hasil_d = np.dot(u, v)
print(f"   Hasil: {hasil_d}")

print("e) Magnitude ||u||")
hasil_e = np.linalg.norm(u)
print(f"   Hasil: {hasil_e:.4f}")

print("f) Unit vector û")
hasil_f = u / np.linalg.norm(u)
print(f"   Hasil: {hasil_f}")

# ============================================================================
# SOAL 2: Cross Product dan Proyeksi Vektor
# ============================================================================
print("\n" + "="*50)
print("SOAL 2: Cross Product dan Proyeksi Vektor")
print("="*50)

print("Diberikan dua vektor 3D:")
a = np.array([2, 1, 3])
b = np.array([1, 4, 2])
print(f"a = {a}")
print(f"b = {b}")

print("\nHitung:")
print("a) Cross product a × b")
cross_product = np.cross(a, b)
print(f"   Hasil: {cross_product}")

print("b) Proyeksi vektor a pada b")
proj_a_on_b = (np.dot(a, b) / np.dot(b, b)) * b
print(f"   Hasil: {proj_a_on_b}")

print("c) Sudut antara vektor a dan b (dalam derajat)")
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta_rad = np.arccos(cos_theta)
theta_deg = np.degrees(theta_rad)
print(f"   Hasil: {theta_deg:.2f}°")

# ============================================================================
# SOAL 3: Operasi Matriks Dasar
# ============================================================================
print("\n" + "="*50)
print("SOAL 3: Operasi Matriks Dasar")
print("="*50)

print("Diberikan dua matriks:")
A = np.array([[2, 3], [1, 4]])
B = np.array([[5, 1], [2, 3]])
print(f"A = \n{A}")
print(f"B = \n{B}")

print("\nHitung:")
print("a) A + B")
hasil_add = A + B
print(f"   Hasil:\n{hasil_add}")

print("b) A - B")
hasil_sub = A - B
print(f"   Hasil:\n{hasil_sub}")

print("c) A × B (perkalian matriks)")
hasil_mul = np.dot(A, B)
print(f"   Hasil:\n{hasil_mul}")

print("d) A^T (transpose A)")
hasil_transpose = A.T
print(f"   Hasil:\n{hasil_transpose}")

print("e) det(A) (determinan A)")
hasil_det = np.linalg.det(A)
print(f"   Hasil: {hasil_det}")

print("f) A^(-1) (invers A)")
hasil_inv = np.linalg.inv(A)
print(f"   Hasil:\n{hasil_inv}")

# ============================================================================
# SOAL 4: Sistem Persamaan Linear
# ============================================================================
print("\n" + "="*50)
print("SOAL 4: Sistem Persamaan Linear")
print("="*50)

print("Selesaikan sistem persamaan linear:")
print("2x + 3y = 7")
print("x + 4y = 6")
print("\nDalam bentuk matriks: Ax = b")

A_sys = np.array([[2, 3], [1, 4]])
b_sys = np.array([7, 6])
print(f"A = \n{A_sys}")
print(f"b = {b_sys}")

print("\nSolusi menggunakan np.linalg.solve:")
x_solution = np.linalg.solve(A_sys, b_sys)
print(f"x = {x_solution}")
print(f"Jadi x = {x_solution[0]:.4f}, y = {x_solution[1]:.4f}")

print("\nVerifikasi: A × x = b")
verification = np.dot(A_sys, x_solution)
print(f"A × x = {verification}")
print(f"b = {b_sys}")
print(f"Sama? {np.allclose(verification, b_sys)}")

# ============================================================================
# SOAL 5: Eigenvalues dan Eigenvectors
# ============================================================================
print("\n" + "="*50)
print("SOAL 5: Eigenvalues dan Eigenvectors")
print("="*50)

print("Diberikan matriks:")
C = np.array([[4, 2], [1, 3]])
print(f"C = \n{C}")

print("\nHitung eigenvalues dan eigenvectors:")
eigenvalues, eigenvectors = np.linalg.eig(C)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

print("\nVerifikasi: C × v = λ × v untuk setiap eigenvector")
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_val = eigenvalues[i]
    Cv = np.dot(C, v)
    lambda_v = lambda_val * v
    print(f"\nEigenvector {i+1}: {v}")
    print(f"C × v = {Cv}")
    print(f"λ × v = {lambda_v}")
    print(f"Sama? {np.allclose(Cv, lambda_v)}")

# ============================================================================
# SOAL 6: Dekomposisi Matriks (SVD)
# ============================================================================
print("\n" + "="*50)
print("SOAL 6: Singular Value Decomposition (SVD)")
print("="*50)

print("Diberikan matriks:")
D = np.array([[3, 2, 2], [2, 3, -2]])
print(f"D = \n{D}")

print("\nLakukan SVD: D = U × Σ × V^T")
U, sigma, Vt = np.linalg.svd(D)
print(f"U = \n{U}")
print(f"Σ (singular values) = {sigma}")
print(f"V^T = \n{Vt}")

print("\nRekonstruksi matriks dari SVD:")
# Untuk matriks non-persegi, perlu menyesuaikan dimensi sigma
Sigma = np.zeros((U.shape[1], Vt.shape[0]))
Sigma[:len(sigma), :len(sigma)] = np.diag(sigma)
D_reconstructed = np.dot(U, np.dot(Sigma, Vt))
print(f"D (rekonstruksi) = \n{D_reconstructed}")
print(f"Sama dengan D asli? {np.allclose(D, D_reconstructed)}")

# ============================================================================
# SOAL 7: Aplikasi dalam Linear Regression
# ============================================================================
print("\n" + "="*50)
print("SOAL 7: Linear Regression dengan Aljabar Linear")
print("="*50)

print("Dataset sederhana untuk regresi linear:")
# Data: y = 2x + 1 + noise
np.random.seed(42)
X_data = np.array([[1], [2], [3], [4], [5]])
y_data = np.array([3.1, 4.9, 7.2, 8.8, 11.1])

print(f"X = \n{X_data}")
print(f"y = {y_data}")

print("\nMenambahkan bias term (kolom 1s):")
X_with_bias = np.column_stack([np.ones(X_data.shape[0]), X_data])
print(f"X dengan bias = \n{X_with_bias}")

print("\nMenghitung parameter optimal: w = (X^T × X)^(-1) × X^T × y")
XtX = np.dot(X_with_bias.T, X_with_bias)
XtX_inv = np.linalg.inv(XtX)
Xty = np.dot(X_with_bias.T, y_data)
w_optimal = np.dot(XtX_inv, Xty)

print(f"w = {w_optimal}")
print(f"Bias (w0) = {w_optimal[0]:.4f}")
print(f"Slope (w1) = {w_optimal[1]:.4f}")

print("\nPrediksi:")
y_pred = np.dot(X_with_bias, w_optimal)
print(f"y_pred = {y_pred}")

print("\nMean Squared Error:")
mse = np.mean((y_data - y_pred)**2)
print(f"MSE = {mse:.4f}")

# ============================================================================
# SOAL 8: Principal Component Analysis (PCA) Manual
# ============================================================================
print("\n" + "="*50)
print("SOAL 8: Principal Component Analysis (PCA) Manual")
print("="*50)

print("Dataset 2D untuk PCA:")
np.random.seed(42)
# Membuat data yang berkorelasi
data_original = np.random.randn(100, 2)
# Transformasi untuk membuat korelasi
transform_matrix = np.array([[2, 1], [1, 1]])
data_correlated = np.dot(data_original, transform_matrix.T)

print(f"Shape data: {data_correlated.shape}")
print(f"Mean data: {np.mean(data_correlated, axis=0)}")
print(f"Std data: {np.std(data_correlated, axis=0)}")

print("\nLangkah 1: Standarisasi data (mean=0, std=1)")
data_mean = np.mean(data_correlated, axis=0)
data_std = np.std(data_correlated, axis=0)
data_scaled = (data_correlated - data_mean) / data_std

print("\nLangkah 2: Hitung matriks kovarians")
cov_matrix = np.cov(data_scaled.T)
print(f"Matriks kovarians:\n{cov_matrix}")

print("\nLangkah 3: Hitung eigenvalues dan eigenvectors")
eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")

print("\nLangkah 4: Urutkan berdasarkan eigenvalues")
idx = np.argsort(eigenvals)[::-1]
eigenvals_sorted = eigenvals[idx]
eigenvecs_sorted = eigenvecs[:, idx]
print(f"Eigenvalues (sorted): {eigenvals_sorted}")

print("\nLangkah 5: Proyeksi ke principal components")
# Ambil 1 komponen utama
pc1 = eigenvecs_sorted[:, 0]
data_pca = np.dot(data_scaled, pc1.reshape(-1, 1))
print(f"Data setelah PCA (1 komponen): shape = {data_pca.shape}")

print(f"\nVariance explained ratio:")
variance_ratio = eigenvals_sorted / np.sum(eigenvals_sorted)
print(f"PC1: {variance_ratio[0]:.4f}")
print(f"PC2: {variance_ratio[1]:.4f}")

# ============================================================================
# SOAL 9: Cosine Similarity
# ============================================================================
print("\n" + "="*50)
print("SOAL 9: Cosine Similarity untuk Rekomendasi")
print("="*50)

print("Matriks rating pengguna-item (baris=pengguna, kolom=item):")
ratings = np.array([
    [5, 3, 0, 1],  # User 1
    [4, 0, 0, 1],  # User 2
    [1, 1, 0, 5],  # User 3
    [1, 0, 0, 4],  # User 4
    [0, 1, 5, 4]   # User 5
])
print(f"Ratings:\n{ratings}")

print("\nHitung cosine similarity antara User 1 dan User 2:")
user1 = ratings[0]
user2 = ratings[1]

print(f"User 1 ratings: {user1}")
print(f"User 2 ratings: {user2}")

# Cosine similarity = (u·v) / (||u|| × ||v||)
dot_product = np.dot(user1, user2)
norm_user1 = np.linalg.norm(user1)
norm_user2 = np.linalg.norm(user2)
cosine_sim = dot_product / (norm_user1 * norm_user2)

print(f"\nDot product: {dot_product}")
print(f"||User 1||: {norm_user1:.4f}")
print(f"||User 2||: {norm_user2:.4f}")
print(f"Cosine similarity: {cosine_sim:.4f}")

print("\nHitung similarity matrix untuk semua pengguna:")
n_users = ratings.shape[0]
similarity_matrix = np.zeros((n_users, n_users))

for i in range(n_users):
    for j in range(n_users):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            user_i = ratings[i]
            user_j = ratings[j]
            # Hindari pembagian dengan nol
            norm_i = np.linalg.norm(user_i)
            norm_j = np.linalg.norm(user_j)
            if norm_i > 0 and norm_j > 0:
                similarity_matrix[i, j] = np.dot(user_i, user_j) / (norm_i * norm_j)

print(f"Similarity matrix:\n{similarity_matrix}")

# ============================================================================
# SOAL 10: Matrix Factorization untuk Recommendation System
# ============================================================================
print("\n" + "="*50)
print("SOAL 10: Matrix Factorization dengan SVD")
print("="*50)

print("Menggunakan SVD untuk matrix factorization pada data rating:")
print(f"Original ratings matrix:\n{ratings}")

print("\nLakukan SVD pada matriks rating:")
U_rating, sigma_rating, Vt_rating = np.linalg.svd(ratings, full_matrices=False)

print(f"U shape: {U_rating.shape}")
print(f"Sigma shape: {sigma_rating.shape}")
print(f"Vt shape: {Vt_rating.shape}")
print(f"Singular values: {sigma_rating}")

print("\nReduksi dimensi dengan mengambil k=2 komponen utama:")
k = 2
U_k = U_rating[:, :k]
sigma_k = sigma_rating[:k]
Vt_k = Vt_rating[:k, :]

print(f"U_k shape: {U_k.shape}")
print(f"Sigma_k: {sigma_k}")
print(f"Vt_k shape: {Vt_k.shape}")

print("\nRekonstruksi matriks dengan k=2 komponen:")
ratings_reconstructed = np.dot(U_k, np.dot(np.diag(sigma_k), Vt_k))
print(f"Reconstructed ratings:\n{ratings_reconstructed}")

print("\nPerbandingan dengan matriks asli:")
print("Original vs Reconstructed:")
for i in range(ratings.shape[0]):
    for j in range(ratings.shape[1]):
        original = ratings[i, j]
        reconstructed = ratings_reconstructed[i, j]
        print(f"({i+1},{j+1}): {original:.1f} -> {reconstructed:.2f}")

print("\nMean Squared Error rekonstruksi:")
mse_recon = np.mean((ratings - ratings_reconstructed)**2)
print(f"MSE: {mse_recon:.4f}")

print("\n" + "="*60)
print("SELESAI - SEMUA SOAL TELAH DIKERJAKAN")
print("="*60)
```


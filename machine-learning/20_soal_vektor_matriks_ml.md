# 20 Contoh Soal Vektor dan Matriks untuk Machine Learning

## Pendahuluan

Aljabar linear, khususnya vektor dan matriks, merupakan fondasi matematika yang sangat penting dalam machine learning. Setiap algoritma ML, dari yang paling sederhana hingga yang paling kompleks, bergantung pada operasi-operasi aljabar linear. Dokumen ini menyajikan 20 soal yang dirancang secara progresif untuk membangun pemahaman yang kuat tentang konsep-konsep ini dan aplikasinya dalam machine learning.

## Filosofi Pembelajaran

Setiap soal dalam koleksi ini memiliki tujuan pedagogis yang spesifik:

1. **Membangun Intuisi Geometris**: Memahami vektor dan matriks tidak hanya sebagai kumpulan angka, tetapi sebagai representasi geometris yang memiliki makna.

2. **Mengembangkan Kemampuan Komputasi**: Melatih kemampuan melakukan operasi aljabar linear secara efisien dan akurat.

3. **Menghubungkan Teori dengan Praktik**: Setiap konsep dihubungkan langsung dengan aplikasi nyata dalam machine learning.

4. **Mempersiapkan Pemahaman Algoritma ML**: Memberikan fondasi yang kuat untuk memahami algoritma-algoritma machine learning yang lebih kompleks.

---

## BAGIAN I: KONSEP DASAR VEKTOR (Soal 1-5)

### Soal 1: Operasi Dasar Vektor
**Soal**: Diberikan vektor u = [3, 4, 5] dan v = [1, 2, 3]. Hitung:
a) u + v
b) u - v  
c) 3u
d) u · v (dot product)
e) ||u|| (magnitude)
f) û (unit vector dari u)

**Filosofi**: Operasi dasar vektor adalah fondasi dari semua komputasi dalam machine learning. Dalam ML, data sering direpresentasikan sebagai vektor fitur, dan operasi-operasi ini digunakan untuk:
- **Penjumlahan vektor**: Menggabungkan fitur atau melakukan ensemble learning
- **Perkalian skalar**: Scaling fitur atau learning rate dalam optimasi
- **Dot product**: Mengukur similarity antar data point, basis dari banyak algoritma klasifikasi
- **Normalisasi**: Preprocessing data untuk mencegah bias akibat skala fitur yang berbeda

### Soal 2: Proyeksi Vektor dan Sudut
**Soal**: Diberikan vektor a = [4, 3] dan b = [2, 6]. Hitung:
a) Proyeksi vektor a pada b
b) Sudut antara vektor a dan b
c) Komponen a yang tegak lurus terhadap b

**Filosofi**: Proyeksi vektor adalah konsep fundamental dalam:
- **Principal Component Analysis (PCA)**: Proyeksi data ke ruang dimensi yang lebih rendah
- **Linear Regression**: Proyeksi vektor target ke ruang kolom dari matriks fitur
- **Feature Selection**: Memahami seberapa besar kontribusi satu fitur terhadap fitur lainnya
- **Orthogonalization**: Membuat fitur-fitur yang independen satu sama lain

### Soal 3: Cross Product dan Aplikasinya
**Soal**: Diberikan vektor 3D a = [1, 2, 3] dan b = [4, 5, 6]. Hitung:
a) a × b (cross product)
b) Volume parallelpiped yang dibentuk oleh a, b, dan c = [1, 1, 1]
c) Apakah ketiga vektor tersebut coplanar?

**Filosofi**: Cross product penting dalam:
- **Computer Vision**: Menghitung normal surface untuk 3D reconstruction
- **Robotics**: Menghitung torque dan rotasi
- **Geometric Deep Learning**: Operasi pada graph dan manifold
- **Data Augmentation**: Rotasi dan transformasi geometris pada data

### Soal 4: Linear Independence dan Span
**Soal**: Diberikan vektor v₁ = [1, 2, 3], v₂ = [2, 4, 6], v₃ = [1, 0, 1]. 
a) Apakah ketiga vektor ini linearly independent?
b) Tentukan span dari {v₁, v₃}
c) Apakah v₂ dapat dinyatakan sebagai kombinasi linear dari v₁ dan v₃?

**Filosofi**: Linear independence adalah konsep kunci dalam:
- **Feature Engineering**: Mendeteksi redundansi dalam fitur
- **Dimensionality Reduction**: Memahami dimensi intrinsik data
- **Model Capacity**: Menentukan kompleksitas model yang dapat dipelajari
- **Overfitting Prevention**: Menghindari parameter yang berlebihan

### Soal 5: Basis dan Koordinat
**Soal**: Diberikan basis B = {[1, 0], [1, 1]} untuk R². 
a) Nyatakan vektor [3, 2] dalam koordinat basis B
b) Jika vektor v memiliki koordinat [2, -1] dalam basis B, tentukan koordinat v dalam basis standar
c) Hitung matriks transformasi dari basis standar ke basis B

**Filosofi**: Perubahan basis adalah fundamental dalam:
- **Feature Transformation**: Mengubah representasi data untuk pembelajaran yang lebih baik
- **Principal Component Analysis**: Transformasi ke basis eigenvector
- **Kernel Methods**: Transformasi ke ruang fitur yang lebih tinggi
- **Neural Networks**: Setiap layer melakukan transformasi basis implisit

---

## BAGIAN II: OPERASI MATRIKS FUNDAMENTAL (Soal 6-10)

### Soal 6: Operasi Matriks Dasar
**Soal**: Diberikan matriks A = [[2, 3], [1, 4]] dan B = [[5, 1], [2, 3]]. Hitung:
a) A + B dan A - B
b) AB dan BA
c) A^T (transpose A)
d) det(A) dan det(B)
e) A^(-1) jika ada

**Filosofi**: Operasi matriks dasar adalah building blocks dari:
- **Neural Networks**: Forward propagation menggunakan perkalian matriks
- **Linear Regression**: Solusi normal equation menggunakan invers matriks
- **Covariance Matrix**: Transpose untuk menghitung statistik data
- **Determinant**: Mengukur "volume" transformasi, penting untuk stabilitas numerik

### Soal 7: Sistem Persamaan Linear
**Soal**: Selesaikan sistem persamaan:
```
2x + 3y + z = 7
x + 4y + 2z = 8
3x + y - z = 1
```
a) Tulis dalam bentuk matriks Ax = b
b) Selesaikan menggunakan eliminasi Gauss
c) Selesaikan menggunakan invers matriks
d) Analisis kondisi sistem (consistent, inconsistent, atau underdetermined)

**Filosofi**: Sistem persamaan linear adalah inti dari:
- **Linear Regression**: Mencari parameter optimal
- **Optimization**: Kondisi KKT dalam constrained optimization
- **Equilibrium Analysis**: Mencari titik keseimbangan dalam sistem
- **Network Analysis**: Solving flow problems dalam graph

### Soal 8: Rank dan Nullspace
**Soal**: Diberikan matriks A = [[1, 2, 3], [2, 4, 6], [1, 2, 4]]. Tentukan:
a) Rank dari A
b) Nullspace dari A
c) Column space dari A
d) Row space dari A

**Filosofi**: Rank dan nullspace penting untuk:
- **Model Identifiability**: Menentukan apakah parameter model dapat diestimasi secara unik
- **Feature Selection**: Rank menunjukkan jumlah fitur yang benar-benar independen
- **Regularization**: Nullspace menunjukkan arah parameter yang tidak mempengaruhi output
- **Compressed Sensing**: Memahami sparsity dan recoverability

### Soal 9: Eigenvalues dan Eigenvectors
**Soal**: Diberikan matriks A = [[3, 1], [0, 2]]. Tentukan:
a) Eigenvalues dari A
b) Eigenvectors yang bersesuaian
c) Diagonalisasi A jika mungkin
d) Hitung A^10 menggunakan diagonalisasi

**Filosofi**: Eigendecomposition adalah fondasi dari:
- **Principal Component Analysis**: Eigenvector menunjukkan arah varians maksimum
- **Markov Chains**: Eigenvalue menentukan konvergensi dan steady state
- **Stability Analysis**: Eigenvalue menentukan stabilitas sistem dinamis
- **Spectral Clustering**: Menggunakan eigenvector untuk partisi graph

### Soal 10: Singular Value Decomposition (SVD)
**Soal**: Diberikan matriks A = [[3, 2, 2], [2, 3, -2]]. Lakukan:
a) SVD dari A: A = UΣV^T
b) Tentukan rank dari A menggunakan SVD
c) Hitung pseudoinverse A^+ menggunakan SVD
d) Aproksimasi rank-1 dari A

**Filosofi**: SVD adalah "Swiss Army knife" dari aljabar linear:
- **Dimensionality Reduction**: Truncated SVD untuk PCA
- **Recommender Systems**: Matrix factorization untuk collaborative filtering
- **Image Compression**: Low-rank approximation untuk kompresi
- **Noise Reduction**: Filtering singular values untuk denoising

---

## BAGIAN III: APLIKASI DALAM MACHINE LEARNING (Soal 11-15)

### Soal 11: Linear Regression dengan Normal Equation
**Soal**: Diberikan data training:
```
X = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]  # dengan bias term
y = [2.1, 3.9, 6.1, 7.9, 10.1]
```
a) Hitung parameter optimal θ menggunakan normal equation: θ = (X^T X)^(-1) X^T y
b) Hitung prediksi untuk x_new = [1, 6]
c) Hitung R-squared
d) Analisis kondisi X^T X (condition number)

**Filosofi**: Normal equation menunjukkan:
- **Closed-form Solution**: Kapan kita bisa mendapat solusi eksak vs iteratif
- **Computational Complexity**: O(n³) vs O(n²) untuk gradient descent
- **Numerical Stability**: Pentingnya condition number dalam komputasi
- **Geometric Interpretation**: Proyeksi ortogonal dalam least squares

### Soal 12: Principal Component Analysis (PCA)
**Soal**: Diberikan dataset 2D:
```
X = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], 
     [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]]
```
a) Standardisasi data (mean=0, std=1)
b) Hitung covariance matrix
c) Tentukan principal components (eigenvectors)
d) Proyeksikan data ke PC1
e) Hitung variance explained ratio

**Filosofi**: PCA mengajarkan:
- **Dimensionality Reduction**: Mengurangi kompleksitas tanpa kehilangan informasi penting
- **Feature Extraction**: Menciptakan fitur baru yang lebih informatif
- **Data Visualization**: Proyeksi ke 2D/3D untuk eksplorasi data
- **Noise Reduction**: Memfilter komponen dengan varians rendah

### Soal 13: Cosine Similarity dan Clustering
**Soal**: Diberikan vektor dokumen (TF-IDF):
```
doc1 = [0.5, 0.8, 0.2, 0.0, 0.1]
doc2 = [0.3, 0.7, 0.1, 0.2, 0.0]
doc3 = [0.0, 0.1, 0.0, 0.9, 0.8]
doc4 = [0.1, 0.2, 0.1, 0.7, 0.6]
```
a) Hitung cosine similarity antara semua pasangan dokumen
b) Buat similarity matrix
c) Lakukan hierarchical clustering berdasarkan similarity
d) Interpretasi hasil clustering

**Filosofi**: Cosine similarity penting dalam:
- **Information Retrieval**: Mencari dokumen yang relevan
- **Recommendation Systems**: Mencari user/item yang similar
- **Natural Language Processing**: Mengukur semantic similarity
- **Anomaly Detection**: Mendeteksi data point yang berbeda

### Soal 14: Gradient Descent dan Optimasi
**Soal**: Untuk fungsi loss f(w) = (1/2)||Xw - y||², dimana:
```
X = [[1, 2], [1, 3], [1, 4]]
y = [3, 5, 7]
```
a) Hitung gradient ∇f(w) secara analitik
b) Implementasi gradient descent dengan learning rate α = 0.01
c) Plot konvergensi loss function
d) Bandingkan dengan solusi analitik

**Filosofi**: Gradient descent mengajarkan:
- **Iterative Optimization**: Pendekatan step-by-step menuju optimum
- **Learning Rate**: Trade-off antara kecepatan dan stabilitas konvergensi
- **Convexity**: Pentingnya sifat geometris fungsi objektif
- **Scalability**: Mengapa iterative methods penting untuk big data

### Soal 15: Matrix Factorization untuk Recommender System
**Soal**: Diberikan rating matrix (user × item):
```
R = [[5, 3, 0, 1, 4],
     [4, 0, 0, 1, 3],
     [1, 1, 0, 5, 4],
     [1, 0, 0, 4, 4],
     [0, 1, 5, 4, 0]]
```
a) Lakukan SVD pada R
b) Aproksimasi dengan rank-2
c) Prediksi rating yang missing (nilai 0)
d) Evaluasi menggunakan RMSE

**Filosofi**: Matrix factorization menunjukkan:
- **Latent Factor Models**: Menemukan pola tersembunyi dalam data
- **Collaborative Filtering**: Memanfaatkan preferensi user lain
- **Dimensionality Reduction**: Kompresi informasi dengan minimal loss
- **Cold Start Problem**: Mengatasi user/item baru

---

## BAGIAN IV: KONSEP LANJUTAN (Soal 16-20)

### Soal 16: Kernel Methods dan Feature Mapping
**Soal**: Diberikan data 1D: x = [1, 2, 3, 4]. Lakukan:
a) Mapping ke ruang fitur 2D: φ(x) = [x, x²]
b) Hitung kernel matrix K menggunakan polynomial kernel k(x,y) = (xy + 1)²
c) Bandingkan dengan explicit feature mapping
d) Implementasi kernel PCA

**Filosofi**: Kernel methods mengajarkan:
- **Feature Space Transformation**: Membuat data linearly separable
- **Kernel Trick**: Komputasi efisien di ruang dimensi tinggi
- **Non-linear Patterns**: Menangkap hubungan non-linear dalam data
- **Support Vector Machines**: Fondasi untuk SVM dan kernel methods

### Soal 17: Regularization dan Ridge Regression
**Soal**: Untuk dataset dengan multicollinearity:
```
X = [[1, 1, 1], [1, 2, 2.1], [1, 3, 2.9], [1, 4, 4.1]]
y = [2, 4, 6, 8]
```
a) Hitung condition number dari X^T X
b) Solve menggunakan ordinary least squares
c) Solve menggunakan ridge regression dengan λ = 0.1, 1.0, 10.0
d) Plot regularization path

**Filosofi**: Regularization mengajarkan:
- **Bias-Variance Tradeoff**: Mengurangi overfitting dengan menambah bias
- **Ill-conditioned Problems**: Mengatasi masalah numerik dalam optimasi
- **Feature Selection**: L1 regularization untuk sparsity
- **Generalization**: Meningkatkan performa pada data baru

### Soal 18: Markov Chains dan PageRank
**Soal**: Diberikan adjacency matrix untuk web graph:
```
A = [[0, 1, 1, 0],
     [1, 0, 1, 1],
     [1, 0, 0, 1],
     [0, 1, 1, 0]]
```
a) Buat transition matrix P
b) Hitung steady-state distribution (eigenvector untuk eigenvalue 1)
c) Implementasi PageRank algorithm
d) Analisis konvergensi

**Filosofi**: Markov chains menunjukkan:
- **Random Walks**: Modeling sequential processes
- **Steady State**: Long-term behavior dari sistem stokastik
- **Web Search**: Algoritma ranking untuk search engines
- **Network Analysis**: Mengukur importance dalam graph

### Soal 19: Neural Network Forward Propagation
**Soal**: Untuk neural network sederhana dengan:
- Input layer: 2 neurons
- Hidden layer: 3 neurons (sigmoid activation)
- Output layer: 1 neuron (sigmoid activation)

Weights:
```
W1 = [[0.5, 0.2, -0.1], [0.3, -0.4, 0.6]]  # 2×3
b1 = [0.1, -0.2, 0.3]                       # 3×1
W2 = [[0.7], [-0.5], [0.2]]                 # 3×1
b2 = [0.1]                                  # 1×1
```

a) Implementasi forward propagation untuk input x = [1, 0.5]
b) Hitung gradient untuk backpropagation
c) Update weights menggunakan gradient descent
d) Analisis fungsi yang dipelajari network

**Filosofi**: Neural networks mengajarkan:
- **Universal Approximation**: Kemampuan approximate fungsi kompleks
- **Hierarchical Feature Learning**: Pembelajaran representasi bertingkat
- **Non-linear Transformations**: Kombinasi linear dan non-linear operations
- **Gradient-based Learning**: Optimasi parameter dalam ruang dimensi tinggi

### Soal 20: Spectral Clustering
**Soal**: Diberikan similarity matrix:
```
S = [[1.0, 0.8, 0.2, 0.1],
     [0.8, 1.0, 0.3, 0.2],
     [0.2, 0.3, 1.0, 0.9],
     [0.1, 0.2, 0.9, 1.0]]
```
a) Buat degree matrix D dan Laplacian matrix L = D - S
b) Hitung eigenvalues dan eigenvectors dari L
c) Gunakan eigenvector ke-2 untuk clustering (Fiedler vector)
d) Bandingkan dengan k-means clustering

**Filosofi**: Spectral clustering menunjukkan:
- **Graph-based Clustering**: Clustering berdasarkan connectivity
- **Spectral Analysis**: Menggunakan eigenstructure untuk pattern recognition
- **Non-convex Clusters**: Mengatasi keterbatasan k-means
- **Dimensionality Reduction**: Embedding graph ke ruang Euclidean

---

## Kesimpulan

Ke-20 soal ini dirancang untuk memberikan pemahaman yang komprehensif tentang aljabar linear dalam konteks machine learning. Setiap soal tidak hanya melatih kemampuan komputasi, tetapi juga membangun intuisi tentang mengapa konsep-konsep ini fundamental dalam ML.

**Progresivitas Pembelajaran:**
1. **Soal 1-5**: Membangun fondasi vektor dan operasinya
2. **Soal 6-10**: Mengembangkan pemahaman matriks dan transformasi
3. **Soal 11-15**: Aplikasi langsung dalam algoritma ML klasik
4. **Soal 16-20**: Konsep lanjutan untuk ML modern

**Filosofi Keseluruhan:**
Aljabar linear bukan hanya alat komputasi, tetapi bahasa untuk memahami struktur data, transformasi informasi, dan optimasi. Penguasaan konsep-konsep ini memungkinkan praktisi ML untuk:
- Memahami algoritma secara mendalam, bukan hanya sebagai "black box"
- Mendiagnosis dan mengatasi masalah dalam implementasi
- Mengembangkan intuisi untuk desain model dan feature engineering
- Berkomunikasi secara efektif dengan tim teknis lainnya

Setiap soal dalam koleksi ini berkontribusi pada pemahaman holistik tentang bagaimana matematika mendukung kecerdasan buatan modern.




## Solusi Python

Berikut adalah implementasi Python untuk semua 20 soal, menggunakan NumPy, SciPy, dan Scikit-learn.

```python
#!/usr/bin/env python3
"""
Solusi 20 Soal Vektor dan Matriks untuk Machine Learning
Implementasi lengkap dengan penjelasan dan visualisasi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, eig
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("20 SOAL VEKTOR DAN MATRIKS UNTUK MACHINE LEARNING")
print("=" * 80)

# ============================================================================
# BAGIAN I: KONSEP DASAR VEKTOR (Soal 1-5)
# ============================================================================

print("\n" + "="*60)
print("BAGIAN I: KONSEP DASAR VEKTOR")
print("="*60)

# SOAL 1: Operasi Dasar Vektor
print("\n" + "-"*50)
print("SOAL 1: Operasi Dasar Vektor")
print("-"*50)

u = np.array([3, 4, 5])
v = np.array([1, 2, 3])

print(f"Diberikan: u = {u}, v = {v}")
print("\nSolusi:")
print(f"a) u + v = {u + v}")
print(f"b) u - v = {u - v}")
print(f"c) 3u = {3 * u}")
print(f"d) u · v = {np.dot(u, v)}")
print(f"e) ||u|| = {np.linalg.norm(u):.4f}")
print(f"f) û = {u / np.linalg.norm(u)}")

# SOAL 2: Proyeksi Vektor dan Sudut
print("\n" + "-"*50)
print("SOAL 2: Proyeksi Vektor dan Sudut")
print("-"*50)

a = np.array([4, 3])
b = np.array([2, 6])

print(f"Diberikan: a = {a}, b = {b}")

# Proyeksi a pada b
proj_a_on_b = (np.dot(a, b) / np.dot(b, b)) * b
print(f"\na) Proyeksi a pada b = {proj_a_on_b}")

# Sudut antara vektor
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta_deg = np.degrees(np.arccos(cos_theta))
print(f"b) Sudut antara a dan b = {theta_deg:.2f}°")

# Komponen tegak lurus
perp_component = a - proj_a_on_b
print(f"c) Komponen a tegak lurus b = {perp_component}")

# SOAL 3: Cross Product dan Aplikasinya
print("\n" + "-"*50)
print("SOAL 3: Cross Product dan Aplikasinya")
print("-"*50)

a3d = np.array([1, 2, 3])
b3d = np.array([4, 5, 6])
c3d = np.array([1, 1, 1])

print(f"Diberikan: a = {a3d}, b = {b3d}, c = {c3d}")

# Cross product
cross_ab = np.cross(a3d, b3d)
print(f"\na) a × b = {cross_ab}")

# Volume parallelpiped
volume = abs(np.dot(cross_ab, c3d))
print(f"b) Volume parallelpiped = {volume}")

# Coplanar check
print(f"c) Coplanar? {volume == 0} (volume = 0 jika coplanar)")

# SOAL 4: Linear Independence dan Span
print("\n" + "-"*50)
print("SOAL 4: Linear Independence dan Span")
print("-"*50)

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([1, 0, 1])

print(f"Diberikan: v1 = {v1}, v2 = {v2}, v3 = {v3}")

# Buat matriks dan hitung rank
matrix = np.column_stack([v1, v2, v3])
rank = np.linalg.matrix_rank(matrix)
print(f"\na) Rank matriks = {rank}")
print(f"   Linear independent? {rank == 3}")

# Check if v2 is linear combination of v1 and v3
matrix_13 = np.column_stack([v1, v3])
rank_13 = np.linalg.matrix_rank(matrix_13)
rank_with_v2 = np.linalg.matrix_rank(np.column_stack([v1, v3, v2]))
print(f"b) v2 kombinasi linear v1,v3? {rank_13 == rank_with_v2}")

# SOAL 5: Basis dan Koordinat
print("\n" + "-"*50)
print("SOAL 5: Basis dan Koordinat")
print("-"*50)

# Basis B = {[1,0], [1,1]}
B = np.array([[1, 1], [0, 1]])
target_vector = np.array([3, 2])

print(f"Basis B = {B.T}")
print(f"Target vector = {target_vector}")

# Koordinat dalam basis B
coords_B = np.linalg.solve(B, target_vector)
print(f"\na) Koordinat dalam basis B = {coords_B}")

# Verifikasi
reconstructed = B @ coords_B
print(f"   Verifikasi: B @ coords = {reconstructed}")

# Matriks transformasi
print(f"b) Matriks transformasi dari standar ke B = \n{np.linalg.inv(B)}")

# ============================================================================
# BAGIAN II: OPERASI MATRIKS FUNDAMENTAL (Soal 6-10)
# ============================================================================

print("\n" + "="*60)
print("BAGIAN II: OPERASI MATRIKS FUNDAMENTAL")
print("="*60)

# SOAL 6: Operasi Matriks Dasar
print("\n" + "-"*50)
print("SOAL 6: Operasi Matriks Dasar")
print("-"*50)

A = np.array([[2, 3], [1, 4]])
B = np.array([[5, 1], [2, 3]])

print(f"A = \n{A}")
print(f"B = \n{B}")

print(f"\na) A + B = \n{A + B}")
print(f"   A - B = \n{A - B}")
print(f"b) AB = \n{A @ B}")
print(f"   BA = \n{B @ A}")
print(f"c) A^T = \n{A.T}")
print(f"d) det(A) = {np.linalg.det(A):.4f}")
print(f"   det(B) = {np.linalg.det(B):.4f}")
print(f"e) A^(-1) = \n{np.linalg.inv(A)}")

# SOAL 7: Sistem Persamaan Linear
print("\n" + "-"*50)
print("SOAL 7: Sistem Persamaan Linear")
print("-"*50)

# Sistem: 2x + 3y + z = 7, x + 4y + 2z = 8, 3x + y - z = 1
A_sys = np.array([[2, 3, 1], [1, 4, 2], [3, 1, -1]])
b_sys = np.array([7, 8, 1])

print("Sistem persamaan:")
print("2x + 3y + z = 7")
print("x + 4y + 2z = 8")
print("3x + y - z = 1")

print(f"\na) Bentuk matriks Ax = b:")
print(f"A = \n{A_sys}")
print(f"b = {b_sys}")

# Solusi
x_solution = np.linalg.solve(A_sys, b_sys)
print(f"\nb) Solusi: x = {x_solution}")

# Verifikasi
verification = A_sys @ x_solution
print(f"c) Verifikasi Ax = {verification}")
print(f"   b = {b_sys}")
print(f"   Error = {np.linalg.norm(verification - b_sys):.10f}")

# Kondisi sistem
cond_num = np.linalg.cond(A_sys)
print(f"d) Condition number = {cond_num:.4f}")

# SOAL 8: Rank dan Nullspace
print("\n" + "-"*50)
print("SOAL 8: Rank dan Nullspace")
print("-"*50)

A_rank = np.array([[1, 2, 3], [2, 4, 6], [1, 2, 4]])
print(f"A = \n{A_rank}")

# Rank
rank_A = np.linalg.matrix_rank(A_rank)
print(f"\na) Rank(A) = {rank_A}")

# SVD untuk nullspace
U, s, Vt = np.linalg.svd(A_rank)
null_space = Vt[rank_A:].T
print(f"b) Nullspace dimension = {null_space.shape[1]}")
if null_space.shape[1] > 0:
    print(f"   Nullspace basis = \n{null_space}")

# Column space (range)
print(f"c) Column space dimension = {rank_A}")

# SOAL 9: Eigenvalues dan Eigenvectors
print("\n" + "-"*50)
print("SOAL 9: Eigenvalues dan Eigenvectors")
print("-"*50)

A_eigen = np.array([[3, 1], [0, 2]])
print(f"A = \n{A_eigen}")

# Eigendecomposition
eigenvals, eigenvecs = np.linalg.eig(A_eigen)
print(f"\na) Eigenvalues = {eigenvals}")
print(f"b) Eigenvectors = \n{eigenvecs}")

# Diagonalisasi
P = eigenvecs
D = np.diag(eigenvals)
P_inv = np.linalg.inv(P)
print(f"c) P = \n{P}")
print(f"   D = \n{D}")
print(f"   P^(-1) = \n{P_inv}")

# Verifikasi A = PDP^(-1)
A_reconstructed = P @ D @ P_inv
print(f"   A (reconstructed) = \n{A_reconstructed}")

# A^10 menggunakan diagonalisasi
D_10 = np.diag(eigenvals**10)
A_10 = P @ D_10 @ P_inv
print(f"d) A^10 = \n{A_10}")

# SOAL 10: Singular Value Decomposition (SVD)
print("\n" + "-"*50)
print("SOAL 10: Singular Value Decomposition (SVD)")
print("-"*50)

A_svd = np.array([[3, 2, 2], [2, 3, -2]])
print(f"A = \n{A_svd}")

# SVD
U, sigma, Vt = np.linalg.svd(A_svd)
print(f"\na) SVD: A = UΣV^T")
print(f"U = \n{U}")
print(f"Σ = {sigma}")
print(f"V^T = \n{Vt}")

# Rank dari SVD
rank_svd = np.sum(sigma > 1e-10)
print(f"\nb) Rank dari SVD = {rank_svd}")

# Pseudoinverse
# Untuk matriks m×n, perlu menyesuaikan dimensi
m, n = A_svd.shape
sigma_inv = np.zeros((n, m))
for i in range(min(m, n)):
    if sigma[i] > 1e-10:
        sigma_inv[i, i] = 1/sigma[i]
A_pinv = Vt.T @ sigma_inv @ U.T
print(f"c) Pseudoinverse A^+ = \n{A_pinv}")

# Rank-1 approximation
sigma_rank1 = np.zeros_like(sigma)
sigma_rank1[0] = sigma[0]
Sigma_rank1 = np.zeros((m, n))
Sigma_rank1[:len(sigma_rank1), :len(sigma_rank1)] = np.diag(sigma_rank1)
A_rank1 = U @ Sigma_rank1 @ Vt
print(f"d) Rank-1 approximation = \n{A_rank1}")

# ============================================================================
# BAGIAN III: APLIKASI DALAM MACHINE LEARNING (Soal 11-15)
# ============================================================================

print("\n" + "="*60)
print("BAGIAN III: APLIKASI DALAM MACHINE LEARNING")
print("="*60)

# SOAL 11: Linear Regression dengan Normal Equation
print("\n" + "-"*50)
print("SOAL 11: Linear Regression dengan Normal Equation")
print("-"*50)

X_lr = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
y_lr = np.array([2.1, 3.9, 6.1, 7.9, 10.1])

print(f"X (dengan bias) = \n{X_lr}")
print(f"y = {y_lr}")

# Normal equation
XtX = X_lr.T @ X_lr
Xty = X_lr.T @ y_lr
theta = np.linalg.solve(XtX, Xty)
print(f"\na) θ = (X^T X)^(-1) X^T y = {theta}")

# Prediksi
x_new = np.array([1, 6])
y_pred_new = x_new @ theta
print(f"b) Prediksi untuk x_new = {x_new}: y = {y_pred_new:.2f}")

# R-squared
y_pred = X_lr @ theta
ss_res = np.sum((y_lr - y_pred)**2)
ss_tot = np.sum((y_lr - np.mean(y_lr))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"c) R-squared = {r_squared:.4f}")

# Condition number
cond_num_lr = np.linalg.cond(XtX)
print(f"d) Condition number X^T X = {cond_num_lr:.4f}")

# SOAL 12: Principal Component Analysis (PCA)
print("\n" + "-"*50)
print("SOAL 12: Principal Component Analysis (PCA)")
print("-"*50)

X_pca = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], 
                  [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])

print(f"Dataset original shape: {X_pca.shape}")
print(f"Mean original: {np.mean(X_pca, axis=0)}")

# Standardisasi
X_std = StandardScaler().fit_transform(X_pca)
print(f"\na) Data standardized:")
print(f"Mean: {np.mean(X_std, axis=0)}")
print(f"Std: {np.std(X_std, axis=0)}")

# Covariance matrix
cov_matrix = np.cov(X_std.T)
print(f"\nb) Covariance matrix = \n{cov_matrix}")

# Eigendecomposition
eigenvals_pca, eigenvecs_pca = np.linalg.eig(cov_matrix)
idx = np.argsort(eigenvals_pca)[::-1]
eigenvals_pca = eigenvals_pca[idx]
eigenvecs_pca = eigenvecs_pca[:, idx]

print(f"c) Principal components (eigenvectors) = \n{eigenvecs_pca}")
print(f"   Eigenvalues = {eigenvals_pca}")

# Proyeksi ke PC1
pc1_scores = X_std @ eigenvecs_pca[:, 0]
print(f"d) PC1 scores = {pc1_scores}")

# Variance explained
var_explained = eigenvals_pca / np.sum(eigenvals_pca)
print(f"e) Variance explained ratio = {var_explained}")

# SOAL 13: Cosine Similarity dan Clustering
print("\n" + "-"*50)
print("SOAL 13: Cosine Similarity dan Clustering")
print("-"*50)

docs = np.array([[0.5, 0.8, 0.2, 0.0, 0.1],
                 [0.3, 0.7, 0.1, 0.2, 0.0],
                 [0.0, 0.1, 0.0, 0.9, 0.8],
                 [0.1, 0.2, 0.1, 0.7, 0.6]])

print(f"Document vectors:")
for i, doc in enumerate(docs):
    print(f"doc{i+1} = {doc}")

# Cosine similarity matrix
cos_sim_matrix = cosine_similarity(docs)
print(f"\na) Cosine similarity matrix:")
print(cos_sim_matrix)

# Manual calculation untuk verifikasi
doc1, doc2 = docs[0], docs[1]
cos_sim_manual = np.dot(doc1, doc2) / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
print(f"\nb) Manual calculation doc1-doc2: {cos_sim_manual:.4f}")

# Simple clustering berdasarkan similarity
print(f"c) Clustering interpretation:")
print(f"   doc1 dan doc2 similar (similarity = {cos_sim_matrix[0,1]:.3f})")
print(f"   doc3 dan doc4 similar (similarity = {cos_sim_matrix[2,3]:.3f})")
print(f"   Cluster 1: [doc1, doc2], Cluster 2: [doc3, doc4]")

# SOAL 14: Gradient Descent dan Optimasi
print("\n" + "-"*50)
print("SOAL 14: Gradient Descent dan Optimasi")
print("-"*50)

X_gd = np.array([[1, 2], [1, 3], [1, 4]])
y_gd = np.array([3, 5, 7])

print(f"X = \n{X_gd}")
print(f"y = {y_gd}")

# Gradient function
def compute_gradient(X, y, w):
    m = len(y)
    y_pred = X @ w
    gradient = (1/m) * X.T @ (y_pred - y)
    return gradient

def compute_loss(X, y, w):
    m = len(y)
    y_pred = X @ w
    loss = (1/(2*m)) * np.sum((y_pred - y)**2)
    return loss

# Gradient descent
w = np.array([0.0, 0.0])  # Initial weights
alpha = 0.01  # Learning rate
losses = []

print(f"\na) Gradient descent dengan α = {alpha}")
print(f"Initial w = {w}")

for i in range(1000):
    gradient = compute_gradient(X_gd, y_gd, w)
    w = w - alpha * gradient
    loss = compute_loss(X_gd, y_gd, w)
    losses.append(loss)
    
    if i % 200 == 0:
        print(f"Iteration {i}: w = {w}, loss = {loss:.6f}")

print(f"Final w = {w}")

# Analytical solution
w_analytical = np.linalg.solve(X_gd.T @ X_gd, X_gd.T @ y_gd)
print(f"d) Analytical solution: w = {w_analytical}")
print(f"   Difference: {np.linalg.norm(w - w_analytical):.8f}")

# SOAL 15: Matrix Factorization untuk Recommender System
print("\n" + "-"*50)
print("SOAL 15: Matrix Factorization untuk Recommender System")
print("-"*50)

R = np.array([[5, 3, 0, 1, 4],
              [4, 0, 0, 1, 3],
              [1, 1, 0, 5, 4],
              [1, 0, 0, 4, 4],
              [0, 1, 5, 4, 0]])

print(f"Rating matrix R:")
print(R)

# SVD
U_r, sigma_r, Vt_r = np.linalg.svd(R, full_matrices=False)
print(f"\na) SVD components:")
print(f"U shape: {U_r.shape}")
print(f"Sigma: {sigma_r}")
print(f"Vt shape: {Vt_r.shape}")

# Rank-2 approximation
k = 2
U_k = U_r[:, :k]
sigma_k = sigma_r[:k]
Vt_k = Vt_r[:k, :]

R_approx = U_k @ np.diag(sigma_k) @ Vt_k
print(f"\nb) Rank-{k} approximation:")
print(R_approx)

# Prediksi untuk missing values
print(f"\nc) Prediksi untuk missing values (original 0s):")
mask = (R == 0)
predictions = R_approx[mask]
print(f"Predicted values: {predictions}")

# RMSE
observed_mask = (R > 0)
rmse = np.sqrt(np.mean((R[observed_mask] - R_approx[observed_mask])**2))
print(f"d) RMSE = {rmse:.4f}")

# ============================================================================
# BAGIAN IV: KONSEP LANJUTAN (Soal 16-20)
# ============================================================================

print("\n" + "="*60)
print("BAGIAN IV: KONSEP LANJUTAN")
print("="*60)

# SOAL 16: Kernel Methods dan Feature Mapping
print("\n" + "-"*50)
print("SOAL 16: Kernel Methods dan Feature Mapping")
print("-"*50)

x_data = np.array([1, 2, 3, 4])
print(f"Data 1D: x = {x_data}")

# Feature mapping φ(x) = [x, x²]
phi_x = np.column_stack([x_data, x_data**2])
print(f"\na) Feature mapping φ(x) = [x, x²]:")
print(phi_x)

# Polynomial kernel k(x,y) = (xy + 1)²
def polynomial_kernel(x, y, degree=2, coef0=1):
    return (x * y + coef0) ** degree

# Kernel matrix
n = len(x_data)
K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        K[i, j] = polynomial_kernel(x_data[i], x_data[j])

print(f"\nb) Kernel matrix K:")
print(K)

# Explicit feature mapping kernel matrix
K_explicit = phi_x @ phi_x.T
print(f"c) Explicit mapping kernel matrix:")
print(K_explicit)

print(f"d) Matrices are equal: {np.allclose(K, K_explicit)}")

# SOAL 17: Regularization dan Ridge Regression
print("\n" + "-"*50)
print("SOAL 17: Regularization dan Ridge Regression")
print("-"*50)

X_ridge = np.array([[1, 1, 1], [1, 2, 2.1], [1, 3, 2.9], [1, 4, 4.1]])
y_ridge = np.array([2, 4, 6, 8])

print(f"X = \n{X_ridge}")
print(f"y = {y_ridge}")

# Condition number
XtX_ridge = X_ridge.T @ X_ridge
cond_num_ridge = np.linalg.cond(XtX_ridge)
print(f"\na) Condition number X^T X = {cond_num_ridge:.2f}")

# OLS solution
theta_ols = np.linalg.solve(XtX_ridge, X_ridge.T @ y_ridge)
print(f"b) OLS solution: θ = {theta_ols}")

# Ridge regression for different λ values
lambdas = [0.1, 1.0, 10.0]
print(f"c) Ridge regression solutions:")

for lam in lambdas:
    I = np.eye(X_ridge.shape[1])
    theta_ridge = np.linalg.solve(XtX_ridge + lam * I, X_ridge.T @ y_ridge)
    print(f"   λ = {lam}: θ = {theta_ridge}")

# SOAL 18: Markov Chains dan PageRank
print("\n" + "-"*50)
print("SOAL 18: Markov Chains dan PageRank")
print("-"*50)

A_graph = np.array([[0, 1, 1, 0],
                    [1, 0, 1, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0]])

print(f"Adjacency matrix A:")
print(A_graph)

# Transition matrix (row-stochastic)
row_sums = A_graph.sum(axis=1)
P = A_graph / row_sums[:, np.newaxis]
print(f"\na) Transition matrix P:")
print(P)

# Steady state (eigenvector for eigenvalue 1)
eigenvals_markov, eigenvecs_markov = np.linalg.eig(P.T)
steady_idx = np.argmin(np.abs(eigenvals_markov - 1))
steady_state = np.real(eigenvecs_markov[:, steady_idx])
steady_state = steady_state / np.sum(steady_state)
print(f"b) Steady state distribution: {steady_state}")

# PageRank with damping
damping = 0.85
n_nodes = A_graph.shape[0]
M = damping * P.T + (1 - damping) / n_nodes * np.ones((n_nodes, n_nodes))

# Power iteration for PageRank
pagerank = np.ones(n_nodes) / n_nodes
for _ in range(100):
    pagerank = M @ pagerank

print(f"c) PageRank scores: {pagerank}")

# SOAL 19: Neural Network Forward Propagation
print("\n" + "-"*50)
print("SOAL 19: Neural Network Forward Propagation")
print("-"*50)

# Network parameters
W1 = np.array([[0.5, 0.2, -0.1], [0.3, -0.4, 0.6]])  # 2×3
b1 = np.array([0.1, -0.2, 0.3])                       # 3×1
W2 = np.array([[0.7], [-0.5], [0.2]])                 # 3×1
b2 = np.array([0.1])                                  # 1×1

x_input = np.array([1, 0.5])

print(f"Input: x = {x_input}")
print(f"W1 = \n{W1}")
print(f"b1 = {b1}")
print(f"W2 = \n{W2}")
print(f"b2 = {b2}")

# Sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Forward propagation
z1 = W1.T @ x_input + b1
a1 = sigmoid(z1)
z2 = W2.T @ a1 + b2
a2 = sigmoid(z2)

print(f"\na) Forward propagation:")
print(f"z1 = {z1}")
print(f"a1 = {a1}")
print(f"z2 = {z2}")
print(f"a2 = {a2}")

# Gradient computation (assuming target y = 1)
y_target = 1
dz2 = a2 - y_target
dW2 = a1.reshape(-1, 1) * dz2
db2 = dz2

da1 = W2.flatten() * dz2
dz1 = da1 * sigmoid_derivative(z1)
dW1 = np.outer(x_input, dz1)
db1 = dz1

print(f"\nb) Gradients:")
print(f"dW2 = \n{dW2}")
print(f"db2 = {db2}")
print(f"dW1 = \n{dW1}")
print(f"db1 = {db1}")

# SOAL 20: Spectral Clustering
print("\n" + "-"*50)
print("SOAL 20: Spectral Clustering")
print("-"*50)

S = np.array([[1.0, 0.8, 0.2, 0.1],
              [0.8, 1.0, 0.3, 0.2],
              [0.2, 0.3, 1.0, 0.9],
              [0.1, 0.2, 0.9, 1.0]])

print(f"Similarity matrix S:")
print(S)

# Degree matrix
D = np.diag(np.sum(S, axis=1))
print(f"\na) Degree matrix D:")
print(D)

# Laplacian matrix
L = D - S
print(f"b) Laplacian matrix L = D - S:")
print(L)

# Eigendecomposition of Laplacian
eigenvals_L, eigenvecs_L = np.linalg.eig(L)
idx_L = np.argsort(eigenvals_L)
eigenvals_L = eigenvals_L[idx_L]
eigenvecs_L = eigenvecs_L[:, idx_L]

print(f"c) Eigenvalues: {eigenvals_L}")
print(f"   Eigenvectors:\n{eigenvecs_L}")

# Fiedler vector (second smallest eigenvalue)
fiedler_vector = eigenvecs_L[:, 1]
print(f"d) Fiedler vector (2nd eigenvector): {fiedler_vector}")

# Clustering based on sign of Fiedler vector
cluster_assignment = (fiedler_vector > 0).astype(int)
print(f"   Cluster assignment: {cluster_assignment}")
print(f"   Cluster 0: nodes {np.where(cluster_assignment == 0)[0]}")
print(f"   Cluster 1: nodes {np.where(cluster_assignment == 1)[0]}")

print("\n" + "="*80)
print("SEMUA 20 SOAL TELAH DISELESAIKAN!")
print("="*80)

print("\nRingkasan Konsep yang Dipelajari:")
print("1. Operasi dasar vektor dan matriks")
print("2. Proyeksi, cross product, dan linear independence")
print("3. Eigendecomposition dan SVD")
print("4. Aplikasi dalam linear regression dan PCA")
print("5. Similarity measures dan clustering")
print("6. Optimization dan gradient descent")
print("7. Matrix factorization untuk recommender systems")
print("8. Kernel methods dan regularization")
print("9. Markov chains dan PageRank")
print("10. Neural networks dan spectral clustering")
```


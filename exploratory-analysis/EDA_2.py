
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import os
import time
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Vienna RNA paketi kuruluysa import et (yoksa bu bölümü atlayabilirsiniz)
try:
    import RNA
    vienna_available = True
except ImportError:
    vienna_available = False
    print("Vienna RNA paketi bulunamadı, sekonder yapı analizleri atlanacak.")

# Başlangıç zamanı
start_time = time.time()

# Dosya yolları - kendinize göre ayarlayın

data_dir = r"C:\Users\ASUS\Desktop\comp3\data"
output_dir = r"C:\Users\ASUS\Desktop\comp3"

# Çıktı dizinini oluştur (yoksa)
os.makedirs(output_dir, exist_ok=True)

print("RNA 3D Folding Yarışması - Gelişmiş EDA Başlıyor...")

# ----------------------------------------------------------------------------------
# 1. VERİ YÜKLEME VE İNCELEME
# ----------------------------------------------------------------------------------
print("\n=== 1. VERİ YÜKLEME VE İNCELEME ===")

# Veri dosyalarını yükle
print("Veri dosyaları yükleniyor...")
try:
    sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    test_sequences = pd.read_csv(os.path.join(data_dir, "test_sequences.csv"))
    train_labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
    train_sequences = pd.read_csv(os.path.join(data_dir, "train_sequences.csv"))
    validation_labels = pd.read_csv(os.path.join(data_dir, "validation_labels.csv"))
    validation_sequences = pd.read_csv(os.path.join(data_dir, "validation_sequences.csv"))

    print("Tüm dosyalar başarıyla yüklendi.")
except Exception as e:
    print(f"Dosya yükleme hatası: {e}")
    exit()

# Veri setlerinin boyutunu yazdır
print("\n--- Veri Seti Boyutları ---")
print(f"sample_submission: {sample_submission.shape}")
print(f"test_sequences: {test_sequences.shape}")
print(f"train_labels: {train_labels.shape}")
print(f"train_sequences: {train_sequences.shape}")
print(f"validation_labels: {validation_labels.shape}")
print(f"validation_sequences: {validation_sequences.shape}")

# Veri setlerinin ilk birkaç satırını incele
print("\n--- Train Sequences İlk 3 Satır ---")
print(train_sequences.head(3))

print("\n--- Train Labels İlk 3 Satır ---")
print(train_labels.head(3))

# Tüm veri setlerindeki sütunları kontrol et
print("\n--- Veri Sütunları ---")
print("test_sequences sütunları:", test_sequences.columns.tolist())
print("train_sequences sütunları:", train_sequences.columns.tolist())
print("validation_sequences sütunları:", validation_sequences.columns.tolist())

# Benzersiz target_id sayıları
print("\n--- Benzersiz Target ID Sayıları ---")
print(f"Train sequences: {train_sequences['target_id'].nunique()}")
print(f"Test sequences: {test_sequences['target_id'].nunique()}")
print(f"Validation sequences: {validation_sequences['target_id'].nunique()}")

# Train Labels'daki benzersiz ID sayısı
unique_ids_train_labels = len(set([id.split('_')[0] for id in train_labels['ID']]))
print(f"Train labels (unique targets): {unique_ids_train_labels}")

# ----------------------------------------------------------------------------------
# 2. RNA DİZİLERİ ANALİZİ
# ----------------------------------------------------------------------------------
print("\n=== 2. RNA DİZİLERİ ANALİZİ ===")

# GC içeriği hesaplama fonksiyonu
def GC(sequence):
    """Bir RNA/DNA dizisindeki G ve C bazlarının yüzdesini hesaplar"""
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    total_length = len(sequence)
    if total_length == 0:
        return 0
    return (gc_count / total_length) * 100

# RNA dizilerinin uzunluk dağılımı
print("\n--- RNA Dizisi Uzunluk Dağılımı ---")
train_seq_lengths = train_sequences['sequence'].apply(len)
test_seq_lengths = test_sequences['sequence'].apply(len)
validation_seq_lengths = validation_sequences['sequence'].apply(len)

print(f"Eğitim seti RNA uzunlukları:")
print(f"  Min: {train_seq_lengths.min()}")
print(f"  Max: {train_seq_lengths.max()}")
print(f"  Mean: {train_seq_lengths.mean():.2f}")
print(f"  Median: {train_seq_lengths.median()}")
print(f"  Std Dev: {train_seq_lengths.std():.2f}")

print(f"\nTest seti RNA uzunlukları:")
print(f"  Min: {test_seq_lengths.min()}")
print(f"  Max: {test_seq_lengths.max()}")
print(f"  Mean: {test_seq_lengths.mean():.2f}")
print(f"  Median: {test_seq_lengths.median()}")
print(f"  Std Dev: {test_seq_lengths.std():.2f}")

# RNA uzunluk dağılımı grafiği
plt.figure(figsize=(12, 6))
plt.hist(train_seq_lengths, bins=30, alpha=0.7, label='Eğitim')
plt.hist(test_seq_lengths, bins=30, alpha=0.7, label='Test')
plt.hist(validation_seq_lengths, bins=30, alpha=0.7, label='Doğrulama')
plt.xlabel('RNA Dizisi Uzunluğu')
plt.ylabel('Frekans')
plt.title('RNA Dizisi Uzunluk Dağılımı')
plt.legend()
plt.savefig(os.path.join(output_dir, 'rna_length_distribution.png'))
plt.close()

# Daha detaylı uzunluk dağılımı (logaritmik ölçekte)
plt.figure(figsize=(12, 6))
plt.hist(train_seq_lengths, bins=30, alpha=0.7, label='Eğitim')
plt.hist(test_seq_lengths, bins=30, alpha=0.7, label='Test')
plt.hist(validation_seq_lengths, bins=30, alpha=0.7, label='Doğrulama')
plt.xlabel('RNA Dizisi Uzunluğu')
plt.ylabel('Frekans (logaritmik)')
plt.title('RNA Dizisi Uzunluk Dağılımı (Logaritmik)')
plt.yscale('log')
plt.legend()
plt.savefig(os.path.join(output_dir, 'rna_length_distribution_log.png'))
plt.close()

# Nükleotid kompozisyonu analizi
def nucleotide_composition(sequences):
    """Her nükleotidin frekansını hesaplar"""
    all_nucleotides = ''.join(sequences)
    counter = Counter(all_nucleotides)
    total = sum(counter.values())
    return {nucleotide: count / total * 100 for nucleotide, count in counter.items()}

# Nükleotid kompozisyonu
print("\n--- Nükleotid Kompozisyonu (%) ---")
train_comp = nucleotide_composition(train_sequences['sequence'])
test_comp = nucleotide_composition(test_sequences['sequence'])
validation_comp = nucleotide_composition(validation_sequences['sequence'])

# Nükleotid kompozisyonu tablosu
comp_df = pd.DataFrame({
    'Eğitim': train_comp,
    'Test': test_comp,
    'Doğrulama': validation_comp
})
print(comp_df)

# Nükleotid kompozisyonu grafiği
plt.figure(figsize=(10, 6))
comp_df.plot(kind='bar')
plt.ylabel('Frekans (%)')
plt.title('Nükleotid Kompozisyonu')
plt.savefig(os.path.join(output_dir, 'nucleotide_composition.png'))
plt.close()

# Di-nükleotid analizi
def dinucleotide_composition(sequences):
    """Her di-nükleotidin frekansını hesaplar"""
    dinucleotides = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            dinucleotides.append(seq[i: i +2])

    counter = Counter(dinucleotides)
    total = sum(counter.values())
    return {dinuc: count / total * 100 for dinuc, count in counter.items()}

# Di-nükleotid kompozisyonu
print("\n--- Di-nükleotid Kompozisyonu (En Yaygın 10) ---")
train_dinuc = dinucleotide_composition(train_sequences['sequence'])
test_dinuc = dinucleotide_composition(test_sequences['sequence'])

# En yaygın 10 di-nükleotidi göster
top_train_dinuc = dict(sorted(train_dinuc.items(), key=lambda x: x[1], reverse=True)[:10])
top_test_dinuc = dict(sorted(test_dinuc.items(), key=lambda x: x[1], reverse=True)[:10])

print("Eğitim seti en yaygın di-nükleotidler:")
for dinuc, freq in top_train_dinuc.items():
    print(f"  {dinuc}: {freq:.2f}%")

print("\nTest seti en yaygın di-nükleotidler:")
for dinuc, freq in top_test_dinuc.items():
    print(f"  {dinuc}: {freq:.2f}%")

# GC içeriği analizi
print("\n--- GC İçeriği (%) ---")
train_gc = train_sequences['sequence'].apply(GC)
test_gc = test_sequences['sequence'].apply(GC)
validation_gc = validation_sequences['sequence'].apply(GC)

print(f"Eğitim GC%: {train_gc.mean():.2f} ± {train_gc.std():.2f}")
print(f"Test GC%: {test_gc.mean():.2f} ± {test_gc.std():.2f}")
print(f"Doğrulama GC%: {validation_gc.mean():.2f} ± {validation_gc.std():.2f}")

# GC içeriği grafiği
plt.figure(figsize=(10, 6))
plt.hist(train_gc, bins=20, alpha=0.7, label='Eğitim')
plt.hist(test_gc, bins=20, alpha=0.7, label='Test')
plt.hist(validation_gc, bins=20, alpha=0.7, label='Doğrulama')
plt.xlabel('GC İçeriği (%)')
plt.ylabel('Frekans')
plt.title('GC İçeriği Dağılımı')
plt.legend()
plt.savefig(os.path.join(output_dir, 'gc_content_distribution.png'))
plt.close()

# ----------------------------------------------------------------------------------
# 3. SEKONDER YAPI ANALİZİ
# ----------------------------------------------------------------------------------
print("\n=== 3. SEKONDER YAPI ANALİZİ ===")

# Vienna RNA paketi yoksa bu bölümü atla
if vienna_available:
    # Örnek olarak 10 RNA dizisi için sekonder yapı tahmini yap
    print("\n--- Sekonder Yapı Tahminleri (Örnek 10 RNA) ---")

    sample_sequences = train_sequences.sample(10)

    for idx, row in sample_sequences.iterrows():
        seq = row['sequence']
        target_id = row['target_id']

        # Dizinin çok uzun olması durumunda kısalt
        display_seq = seq if len(seq) <= 50 else seq[:47] + "..."

        # RNA sekonder yapısını tahmin et (dot-bracket notasyonu)
        structure, mfe = RNA.fold(seq)

        # Saç tokası (hairpin), kök (stem), döngü (loop) sayılarını hesapla
        hairpin_count = structure.count("(...)")  # Basitleştirilmiş hairpin arama
        stem_count = structure.count("(") - hairpin_count
        loop_count = structure.count(".")

        print(f"Target: {target_id}")
        print(f"Seq: {display_seq}")
        print(f"Str: {structure[:50] + '...' if len(structure) > 50 else structure}")
        print(f"MFE: {mfe:.2f} kcal/mol")
        print(f"Hairpin/Stem/Loop sayıları: {hairpin_count}/{stem_count}/{loop_count}")
        print("-" * 80)

    # Test ve Eğitim setlerindeki RNA'ların minimum serbest enerji dağılımları
    print("\n--- Minimum Serbest Enerji (MFE) Dağılımları ---")

    # MFE hesaplama fonksiyonu
    def calculate_mfe(sequence):
        return RNA.fold(sequence)[1]

    # Örnek olarak 100 RNA için MFE hesapla (tüm set çok zaman alabilir)
    train_sample = train_sequences.sample(min(100, len(train_sequences)))
    test_sample = test_sequences.sample(min(len(test_sequences), len(test_sequences)))

    train_mfe = train_sample['sequence'].apply(calculate_mfe)
    test_mfe = test_sample['sequence'].apply(calculate_mfe)

    print(f"Eğitim seti MFE (kcal/mol): {train_mfe.mean():.2f} ± {train_mfe.std():.2f}")
    print(f"Test seti MFE (kcal/mol): {test_mfe.mean():.2f} ± {test_mfe.std():.2f}")

    # MFE dağılımı grafiği
    plt.figure(figsize=(10, 6))
    plt.hist(train_mfe, bins=20, alpha=0.7, label='Eğitim')
    plt.hist(test_mfe, bins=20, alpha=0.7, label='Test')
    plt.xlabel('Minimum Serbest Enerji (kcal/mol)')
    plt.ylabel('Frekans')
    plt.title('RNA Yapılarının Minimum Serbest Enerji Dağılımı')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mfe_distribution.png'))
    plt.close()

    # MFE ve dizi uzunluğu arasındaki ilişki
    plt.figure(figsize=(10, 6))
    plt.scatter(train_sample['sequence'].apply(len), train_mfe, alpha=0.7, label='Eğitim')
    plt.scatter(test_sample['sequence'].apply(len), test_mfe, alpha=0.7, label='Test')
    plt.xlabel('RNA Uzunluğu')
    plt.ylabel('Minimum Serbest Enerji (kcal/mol)')
    plt.title('RNA Uzunluğu ve Minimum Serbest Enerji İlişkisi')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'length_vs_mfe.png'))
    plt.close()
else:
    print("Vienna RNA paketi kurulu olmadığı için sekonder yapı analizleri atlandı.")
    print("Kurulum için: pip install viennarna")

# ----------------------------------------------------------------------------------
# 4. KOORDİNAT ANALİZİ
# ----------------------------------------------------------------------------------
print("\n=== 4. KOORDİNAT ANALİZİ ===")

# Train Labels koordinat analizi
print("\n--- Koordinat Analizi (Train Labels) ---")
coord_columns = ['x_1', 'y_1', 'z_1']
coord_stats = train_labels[coord_columns].describe()
print(coord_stats)

# Koordinat dağılımı (3D scatter plot - ilk 1000 nokta)
print("\n3D koordinat dağılımı grafiği oluşturuluyor...")
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')

# Örnek olarak ilk 1000 noktayı çizdir
sample_size = min(1000, len(train_labels))
sample_data = train_labels.sample(sample_size)

ax.scatter3D(
    sample_data['x_1'],
    sample_data['y_1'],
    sample_data['z_1'],
    c=sample_data['resid'],
    cmap='viridis',
    s=10,
    alpha=0.7
)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Train Labels - 3D Koordinat Dağılımı (Örnek Noktalar)')
plt.savefig(os.path.join(output_dir, 'train_labels_3d_distribution.png'))
plt.close()

# Validation Labels analizi
print("\n--- Validation Labels Analizi ---")
# Validation labels'daki benzersiz yapı sayısını kontrol et
validation_coord_columns = [col for col in validation_labels.columns if col.startswith('x_') or
                            col.startswith('y_') or col.startswith('z_')]
print(f"Validation labels koordinat sütun sayısı: {len(validation_coord_columns)}")
print(f"Bu yapı sayısına denk gelir: {len(validation_coord_columns) // 3}")

# Örnek yapılardaki koordinat istatistikleri
x_columns = [col for col in validation_labels.columns if col.startswith('x_')][:5]  # İlk 5 yapı
for col in x_columns:
    print(f"\n{col} istatistikleri:")
    print(validation_labels[col].describe())

# Validation Labels'da geçersiz değerlerin analizi
print("\n--- Validation Labels Geçersiz Değer Analizi ---")
for i in range(1, 6):  # İlk 5 yapı için kontrol et
    col = f'x_{i}'
    if col in validation_labels.columns:
        invalid_count = validation_labels[col].apply(lambda x: x <= -1e17).sum()
        total_count = len(validation_labels)
        print(f"{col} sütunundaki geçersiz değer sayısı: {invalid_count} ({invalid_count /total_count *100:.2f}%)")

# Farklı yapıların karşılaştırılması (ilk 200 nokta, ilk 3 yapı)
print("\nFarklı yapıların karşılaştırması grafiği oluşturuluyor...")
plt.figure(figsize=(15, 5))

# Örnek olarak ilk 200 noktayı ve ilk 3 yapıyı al
sample_size = min(200, len(validation_labels))
sample_val_data = validation_labels.head(sample_size)

# Geçersiz değerleri filtrele (örnek olarak ilk yapıda)
valid_mask = sample_val_data['x_1'] > -1e17

if valid_mask.sum() > 0:
    for i, suffix in enumerate(['1', '2', '3']):
        ax = plt.subplot(1, 3, i + 1, projection='3d')
        ax.scatter3D(
            sample_val_data.loc[valid_mask, f'x_{suffix}'],
            sample_val_data.loc[valid_mask, f'y_{suffix}'],
            sample_val_data.loc[valid_mask, f'z_{suffix}'],
            c=sample_val_data.loc[valid_mask, 'resid'],
            cmap='viridis',
            s=5,
            alpha=0.7
        )
        ax.set_title(f'Yapı {suffix}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_structures_comparison.png'))
    plt.close()
else:
    print("Geçerli veri yok, grafik oluşturulamadı.")

# Yapılar arası korelasyon (validation)
print("\nYapılar arası korelasyon analizi yapılıyor...")
# İlk 5 yapı için x koordinatları arasındaki korelasyonu hesapla
x_cols = [f'x_{i}' for i in range(1, 6) if f'x_{i}' in validation_labels.columns]

if len(x_cols) > 1:
    # Geçersiz değerleri NaN ile değiştir
    corr_data = validation_labels[x_cols].copy()
    for col in x_cols:
        corr_data.loc[corr_data[col] <= -1e17, col] = np.nan

    # NaN değerleri olmadan korelasyon matrisi hesapla
    corr_matrix = corr_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('İlk 5 Yapının X Koordinatları Arasındaki Korelasyon')
    plt.savefig(os.path.join(output_dir, 'structure_correlation_x.png'))
    plt.close()

    print("İlk 5 yapının X koordinatları arasındaki korelasyon:")
    print(corr_matrix)
else:
    print("Yeterli veri yok, korelasyon analizi yapılamadı.")

# ----------------------------------------------------------------------------------
# 5. YAPILAR ARASI BENZERLİK ANALİZİ
# ----------------------------------------------------------------------------------
print("\n=== 5. YAPILAR ARASI BENZERLİK ANALİZİ ===")

# Validation Labels'taki ilk birkaç yapı için örnek bir RNA seçip RMSD hesapla
print("\n--- Yapılar Arası RMSD Analizi ---")

# Geçerli koordinatlara sahip bir RNA target_id'si bul
valid_targets = []

for target_id in validation_sequences['target_id'].unique():
    # Bu hedefin ilk nükleotidini bul
    target_rows = validation_labels[validation_labels['ID'].str.startswith(f"{target_id}_")]

    if len(target_rows) > 0 and target_rows['x_1'].iloc[0] > -1e17:
        valid_targets.append(target_id)
        if len(valid_targets) >= 3:  # İlk 3 geçerli hedef yeterli
            break

if len(valid_targets) > 0:
    print(f"RMSD analizi için seçilen hedef: {valid_targets[0]}")

    # Bu hedef için tüm nükleotidleri al
    target_rows = validation_labels[validation_labels['ID'].str.startswith(f"{valid_targets[0]}_")]

    # Bu hedefin uzunluğu
    num_residues = len(target_rows)
    print(f"Bu RNA'nın uzunluğu: {num_residues} nükleotid")

    # İlk 5 yapıyı karşılaştır (eğer geçerli koordinatlar varsa)
    valid_structures = []
    for i in range(1, 10):  # İlk 10 yapıdan geçerli olanları seç
        col_x = f'x_{i}'
        col_y = f'y_{i}'
        col_z = f'z_{i}'

        if col_x in target_rows.columns and col_y in target_rows.columns and col_z in target_rows.columns:
            # Bu yapıda geçerli koordinatlar var mı kontrol et
            if (target_rows[col_x] > -1e17).all():
                valid_structures.append(i)
                if len(valid_structures) >= 5:  # İlk 5 geçerli yapı yeterli
                    break

    if len(valid_structures) >= 2:
        print(f"Karşılaştırılacak yapılar: {valid_structures}")

        # RMSD matrisi oluştur
        rmsd_matrix = np.zeros((len(valid_structures), len(valid_structures)))

        # Her yapı çifti için RMSD hesapla
        for i, struct1 in enumerate(valid_structures):
            for j, struct2 in enumerate(valid_structures):
                if i <= j:  # Simetrik matris, sadece yarısını hesapla
                    coords1 = target_rows[[f'x_{struct1}', f'y_{struct1}', f'z_{struct1}']].values
                    coords2 = target_rows[[f'x_{struct2}', f'y_{struct2}', f'z_{struct2}']].values

                    # RMSD hesapla (basit şekilde)
                    squared_diff = np.sum((coords1 - coords2) ** 2, axis=1)
                    rmsd = np.sqrt(np.mean(squared_diff))

                    rmsd_matrix[i, j] = rmsd
                    rmsd_matrix[j, i] = rmsd  # Simetrik matris

        # RMSD matrisini görselleştir
        plt.figure(figsize=(8, 6))
        sns.heatmap(rmsd_matrix, annot=True, fmt=".2f", cmap="viridis",
                    xticklabels=[f"Yapı {s}" for s in valid_structures],
                    yticklabels=[f"Yapı {s}" for s in valid_structures])
        plt.title(f"Yapılar Arası RMSD Matrisi (Angstrom) - {valid_targets[0]}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'structure_rmsd_matrix.png'))
        plt.close()

        print("\nRMSD Matrisi:")
        for i, s1 in enumerate(valid_structures):
            row_str = [f"{rmsd_matrix[i, j]:.2f}" for j in range(len(valid_structures))]
            print(f"Yapı {s1}: {', '.join(row_str)}")

        # PCA ile yapıların benzerliğini analiz et
        print("\n--- Yapılar Arası PCA Analizi ---")

        # Her yapı için tüm koordinatları düzleştir
        flattened_structures = []
        for struct in valid_structures:
            coords = target_rows[[f'x_{struct}', f'y_{struct}', f'z_{struct}']].values.flatten()
            flattened_structures.append(coords)

        # PCA uygula
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(flattened_structures)

        # PCA sonuçlarını görselleştir
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], s=100)

        # Her noktayı yapı numarasıyla etiketle
        for i, struct in enumerate(valid_structures):
            plt.annotate(f"Yapı {struct}", (pca_result[i, 0], pca_result[i, 1]),
                         textcoords="offset points", xytext=(0 ,10), ha='center')

        plt.title(f"Yapılar Arası PCA Analizi - {valid_targets[0]}")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} varyans)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} varyans)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'structure_pca.png'))
        plt.close()

        print(f"PCA açıklanan varyans oranları: {pca.explained_variance_ratio_}")
    else:
        print("Karşılaştırma için yeterli geçerli yapı bulunamadı.")
else:
    print("RMSD analizi için geçerli koordinatlara sahip hedef bulunamadı.")

# ----------------------------------------------------------------------------------
# 6. TEMPORAL CUTOFF ANALİZİ
# ----------------------------------------------------------------------------------
print("\n=== 6. TEMPORAL CUTOFF ANALİZİ ===")

# Temporal cutoff analizi
print("\n--- Temporal Cutoff Analizi ---")
train_temporal = pd.to_datetime(train_sequences['temporal_cutoff'])
test_temporal = pd.to_datetime(test_sequences['temporal_cutoff'])
validation_temporal = pd.to_datetime(validation_sequences['temporal_cutoff'])

print(f"Eğitim temporal cutoff aralığı: {train_temporal.min()} - {train_temporal.max()}")
print(f"Test temporal cutoff aralığı: {test_temporal.min()} - {test_temporal.max()}")
print(f"Doğrulama temporal cutoff aralığı: {validation_temporal.min()} - {validation_temporal.max()}")

# Temporal cutoff dağılımı grafiği
plt.figure(figsize=(12, 6))

# Eğitim setini yıllara göre histogramla göster
plt.hist(train_temporal.dt.year, bins=np.arange(train_temporal.dt.year.min(), train_temporal.dt.year.max( ) +2) - 0.5,
         alpha=0.7, label='Eğitim')

# Test ve validation set çok küçük olduğu için scatter plot kullan
plt.scatter(test_temporal.dt.year, np.ones(len(test_temporal)), color='orange', s=100, label='Test')
plt.scatter(validation_temporal.dt.year, np.ones(len(validation_temporal) ) *2, color='green', s=100, label='Doğrulama')

plt.xlabel('Yıl')
plt.ylabel('Frekans')
plt.title('Temporal Cutoff Dağılımı')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, 'temporal_cutoff_distribution.png'))
plt.close()

# ----------------------------------------------------------------------------------
# 7. TEST ve VALIDATION HEDEFLERİ KARŞILAŞTIRMASI
# ----------------------------------------------------------------------------------
print("\n=== 7. TEST ve VALIDATION HEDEFLERİ KARŞILAŞTIRMASI ===")

# Test ve validation setlerindeki aynı hedefleri kontrol et
print("\n--- Test ve Validation Setlerindeki Ortak Hedefler ---")
test_targets = set(test_sequences['target_id'])
validation_targets = set(validation_sequences['target_id'])
common_targets = test_targets.intersection(validation_targets)

print(f"Test setindeki hedef sayısı: {len(test_targets)}")
print(f"Validation setindeki hedef sayısı: {len(validation_targets)}")
print(f"Ortak hedef sayısı: {len(common_targets)}")

if common_targets:
    print("\nOrtak hedefler:")
    for target in common_targets:
        test_seq = test_sequences[test_sequences['target_id'] == target]['sequence'].iloc[0]
        valid_seq = validation_sequences[validation_sequences['target_id'] == target]['sequence'].iloc[0]

        seq_match = "AYNI" if test_seq == valid_seq else "FARKLI"

        print(f"  {target}: Dizi uzunluğu {len(test_seq)}, Test ve Validation dizileri {seq_match}")

# Test ve validation hedeflerini karşılaştırma tablosu
print("\n--- Test ve Validation Hedeflerinin Karşılaştırması ---")

comparison_data = []
for target in common_targets:
    test_row = test_sequences[test_sequences['target_id'] == target].iloc[0]
    valid_row = validation_sequences[validation_sequences['target_id'] == target].iloc[0]

    comparison_data.append({
        'target_id': target,
        'test_length': len(test_row['sequence']),
        'valid_length': len(valid_row['sequence']),
        'test_gc': GC(test_row['sequence']),
        'valid_gc': GC(valid_row['sequence']),
        'test_cutoff': test_row['temporal_cutoff'],
        'valid_cutoff': valid_row['temporal_cutoff'],
        'description': test_row['description']
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df[['target_id', 'test_length', 'valid_length', 'test_gc', 'valid_gc']])

# Ortak hedeflerin özellik karşılaştırması (uzunluk, GC içeriği)
if len(comparison_df) > 0:
    plt.figure(figsize=(10, 6))

    # Uzunluk karşılaştırması
    plt.subplot(1, 2, 1)
    plt.scatter(comparison_df['test_length'], comparison_df['valid_length'])
    min_len = min(comparison_df['test_length'].min(), comparison_df['valid_length'].min())
    max_len = max(comparison_df['test_length'].max(), comparison_df['valid_length'].max())
    plt.plot([min_len, max_len], [min_len, max_len], 'r--')  # Eşitlik çizgisi
    plt.xlabel('Test Dizi Uzunluğu')
    plt.ylabel('Validation Dizi Uzunluğu')
    plt.title('Dizi Uzunluğu Karşılaştırması')

    # GC içeriği karşılaştırması
    plt.subplot(1, 2, 2)
    plt.scatter(comparison_df['test_gc'], comparison_df['valid_gc'])
    min_gc = min(comparison_df['test_gc'].min(), comparison_df['valid_gc'].min())
    max_gc = max(comparison_df['test_gc'].max(), comparison_df['valid_gc'].max())
    plt.plot([min_gc, max_gc], [min_gc, max_gc], 'r--')  # Eşitlik çizgisi
    plt.xlabel('Test GC İçeriği (%)')
    plt.ylabel('Validation GC İçeriği (%)')
    plt.title('GC İçeriği Karşılaştırması')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_validation_comparison.png'))
    plt.close()

# ----------------------------------------------------------------------------------
# 8. ÖZET ve ÖNERİLER
# ----------------------------------------------------------------------------------
print("\n=== 8. ÖZET ve ÖNERİLER ===")

print("""
RNA 3D Folding Yarışması - EDA Analizi Sonuçları ve Öneriler:

1. Veri Yapısı ve Zorluklar:
   - Eğitim seti (844 RNA) test/validation setlerine (12 RNA) göre çok daha büyük
   - Eğitim verileri için tek yapı, validation için 40 yapı var
   - Validation verileri çoğunlukla geçersiz koordinat değerleri içeriyor (-1e18)
   - Test ve validation setlerindeki hedefler birebir aynı

2. Veri Özellikleri:
   - RNA uzunlukları çok değişken (3-4298 nükleotid)
   - Çoğunlukla kısa RNA'lar (medyan ~40 nükleotid)
   - GC içeriği tüm setlerde %50-60 arasında
   - Test dizileri eğitim setine göre ortalamada daha uzun

3. Temporal Analiz:
   - Eğitim verileri 1995-2024 arasını kapsıyor
   - Test/validation verileri sadece 2022 yılından

4. Korelasyon Analizleri:
   - Bazı validation yapıları (3-4-5) arasında tam korelasyon var
   - Yapılar arasında gruplar/kümeler mevcut

Öneriler:

1. Veri Önişleme:
   - Validation verisindeki geçersiz değerleri (-1e18) temizle
   - Zaman sınırlamasına göre eğitim verilerini filtrele

2. Model Stratejisi:
   - Ensemble yaklaşımı kullan (5 farklı yapı tahmin etmek için)
   - Sequence-to-structure dönüşümü için derin öğrenme
   - Sekonder yapı tahminini ara adım olarak kullan

3. Değerlendirme:
   - TM-score optimizasyonu
   - Çeşitlilik-doğruluk dengesini gözet

4. Gelişmiş Teknikler:
   - PCA/kümeleme ile çeşitli yapılar üretme
   - Fizik-temelli kısıtlamalar uygulama
   - Transfer öğrenme (protein yapılarından)
""")

# EDA tamamlandı
end_time = time.time()
runtime = end_time - start_time

print(f"\nEDA tamamlandı! Çalışma süresi: {runtime:.2f} saniye")
print(f"Tüm grafikler ve analizler {output_dir} klasörüne kaydedildi.")
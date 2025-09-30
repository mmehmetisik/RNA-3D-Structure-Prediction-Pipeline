
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import os
from collections import Counter
#from Bio.SeqUtils import GC
import time


def GC(sequence):
    """Bir RNA/DNA dizisindeki G ve C bazlarının yüzdesini hesaplar"""
    sequence = sequence.upper()  # Tüm harfleri büyük harfe çevirelim
    gc_count = sequence.count('G') + sequence.count('C')
    total_length = len(sequence)
    if total_length == 0:
        return 0
    return (gc_count / total_length) * 100
# Başlangıç zamanı
start_time = time.time()

# Dosya yolları
data_dir = r"C:\Users\ASUS\Desktop\comp3\data"
output_dir = r"C:\Users\ASUS\Desktop\comp3"

# Çıktı dizinini oluştur (yoksa)
os.makedirs(output_dir, exist_ok=True)

# Veri dosyalarını yükle
print("Veri dosyaları yükleniyor...")
sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
test_sequences = pd.read_csv(os.path.join(data_dir, "test_sequences.csv"))
train_labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
train_sequences = pd.read_csv(os.path.join(data_dir, "train_sequences.csv"))
validation_labels = pd.read_csv(os.path.join(data_dir, "validation_labels.csv"))
validation_sequences = pd.read_csv(os.path.join(data_dir, "validation_sequences.csv"))

# Tüm veri setlerinin boyutunu yazdır
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

# Benzersiz target_id sayıları
print("\n--- Benzersiz Target ID Sayıları ---")
print(f"Train sequences: {train_sequences['target_id'].nunique()}")
print(f"Test sequences: {test_sequences['target_id'].nunique()}")
print(f"Validation sequences: {validation_sequences['target_id'].nunique()}")

# Train Labels'daki benzersiz ID sayısı
unique_ids_train_labels = len(set([id.split('_')[0] for id in train_labels['ID']]))
print(f"Train labels (unique targets): {unique_ids_train_labels}")

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

# RNA uzunluk dağılımı grafiği
plt.figure(figsize=(10, 6))
plt.hist(train_seq_lengths, bins=30, alpha=0.7, label='Eğitim')
plt.hist(test_seq_lengths, bins=30, alpha=0.7, label='Test')
plt.hist(validation_seq_lengths, bins=30, alpha=0.7, label='Doğrulama')
plt.xlabel('RNA Dizisi Uzunluğu')
plt.ylabel('Frekans')
plt.title('RNA Dizisi Uzunluk Dağılımı')
plt.legend()
plt.savefig(os.path.join(output_dir, 'rna_length_distribution.png'))
plt.close()


# Nükleotid kompozisyonu analizi
def nucleotide_composition(sequences):
    """Her nükleotidin frekansını hesaplar"""
    all_nucleotides = ''.join(sequences)
    counter = Counter(all_nucleotides)
    total = sum(counter.values())
    return {nucleotide: count / total * 100 for nucleotide, count in counter.items()}


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

# GC içeriği analizi
print("\n--- GC İçeriği (%) ---")
train_gc = train_sequences['sequence'].apply(GC)
test_gc = test_sequences['sequence'].apply(GC)
validation_gc = validation_sequences['sequence'].apply(GC)

print(f"Eğitim GC%: {train_gc.mean():.2f}")
print(f"Test GC%: {test_gc.mean():.2f}")
print(f"Doğrulama GC%: {validation_gc.mean():.2f}")

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

# Koordinat analizi (Train Labels)
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

# Farklı yapıların karşılaştırılması (ilk 200 nokta, ilk 3 yapı)
print("\nFarklı yapıların karşılaştırması grafiği oluşturuluyor...")
plt.figure(figsize=(15, 5))

# Örnek olarak ilk 200 noktayı ve ilk 3 yapıyı al
sample_size = min(200, len(validation_labels))
sample_val_data = validation_labels.head(sample_size)

for i, suffix in enumerate(['1', '2', '3']):
    ax = plt.subplot(1, 3, i + 1, projection='3d')
    ax.scatter3D(
        sample_val_data[f'x_{suffix}'],
        sample_val_data[f'y_{suffix}'],
        sample_val_data[f'z_{suffix}'],
        c=sample_val_data['resid'],
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

# Yapılar arası korelasyon (validation)
print("\nYapılar arası korelasyon analizi yapılıyor...")
# İlk 5 yapı için x koordinatları arasındaki korelasyonu hesapla
x_cols = [f'x_{i}' for i in range(1, 6)]
corr_matrix = validation_labels[x_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('İlk 5 Yapının X Koordinatları Arasındaki Korelasyon')
plt.savefig(os.path.join(output_dir, 'structure_correlation_x.png'))
plt.close()

# Test veri seti analizi
print("\n--- Test Veri Seti Analizi ---")
# Test dizileri ve train/validation dizileri arasındaki benzerlikler
print(f"Test dizilerinin ortalama uzunluğu: {test_seq_lengths.mean():.2f}")
print(f"Eğitim dizilerinin ortalama uzunluğu: {train_seq_lengths.mean():.2f}")

# Temporal cutoff analizi
print("\n--- Temporal Cutoff Analizi ---")
train_temporal = pd.to_datetime(train_sequences['temporal_cutoff'])
test_temporal = pd.to_datetime(test_sequences['temporal_cutoff'])
validation_temporal = pd.to_datetime(validation_sequences['temporal_cutoff'])

print(f"Eğitim temporal cutoff aralığı: {train_temporal.min()} - {train_temporal.max()}")
print(f"Test temporal cutoff aralığı: {test_temporal.min()} - {test_temporal.max()}")
print(f"Doğrulama temporal cutoff aralığı: {validation_temporal.min()} - {validation_temporal.max()}")

# Birkaç hedef için RNA dizilerini ve etiketleri incele
print("\n--- Hedef ve Etiketleri İnceleme ---")
# Örnek bir hedef seç
sample_target = train_sequences['target_id'].iloc[0]
print(f"Örnek hedef: {sample_target}")

# Bu hedef için dizilimi ve özellikleri al
target_seq = train_sequences[train_sequences['target_id'] == sample_target]
print("\nDizilim ve özellikleri:")
print(target_seq[['sequence', 'temporal_cutoff', 'description']].iloc[0])

# Bu hedef için etiketleri al
target_labels = train_labels[train_labels['ID'].str.startswith(sample_target)]
print(f"\nBu hedef için toplam {len(target_labels)} etiket var")
print("\nİlk 5 etiket:")
print(target_labels.head())

# Örnek bir koordinat dosyası oluştur (ilk 3 test hedefi için)
print("\n--- Örnek Koordinat Dosyası Oluşturuluyor ---")
test_sample = test_sequences.head(3)
sample_coords = []

for idx, row in test_sample.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']

    for i, nucleotide in enumerate(sequence):
        resid = i + 1
        entry = {
            'ID': f"{target_id}_{resid}",
            'resname': nucleotide,
            'resid': resid
        }

        # 5 set rastgele koordinat ekle (örnek olarak)
        for j in range(1, 6):
            entry[f'x_{j}'] = np.random.normal(0, 10)
            entry[f'y_{j}'] = np.random.normal(0, 10)
            entry[f'z_{j}'] = np.random.normal(0, 10)

        sample_coords.append(entry)

sample_submission_df = pd.DataFrame(sample_coords)
print(f"Örnek koordinat dosyası boyutu: {sample_submission_df.shape}")
print("\nÖrnek koordinat dosyası ilk 3 satır:")
print(sample_submission_df.head(3))

# Örnek koordinat dosyasını kaydet
sample_submission_df.to_csv(os.path.join(output_dir, 'sample_coords_example.csv'), index=False)

# Test ve validation setlerindeki aynı hedefleri kontrol et
print("\n--- Test ve Validation Setlerindeki Ortak Hedefler ---")
test_targets = set(test_sequences['target_id'])
validation_targets = set(validation_sequences['target_id'])
common_targets = test_targets.intersection(validation_targets)

print(f"Test setindeki hedef sayısı: {len(test_targets)}")
print(f"Validation setindeki hedef sayısı: {len(validation_targets)}")
print(f"Ortak hedef sayısı: {len(common_targets)}")
if common_targets:
    print(f"Ortak hedefler: {common_targets}")

# EDA tamamlandı
end_time = time.time()
runtime = end_time - start_time

print(f"\nEDA tamamlandı! Çalışma süresi: {runtime:.2f} saniye")
print(f"Tüm grafikler ve analizler {output_dir} klasörüne kaydedildi.")
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

# Dosya yolları
data_dir = r"C:\Users\ASUS\Desktop\comp3\data"
output_dir = r"C:\Users\ASUS\Desktop\comp3"

# Çıktı dizinini oluştur (yoksa)
os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)

# Veri dosyalarını yükle
print("Veri dosyaları yükleniyor...")
sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
test_sequences = pd.read_csv(os.path.join(data_dir, "test_sequences.csv"))
train_labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
train_sequences = pd.read_csv(os.path.join(data_dir, "train_sequences.csv"))
validation_labels = pd.read_csv(os.path.join(data_dir, "validation_labels.csv"))
validation_sequences = pd.read_csv(os.path.join(data_dir, "validation_sequences.csv"))

print("Pre-processing başlatılıyor...")

# 1. Temporal Filtreleme: Test tarihlerinden önce yayınlanan eğitim verilerini kullan
# Test ve validation setlerindeki en erken tarih
test_earliest_date = pd.to_datetime(test_sequences['temporal_cutoff']).min()
print(f"Test setindeki en erken tarih: {test_earliest_date}")

# Eğitim verilerini tarihe göre filtrele
train_sequences['temporal_cutoff'] = pd.to_datetime(train_sequences['temporal_cutoff'])
filtered_train_sequences = train_sequences[train_sequences['temporal_cutoff'] < test_earliest_date]
print(f"Filtreleme sonrası eğitim seti boyutu: {len(filtered_train_sequences)} / {len(train_sequences)}")


# 2. Özellik Mühendisliği: RNA dizileri için özellikler oluştur
def extract_features(sequences_df):
    """RNA dizilerinden kullanışlı özellikler çıkarır"""
    features = []

    for idx, row in sequences_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']

        # Temel özellikler
        length = len(sequence)
        gc_content = (sequence.count('G') + sequence.count('C')) / length * 100
        a_content = sequence.count('A') / length * 100
        u_content = sequence.count('U') / length * 100

        # Nükleotid çiftleri sayısı (basit bir özellik)
        gu_pairs = min(sequence.count('G'), sequence.count('U'))
        au_pairs = min(sequence.count('A'), sequence.count('U'))
        gc_pairs = min(sequence.count('G'), sequence.count('C'))

        features.append({
            'target_id': target_id,
            'length': length,
            'gc_content': gc_content,
            'a_content': a_content,
            'u_content': u_content,
            'gu_pairs': gu_pairs,
            'au_pairs': au_pairs,
            'gc_pairs': gc_pairs
        })

    return pd.DataFrame(features)


# Eğitim, test ve validation setleri için özellikler oluştur
train_features = extract_features(filtered_train_sequences)
test_features = extract_features(test_sequences)
validation_features = extract_features(validation_sequences)

print(f"Eğitim özellikleri: {train_features.shape}")
print(f"Test özellikleri: {test_features.shape}")

# 3. Validation verilerindeki geçersiz değerleri temizle
# -1e18 değerleri NaN ile değiştir
print("Validation etiketlerindeki geçersiz değerler temizleniyor...")
invalid_value = -1.0e18
validation_labels_cleaned = validation_labels.copy()

for col in validation_labels.columns:
    if col.startswith('x_') or col.startswith('y_') or col.startswith('z_'):
        # Geçersiz değerleri NaN yap
        validation_labels_cleaned[col] = validation_labels_cleaned[col].apply(
            lambda x: np.nan if x <= invalid_value + 1 else x
        )

# Kaç adet geçerli yapı kaldığını kontrol et
valid_structures = []
for i in range(1, 41):  # 40 yapı var
    # Her yapı için geçerli koordinat sayısı
    valid_count = (~validation_labels_cleaned[[f'x_{i}', f'y_{i}', f'z_{i}']].isna()).all(axis=1).sum()
    valid_structures.append((i, valid_count))

valid_structures_df = pd.DataFrame(valid_structures, columns=['structure_id', 'valid_count'])
print("Yapı başına geçerli koordinat sayısı:")
print(valid_structures_df.sort_values('valid_count', ascending=False).head(10))

# 4. Eğitim etiketlerini hazırla - BU KISIM DÜZELTİLDİ
print("Eğitim etiketleri hazırlanıyor...")

# ID'lerin doğru şekilde ayrıştırılması
train_labels['target_id'] = train_labels['ID'].str.rsplit('_', n=1).str[0]

# Filtrelenen eğitim dizileri için etiketleri al
filtered_train_ids = filtered_train_sequences['target_id'].tolist()
filtered_train_labels = train_labels[train_labels['target_id'].isin(filtered_train_ids)]

print(f"Filtreleme sonrası eğitim etiketleri: {len(filtered_train_labels)} / {len(train_labels)}")

# Eğer filtre sonrasında hala etiket yoksa, farklı bir ayrıştırma stratejisi dene
if len(filtered_train_labels) == 0:
    print("Alternatif ID eşleştirme stratejisi deneniyor...")

    # Her bir ID'nin parçalarını al (örn: "1SCL_A_1" -> ["1SCL", "A", "1"])
    id_parts = train_labels['ID'].str.split('_')

    # Sadece ilk bileşeni (ana hedef ID) kullan
    train_labels['target_id_alt'] = id_parts.str[0]

    # Filtrelenen hedeflere göre tekrar filtrele
    filtered_train_targets = set([target.split('_')[0] for target in filtered_train_ids])
    filtered_train_labels = train_labels[train_labels['target_id_alt'].isin(filtered_train_targets)]

    print(f"Alternatif strateji sonrası eğitim etiketleri: {len(filtered_train_labels)} / {len(train_labels)}")


# 5. Koordinatları normalize et
# Min-max normalizasyon kullan
def normalize_coordinates(labels_df, coord_columns):
    """Koordinatları [0, 1] aralığına normalize eder"""
    normalized_df = labels_df.copy()
    normalization_params = {}

    for col in coord_columns:
        col_min = labels_df[col].min()
        col_max = labels_df[col].max()
        range_val = col_max - col_min

        if range_val > 0:  # Sıfıra bölme hatası önlemi
            normalized_df[col] = (labels_df[col] - col_min) / range_val
            normalization_params[col] = (col_min, col_max)
        else:
            normalized_df[col] = labels_df[col]
            normalization_params[col] = (col_min, col_max)

    return normalized_df, normalization_params


# Eğitim koordinatlarını normalize et
if len(filtered_train_labels) > 0:
    train_coord_columns = ['x_1', 'y_1', 'z_1']
    normalized_train_labels, train_normalization_params = normalize_coordinates(
        filtered_train_labels, train_coord_columns
    )
    print("Koordinat normalizasyon parametreleri:", train_normalization_params)
else:
    print("UYARI: Eğitim etiketi yok, normalizasyon yapılamadı!")
    normalized_train_labels = filtered_train_labels
    train_normalization_params = {}

# 6. Veri setlerini kaydet
# İşlenmiş eğitim verileri
filtered_train_sequences.to_csv(os.path.join(output_dir, "processed", "filtered_train_sequences.csv"), index=False)
train_features.to_csv(os.path.join(output_dir, "processed", "train_features.csv"), index=False)
normalized_train_labels.to_csv(os.path.join(output_dir, "processed", "normalized_train_labels.csv"), index=False)

# İşlenmiş test verileri
test_features.to_csv(os.path.join(output_dir, "processed", "test_features.csv"), index=False)

# İşlenmiş validation verileri
validation_features.to_csv(os.path.join(output_dir, "processed", "validation_features.csv"), index=False)
validation_labels_cleaned.to_csv(os.path.join(output_dir, "processed", "cleaned_validation_labels.csv"), index=False)

# Normalizasyon parametrelerini kaydet
normalization_params = {
    'train_coord_columns': train_coord_columns,
    'train_normalization_params': str(train_normalization_params)
}

# JSON olarak kaydet
with open(os.path.join(output_dir, "processed", "normalization_params.json"), 'w') as f:
    json.dump(normalization_params, f)

print("Pre-processing tamamlandı! Dosyalar şuraya kaydedildi:", os.path.join(output_dir, "processed"))
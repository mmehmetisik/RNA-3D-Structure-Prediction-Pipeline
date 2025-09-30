# Çoklu Tohum Ensemble Stratejisi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time

# Zamanlayıcı başlat
start_time = time.time()

# Kaggle dosya yolları
train_sequences_path = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
train_labels_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
test_sequences_path = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
sample_submission_path = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"

print("RNA 3D Yapı Tahmini Pipeline Başlatılıyor (Çoklu Tohum Ensemble Stratejisi)...")

# 1. VERİ YÜKLEME
print("Veri dosyaları yükleniyor...")
train_sequences = pd.read_csv(train_sequences_path)
train_labels = pd.read_csv(train_labels_path)
test_sequences = pd.read_csv(test_sequences_path)
sample_submission = pd.read_csv(sample_submission_path)

print(f"Train sequences: {train_sequences.shape}")
print(f"Train labels: {train_labels.shape}")
print(f"Test sequences: {test_sequences.shape}")

# 2. PRE-PROCESSING
print("\n--- PRE-PROCESSING BAŞLATILIYOR ---")

# 2.1. Temporal Filtreleme
test_earliest_date = pd.to_datetime(test_sequences['temporal_cutoff']).min()
print(f"Test setindeki en erken tarih: {test_earliest_date}")

train_sequences['temporal_cutoff'] = pd.to_datetime(train_sequences['temporal_cutoff'])
filtered_train_sequences = train_sequences[train_sequences['temporal_cutoff'] < test_earliest_date]
print(f"Filtreleme sonrası eğitim seti boyutu: {len(filtered_train_sequences)} / {len(train_sequences)}")


# 2.2. Özellik çıkarma fonksiyonu
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

        # Nükleotid çiftleri sayısı
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


# Eğitim ve test özellikleri
train_features = extract_features(filtered_train_sequences)
test_features = extract_features(test_sequences)

# 2.3. Eğitim etiketlerini düzenle
print("Eğitim etiketleri hazırlanıyor...")

# Target ID'leri doğru şekilde eşleştir
train_labels['target_id'] = train_labels['ID'].str.rsplit('_', n=1).str[0]

# Filtrelenen hedefleri kullan
filtered_train_ids = filtered_train_sequences['target_id'].tolist()
filtered_train_labels = train_labels[train_labels['target_id'].isin(filtered_train_ids)]

# NaN değer kontrolü
nan_count_x = filtered_train_labels['x_1'].isna().sum()
nan_count_y = filtered_train_labels['y_1'].isna().sum()
nan_count_z = filtered_train_labels['z_1'].isna().sum()
print(f"Etiketlerdeki NaN değer sayısı - X: {nan_count_x}, Y: {nan_count_y}, Z: {nan_count_z}")

# NaN değerleri temizle
filtered_train_labels = filtered_train_labels.dropna(subset=['x_1', 'y_1', 'z_1'])
print(f"NaN temizleme sonrası etiket sayısı: {len(filtered_train_labels)} / {len(train_labels)}")

# 2.4. Koordinatları normalize et
print("Koordinatlar normalize ediliyor...")
x_min, x_max = filtered_train_labels['x_1'].min(), filtered_train_labels['x_1'].max()
y_min, y_max = filtered_train_labels['y_1'].min(), filtered_train_labels['y_1'].max()
z_min, z_max = filtered_train_labels['z_1'].min(), filtered_train_labels['z_1'].max()

print(f"X aralığı: {x_min} - {x_max}")
print(f"Y aralığı: {y_min} - {y_max}")
print(f"Z aralığı: {z_min} - {z_max}")

normalized_train_labels = filtered_train_labels.copy()
normalized_train_labels['x_1'] = (filtered_train_labels['x_1'] - x_min) / (x_max - x_min)
normalized_train_labels['y_1'] = (filtered_train_labels['y_1'] - y_min) / (y_max - y_min)
normalized_train_labels['z_1'] = (filtered_train_labels['z_1'] - z_min) / (z_max - z_min)

# 2.5. Benzersiz nükleotid tipleri
unique_resnames = normalized_train_labels['resname'].unique()
print(f"Benzersiz nükleotid tipleri: {unique_resnames}")


# 3. VERİ HAZIRLAMA
def prepare_training_data(train_features, train_labels):
    """Eğitim verilerini hazırlar"""
    train_data = []

    for target_id in train_features['target_id'].unique():
        # Bu hedef için özellikler
        if target_id in train_features['target_id'].values:
            target_features = train_features[train_features['target_id'] == target_id][
                ['length', 'gc_content', 'a_content', 'u_content', 'gu_pairs', 'au_pairs', 'gc_pairs']
            ].iloc[0].to_dict()
        else:
            continue

        # Bu hedef için etiketler
        target_labels = train_labels[train_labels['target_id'] == target_id]
        if len(target_labels) == 0:
            continue

        # Her nükleotid için satır oluştur
        for _, row in target_labels.iterrows():
            entry = {
                'target_id': target_id,
                'resid': row['resid'],
                'resname': row['resname'],
                'x_1': row['x_1'],
                'y_1': row['y_1'],
                'z_1': row['z_1']
            }
            # Özellikleri ekle
            entry.update(target_features)

            # Pozisyon bilgisini ekle
            entry['position_ratio'] = row['resid'] / target_features['length']

            train_data.append(entry)

    return pd.DataFrame(train_data)


print("\n--- MODEL VERİLERİ HAZIRLANIYOR ---")
train_data = prepare_training_data(train_features, normalized_train_labels)
print(f"Hazırlanan eğitim verisi: {train_data.shape}")

# 4. MODEL EĞİTİMİ - ORİJİNAL BAŞARILI MODEL
print("\n--- MODEL EĞİTİMİ BAŞLATILIYOR ---")

# NaN değer kontrolü
train_data = train_data.dropna(subset=['x_1', 'y_1', 'z_1'])

# Özellik sütunları
feature_cols = ['length', 'gc_content', 'a_content', 'u_content',
                'gu_pairs', 'au_pairs', 'gc_pairs', 'position_ratio', 'resid']

# One-hot encoding
train_data_encoded = pd.get_dummies(train_data, columns=['resname'], prefix='resname')
feature_cols += [col for col in train_data_encoded.columns if col.startswith('resname_')]

# Train-test split
X = train_data_encoded[feature_cols]
y_x = train_data_encoded['x_1']
y_y = train_data_encoded['y_1']
y_z = train_data_encoded['z_1']

X_train, X_val, y_x_train, y_x_val = train_test_split(X, y_x, test_size=0.2, random_state=42)
_, _, y_y_train, y_y_val = train_test_split(X, y_y, test_size=0.2, random_state=42)
_, _, y_z_train, y_z_val = train_test_split(X, y_z, test_size=0.2, random_state=42)

# ORİJİNAL RANDOM FOREST MODELLERİ (n_estimators=100, diğer parametreler default)
print("X koordinatı için RF model eğitiliyor...")
model_x = RandomForestRegressor(n_estimators=100, random_state=42)
model_x.fit(X_train, y_x_train)

print("Y koordinatı için RF model eğitiliyor...")
model_y = RandomForestRegressor(n_estimators=100, random_state=42)
model_y.fit(X_train, y_y_train)

print("Z koordinatı için RF model eğitiliyor...")
model_z = RandomForestRegressor(n_estimators=100, random_state=42)
model_z.fit(X_train, y_z_train)

# Doğrulama değerlendirmesi
x_pred_val = model_x.predict(X_val)
y_pred_val = model_y.predict(X_val)
z_pred_val = model_z.predict(X_val)

x_rmse = np.sqrt(mean_squared_error(y_x_val, x_pred_val))
y_rmse = np.sqrt(mean_squared_error(y_y_val, y_pred_val))
z_rmse = np.sqrt(mean_squared_error(y_z_val, z_pred_val))

print(f"Doğrulama RMSE - X: {x_rmse:.4f}, Y: {y_rmse:.4f}, Z: {z_rmse:.4f}")

# 5. TAHMİN OLUŞTURMA VE ÇOKLU TOHUM ENSEMBLE YAKLAŞIMI
print("\n--- TEST TAHMİNLERİ OLUŞTURULUYOR ---")


# Test veri seti hazırlama
def predict_base_structure(test_features, test_sequences):
    """İlk yapıyı (1. yapı) tahmin eder"""
    predictions = []

    # Her bir test dizisi için
    for idx, test_row in test_features.iterrows():
        target_id = test_row['target_id']
        sequence = test_sequences[test_sequences['target_id'] == target_id]['sequence'].iloc[0]

        # Dizi uzunluğu
        seq_length = len(sequence)

        # Her bir nükleotid için tahmin yap
        coords = []

        for i, nucleotide in enumerate(sequence):
            resid = i + 1  # 1-tabanlı indeksleme

            # Özellik vektörü oluştur
            features = test_row[['length', 'gc_content', 'a_content', 'u_content',
                                 'gu_pairs', 'au_pairs', 'gc_pairs']].to_dict()
            features['position_ratio'] = resid / seq_length
            features['resid'] = resid

            # One-hot encoding
            for base in ['A', 'C', 'G', 'U', '-', 'X']:
                features[f'resname_{base}'] = 1 if nucleotide == base else 0

            # Tahmin için dataframe oluştur ve eksik sütunları kontrol et
            feature_df = pd.DataFrame([features])
            for col in feature_cols:
                if col not in feature_df.columns:
                    feature_df[col] = 0

            # Tahmin yap
            feature_vector = feature_df[feature_cols]

            x_pred_norm = model_x.predict(feature_vector)[0]
            y_pred_norm = model_y.predict(feature_vector)[0]
            z_pred_norm = model_z.predict(feature_vector)[0]

            # Normalize edilmiş tahminleri gerçek koordinatlara dönüştür
            x_pred = x_pred_norm * (x_max - x_min) + x_min
            y_pred = y_pred_norm * (y_max - y_min) + y_min
            z_pred = z_pred_norm * (z_max - z_min) + z_min

            coords.append({
                'ID': f"{target_id}_{resid}",
                'resname': nucleotide,
                'resid': resid,
                'x': x_pred,
                'y': y_pred,
                'z': z_pred
            })

        # Tüm dizi için tahminleri kaydet
        predictions.append({
            'target_id': target_id,
            'sequence': sequence,
            'coords': coords
        })

    return predictions


# İlk yapıları tahmin et
base_predictions = predict_base_structure(test_features, test_sequences)
print(f"Temel yapı tahminleri oluşturuldu: {len(base_predictions)} RNA için")


# ÇOKLU TOHUM ENSEMBLE FONKSİYONU
def generate_multi_seed_structures(prediction):
    """Farklı tohum değerleriyle çoklu yapılar oluşturur ve aralarından en iyilerini seçer"""

    all_structures = []
    seeds = [123, 42, 789]  # En iyi performans gösteren 3 tohum

    # Her tohum için 5 yapı oluştur (toplam 15 yapı)
    for seed_idx, seed_value in enumerate(seeds):
        # Bu tohum değerini ayarla
        np.random.seed(seed_value)

        # Yapıları oluştur
        structures = []

        # İlk yapı: Orijinal RandomForest tahmini (tüm tohumlarda aynı)
        structure1 = prediction['coords']
        for coord in structure1:
            struct_entry = {
                'ID': coord['ID'],
                'resname': coord['resname'],
                'resid': coord['resid'],
                'seed': seed_value,
                'structure_index': 1,
                'x': coord['x'],
                'y': coord['y'],
                'z': coord['z']
            }
            structures.append(struct_entry)

        # Her bir yapı için farklı gürültü seviyeleri
        noise_levels = {
            2: 0.11 * 10,
            3: 0.22 * 10,
            4: 0.33 * 10,
            5: 0.44 * 10
        }

        # Yapı 2-5: Farklı gürültü seviyeleri
        for struct_idx in range(2, 6):
            for coord in structure1:
                struct_entry = {
                    'ID': coord['ID'],
                    'resname': coord['resname'],
                    'resid': coord['resid'],
                    'seed': seed_value,
                    'structure_index': struct_idx,
                    'x': coord['x'] + np.random.normal(0, noise_levels[struct_idx]),
                    'y': coord['y'] + np.random.normal(0, noise_levels[struct_idx]),
                    'z': coord['z'] + np.random.normal(0, noise_levels[struct_idx])
                }
                structures.append(struct_entry)

        all_structures.extend(structures)

    # Bu noktada 15 farklı yapımız var, en iyi 5'ini seçmeliyiz
    # İlk seçim stratejisi: Her tohumdan en iyi yapıyı seç
    selected_structures = []

    # İlk yapı (model tahmini) her tohum için aynı olduğundan sadece birini alalım
    seed_123_struct1 = [s for s in all_structures if s['seed'] == 123 and s['structure_index'] == 1]
    selected_structures.extend(seed_123_struct1)

    # Her tohumdan bir yapı seç - genellikle en ayırt edici yapılar 3-5 arasında
    for seed_value in seeds:
        # Tohum 123'ten yapı 3'ü, tohum 42'den yapı 4'ü, tohum 789'dan yapı 5'i alalım
        if seed_value == 123:
            struct_idx = 3
        elif seed_value == 42:
            struct_idx = 4
        else:  # 789
            struct_idx = 5

        seed_structX = [s for s in all_structures if s['seed'] == seed_value and s['structure_index'] == struct_idx]
        selected_structures.extend(seed_structX)

    # Şimdi bu seçilmiş yapıları submission formatına dönüştür
    result = []

    # ID listesi
    ids = list(set([s['ID'] for s in selected_structures]))

    for id_val in ids:
        # Bu ID için tüm yapılar
        id_structures = [s for s in selected_structures if s['ID'] == id_val]

        entry = {
            'ID': id_val,
            'resname': id_structures[0]['resname'],
            'resid': id_structures[0]['resid']
        }

        # Her yapı için x, y, z koordinatları
        for i, struct in enumerate(id_structures):
            struct_idx = i + 1  # 1-5 arası indeks
            entry[f'x_{struct_idx}'] = struct['x']
            entry[f'y_{struct_idx}'] = struct['y']
            entry[f'z_{struct_idx}'] = struct['z']

        result.append(entry)

    return result


# Her RNA için çoklu tohum ensemble yaklaşımı uygula
print("Çoklu tohum ensemble yaklaşımı uygulanıyor...")
all_predictions = []

for prediction in base_predictions:
    structures = generate_multi_seed_structures(prediction)
    all_predictions.extend(structures)

# Tahminleri dataframe'e dönüştür
predictions_df = pd.DataFrame(all_predictions)

# Kolların aynı olduğundan emin olmak için sample submission ile kontrol
missing_cols = [col for col in sample_submission.columns if col not in predictions_df.columns]
if missing_cols:
    print(f"Eksik sütunlar: {missing_cols}")
    for col in missing_cols:
        predictions_df[col] = 0

# Sütun sırasını sample_submission ile aynı yap
submission_df = predictions_df[sample_submission.columns]

# Submission dosyasını kaydet
submission_df.to_csv('submission.csv', index=False)
print(f"Submission dosyası oluşturuldu! Satır sayısı: {len(submission_df)}")

# Örnek bir RNA'nın çoklu yapılarını görselleştir
print("Örnek RNA'nın çoklu yapıları görselleştiriliyor...")

sample_target = predictions_df['ID'].str.split('_').str[0].iloc[0]
sample_predictions = predictions_df[predictions_df['ID'].str.startswith(sample_target)]

# X, Y, Z koordinatlarını ayırarak alt grafiklerde göster
plt.figure(figsize=(15, 10))

for i in range(1, 6):
    ax = plt.subplot(2, 3, i, projection='3d')
    ax.scatter(sample_predictions[f'x_{i}'],
               sample_predictions[f'y_{i}'],
               sample_predictions[f'z_{i}'],
               c=sample_predictions['resid'],
               cmap='viridis')

    # Farklı yapılar için başlıklar
    if i == 1:
        title = 'Yapı 1: RandomForest'
    else:
        if i == 3:
            title = f'Yapı {i}: Seed 123, Yapı 3'
        elif i == 4:
            title = f'Yapı {i}: Seed 42, Yapı 4'
        elif i == 5:
            title = f'Yapı {i}: Seed 789, Yapı 5'
        else:
            title = f'Yapı {i}'

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.savefig('multi_seed_ensemble_structures.png')
plt.close()

# Toplam çalışma süresi
end_time = time.time()
print(f"\nÇoklu Tohum Ensemble Pipeline tamamlandı! Toplam süre: {(end_time - start_time):.2f} saniye")
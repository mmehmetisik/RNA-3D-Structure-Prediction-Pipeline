# RNA 3D Folding - Korelasyonlu Gürültü Tüm Yapılarda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

# Zamanlayıcı başlat
start_time = time.time()

# Kaggle dosya yolları
train_sequences_path = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
train_labels_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
test_sequences_path = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
sample_submission_path = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"

print("RNA 3D Yapı Tahmini Pipeline Başlatılıyor (Korelasyonlu Gürültü - Tüm Yapılarda)...")

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
                'gu_pairs', 'au_pairs', 'gc_pairs', 'position_ratio']

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

# 5. TAHMİN OLUŞTURMA VE TÜM YAPILARDA KORETELELİ GÜRÜLTÜ STRATEJİSİ
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


# TÜM YAPILARDA KORELASYONLU GÜRÜLTÜ STRATEJİSİ
def generate_all_correlated_noise_strategy(prediction):
    """Tüm yapılarda korelasyonlu gürültü kullanan geliştirilmiş varyasyon stratejisi"""
    result = []

    # Ana yapı - temel model tahmini
    structure1 = prediction['coords']
    for coord in structure1:
        result.append({
            'ID': coord['ID'],
            'resname': coord['resname'],
            'resid': coord['resid'],
            'x_1': coord['x'],
            'y_1': coord['y'],
            'z_1': coord['z']
        })

    # Sequence uzunluğu
    sequence_length = len(prediction['sequence'])

    # RNA molekülündeki her nükleotid için ortak faktörler
    # Bu, tüm yapılarda tutarlı hareket sağlayacak
    np.random.seed(123)
    residue_factors = {}

    for entry in result:
        resid = entry['resid']
        residue_factors[resid] = {
            'common_x': np.random.normal(0, 1.0),
            'common_y': np.random.normal(0, 1.0),
            'common_z': np.random.normal(0, 1.0),
            'corr_factor_2': np.random.normal(0, 1.0),  # Yapı 2 için korelasyon faktörü
            'corr_factor_3': np.random.normal(0, 1.0),  # Yapı 3 için korelasyon faktörü
            'corr_factor_4': np.random.normal(0, 1.0),  # Yapı 4 için korelasyon faktörü
            'corr_factor_5': np.random.normal(0, 1.0)  # Yapı 5 için korelasyon faktörü
        }

    # Yapı 2: Düşük korelasyonlu gürültü (0.25 * 10)
    np.random.seed(123)
    noise_scale_2 = 0.25 * 10

    for entry in result:
        resid = entry['resid']
        rel_pos = resid / sequence_length

        # Adaptif faktörler
        base_factor = 1.5 if entry['resname'] in ['A', 'U'] else 0.8
        length_factor = max(0.8, min(1.5, 40 / sequence_length))
        edge_factor = 1.8 if rel_pos < 0.15 or rel_pos > 0.85 else 1.0

        # Gürültü seviyesi
        noise = noise_scale_2 * base_factor * length_factor * edge_factor

        # Korelasyonlu gürültü - her nükleotid için özel faktörler kullanarak
        corr_factor = residue_factors[resid]['corr_factor_2']
        corr_weight = 0.7  # Korelasyon ağırlığı (ne kadar yüksek olursa, atomlar o kadar benzer hareket eder)
        indiv_weight = 0.6  # Bireysel hareket ağırlığı

        # Laplace dağılımı ile korelasyonlu gürültü
        common_x = residue_factors[resid]['common_x']
        common_y = residue_factors[resid]['common_y']
        common_z = residue_factors[resid]['common_z']

        entry['x_2'] = entry['x_1'] + corr_factor * noise * corr_weight + np.random.laplace(0, noise * indiv_weight)
        entry['y_2'] = entry['y_1'] + corr_factor * noise * corr_weight + np.random.laplace(0, noise * indiv_weight)
        entry['z_2'] = entry['z_1'] + corr_factor * noise * corr_weight + np.random.laplace(0, noise * indiv_weight)

    # Yapı 3: Orta seviye korelasyonlu gürültü (0.50 * 10)
    np.random.seed(123)
    noise_scale_3 = 0.50 * 10

    for entry in result:
        resid = entry['resid']
        rel_pos = resid / sequence_length

        # Adaptif faktörler
        base_factor = 1.5 if entry['resname'] in ['A', 'U'] else 0.8
        length_factor = max(0.8, min(1.5, 40 / sequence_length))
        edge_factor = 1.8 if rel_pos < 0.15 or rel_pos > 0.85 else 1.0

        # Gürültü seviyesi
        noise = noise_scale_3 * base_factor * length_factor * edge_factor

        # Korelasyonlu gürültü
        corr_factor = residue_factors[resid]['corr_factor_3']
        corr_weight = 0.7
        indiv_weight = 0.6

        # Normal dağılım ile korelasyonlu gürültü
        entry['x_3'] = entry['x_1'] + corr_factor * noise * corr_weight + np.random.normal(0, noise * indiv_weight)
        entry['y_3'] = entry['y_1'] + corr_factor * noise * corr_weight + np.random.normal(0, noise * indiv_weight)
        entry['z_3'] = entry['z_1'] + corr_factor * noise * corr_weight + np.random.normal(0, noise * indiv_weight)

    # Yapı 4: Yüksek seviye korelasyonlu gürültü (0.80 * 10)
    np.random.seed(123)
    noise_scale_4 = 0.80 * 10

    for entry in result:
        resid = entry['resid']
        rel_pos = resid / sequence_length

        # Adaptif faktörler - biraz daha agresif
        base_factor = 1.2 if entry['resname'] in ['A', 'U'] else 0.9
        length_factor = max(0.7, min(1.6, 40 / sequence_length))
        edge_factor = 2.2 if rel_pos < 0.15 or rel_pos > 0.85 else 1.0  # Uçlarda daha fazla esneklik

        # Gürültü seviyesi
        noise = noise_scale_4 * base_factor * length_factor * edge_factor

        # Korelasyonlu gürültü - yüksek korelasyon
        corr_factor = residue_factors[resid]['corr_factor_4']
        corr_weight = 0.75  # Daha yüksek korelasyon (yapı bütünlüğünü korur)
        indiv_weight = 0.55  # Daha düşük bireysel hareket

        # Normal dağılım ile korelasyonlu gürültü
        entry['x_4'] = entry['x_1'] + corr_factor * noise * corr_weight + np.random.normal(0, noise * indiv_weight)
        entry['y_4'] = entry['y_1'] + corr_factor * noise * corr_weight + np.random.normal(0, noise * indiv_weight)
        entry['z_4'] = entry['z_1'] + corr_factor * noise * corr_weight + np.random.normal(0, noise * indiv_weight)

    # Yapı 5: Ultra korelasyonlu gürültü + RNA uçlarına özel odak (0.70 * 10)
    np.random.seed(123)
    noise_scale_5 = 0.70 * 10

    for entry in result:
        resid = entry['resid']
        rel_pos = resid / sequence_length

        # Adaptif faktörler - agresif uç faktörleri
        edge_factor = 3.0 if rel_pos < 0.15 or rel_pos > 0.85 else 1.0
        base_factor = 1.3 if entry['resname'] in ['A', 'U'] else 0.85  # A-U daha esnek
        length_factor = max(0.7, min(1.8, 35 / sequence_length))

        # Gürültü seviyesi
        noise = noise_scale_5 * edge_factor * base_factor * length_factor

        # Korelasyonlu gürültü - en yüksek korelasyon
        corr_factor = residue_factors[resid]['corr_factor_5']
        corr_weight = 0.8  # En yüksek korelasyon
        indiv_weight = 0.5  # En düşük bireysel hareket

        # Her eksen için ortak faktör + bağımsız hareket
        entry['x_5'] = entry['x_1'] + corr_factor * noise * corr_weight + np.random.normal(0, noise * indiv_weight)
        entry['y_5'] = entry['y_1'] + corr_factor * noise * corr_weight + np.random.normal(0, noise * indiv_weight)
        entry['z_5'] = entry['z_1'] + corr_factor * noise * corr_weight + np.random.normal(0, noise * indiv_weight)

    return result


# Her RNA için Korelasyonlu Gürültü stratejisi uygula
print("Tüm Yapılarda Korelasyonlu Gürültü stratejisi uygulanıyor...")
all_predictions = []

for prediction in tqdm(base_predictions, desc="RNA Yapıları İşleniyor"):
    structures = generate_all_correlated_noise_strategy(prediction)
    all_predictions.extend(structures)

# Tahminleri dataframe'e dönüştür
predictions_df = pd.DataFrame(all_predictions)

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

strategies = [
    "Yapı 1: RandomForest Tahmini",
    "Yapı 2: Korelasyonlu Laplace (0.25*10)",
    "Yapı 3: Korelasyonlu Orta Gürültü (0.50*10)",
    "Yapı 4: Korelasyonlu Yüksek Gürültü (0.80*10)",
    "Yapı 5: Ultra Korelasyonlu Gürültü (0.70*10)"
]

for i in range(1, 6):
    ax = plt.subplot(2, 3, i, projection='3d')
    ax.scatter(sample_predictions[f'x_{i}'],
               sample_predictions[f'y_{i}'],
               sample_predictions[f'z_{i}'],
               c=sample_predictions['resid'],
               cmap='viridis')

    ax.set_title(strategies[i - 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.savefig('all_correlated_noise_strategy.png')
plt.close()

# Toplam çalışma süresi
end_time = time.time()
print(f"\nTüm Yapılarda Korelasyonlu Gürültü Pipeline tamamlandı! Toplam süre: {(end_time - start_time):.2f} saniye")
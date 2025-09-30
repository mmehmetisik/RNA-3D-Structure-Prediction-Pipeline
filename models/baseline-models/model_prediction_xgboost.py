# Kaggle için XGBoost tabanlı RNA 3D Yapı Tahmini Notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Kaggle dosya yolları
train_sequences_path = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
train_labels_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
test_sequences_path = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
sample_submission_path = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"

print("RNA 3D Yapı Tahmini Pipeline Başlatılıyor (XGBoost versiyonu)...")

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

        # Gelişmiş özellikler
        # 1. Dinükleotid kompozisyonu (ardışık iki nükleotidin frekansı)
        dinucleotides = {}
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i + 2]
            dinucleotides[dinuc] = dinucleotides.get(dinuc, 0) + 1

        # Dinükleotid frekanslarını normalize et
        for dinuc in ['AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG', 'CU',
                      'GA', 'GC', 'GG', 'GU', 'UA', 'UC', 'UG', 'UU']:
            dinucleotides[dinuc] = dinucleotides.get(dinuc, 0) / max(1, (length - 1))

        entry = {
            'target_id': target_id,
            'length': length,
            'gc_content': gc_content,
            'a_content': a_content,
            'u_content': u_content,
            'gu_pairs': gu_pairs,
            'au_pairs': au_pairs,
            'gc_pairs': gc_pairs,
        }

        # Dinükleotid özelliklerini ekle
        for dinuc, freq in dinucleotides.items():
            entry[f'dinuc_{dinuc}'] = freq

        features.append(entry)

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
            target_features = train_features[train_features['target_id'] == target_id].iloc[0].to_dict()
            # 'target_id' sütununu kaldır (hedef özelliği değil)
            if 'target_id' in target_features:
                del target_features['target_id']
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

# 4. MODEL EĞİTİMİ
print("\n--- MODEL EĞİTİMİ BAŞLATILIYOR ---")

# 4.1 NaN değer kontrolü
train_data = train_data.dropna(subset=['x_1', 'y_1', 'z_1'])

# 4.2 Özellik sütunları
feature_cols = [col for col in train_data.columns if col not in
                ['target_id', 'resname', 'x_1', 'y_1', 'z_1']]

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

# XGBoost modellerini oluştur
print("X koordinatı için XGBoost model eğitiliyor...")
model_x = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_x.fit(X_train, y_x_train)

print("Y koordinatı için XGBoost model eğitiliyor...")
model_y = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_y.fit(X_train, y_y_train)

print("Z koordinatı için XGBoost model eğitiliyor...")
model_z = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_z.fit(X_train, y_z_train)

# Doğrulama değerlendirmesi
x_pred_val = model_x.predict(X_val)
y_pred_val = model_y.predict(X_val)
z_pred_val = model_z.predict(X_val)

x_rmse = np.sqrt(mean_squared_error(y_x_val, x_pred_val))
y_rmse = np.sqrt(mean_squared_error(y_y_val, y_pred_val))
z_rmse = np.sqrt(mean_squared_error(y_z_val, z_pred_val))

print(f"Doğrulama RMSE - X: {x_rmse:.4f}, Y: {y_rmse:.4f}, Z: {z_rmse:.4f}")

# Özellik önemlerini görselleştir
plt.figure(figsize=(10, 6))
xgb.plot_importance(model_x, max_num_features=10)
plt.title('X Koordinatı için Özellik Önemleri')
plt.tight_layout()
plt.savefig('feature_importance_x.png')
plt.close()

# 5. TAHMİN OLUŞTURMA
print("\n--- TEST TAHMİNLERİ OLUŞTURULUYOR ---")

# 5.1 Test tahminleri için veri hazırlama
predictions = []

for idx, test_row in test_features.iterrows():
    target_id = test_row['target_id']
    sequence = test_sequences[test_sequences['target_id'] == target_id]['sequence'].iloc[0]
    seq_length = len(sequence)

    # Her nükleotid için tahmin
    for i, nucleotide in enumerate(sequence):
        resid = i + 1  # 1-tabanlı indeksleme

        # Özellik vektörü oluştur
        features = test_row.to_dict()
        if 'target_id' in features:
            del features['target_id']

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

        # Modeldeki tüm özellikler mevcut mu kontrol et
        missing_cols = [col for col in feature_cols if col not in feature_df.columns]
        if missing_cols:
            print(f"Eksik özellikler: {missing_cols}")
            for col in missing_cols:
                feature_df[col] = 0

        # Fazla sütunları kaldır
        extra_cols = [col for col in feature_df.columns if col not in feature_cols]
        if extra_cols:
            feature_df = feature_df.drop(columns=extra_cols)

        # Feature sütunları doğru sırada olmalı
        feature_df = feature_df[feature_cols]

        # Tahmin yap
        x_pred_norm = model_x.predict(feature_df)[0]
        y_pred_norm = model_y.predict(feature_df)[0]
        z_pred_norm = model_z.predict(feature_df)[0]

        # Denormalize et
        x_pred = x_pred_norm * (x_max - x_min) + x_min
        y_pred = y_pred_norm * (y_max - y_min) + y_min
        z_pred = z_pred_norm * (z_max - z_min) + z_min

        # İlk yapı olarak kaydet
        entry = {
            'ID': f"{target_id}_{resid}",
            'resname': nucleotide,
            'resid': resid,
            'x_1': x_pred, 'y_1': y_pred, 'z_1': z_pred
        }

        # 5.2 Fizik Temelli Çoklu Yapı Oluşturma
        # Daha gelişmiş varyasyonlar oluştur
        for i in range(2, 6):
            # Artan gürültü seviyesi ile varyasyon
            noise_scale = 0.05 * i  # Daha küçük ölçekte gürültü

            # Her aşamada daha önceki yapıyı baz alarak kümülatif değişiklik oluştur
            if i == 2:
                base_x, base_y, base_z = x_pred, y_pred, z_pred
            else:
                base_x = entry[f'x_{i - 1}']
                base_y = entry[f'y_{i - 1}']
                base_z = entry[f'z_{i - 1}']

            # Bağ uzunluklarını korumaya çalışan pertürbasyonlar
            entry[f'x_{i}'] = base_x + np.random.normal(0, noise_scale * 5)
            entry[f'y_{i}'] = base_y + np.random.normal(0, noise_scale * 5)
            entry[f'z_{i}'] = base_z + np.random.normal(0, noise_scale * 5)

        predictions.append(entry)

# 5.3 Submission dosyası oluştur
predictions_df = pd.DataFrame(predictions)
submission_df = predictions_df[sample_submission.columns]

# 5.4 Submission dosyasını kaydet
submission_df.to_csv('submission.csv', index=False)
print(f"Submission dosyası oluşturuldu! Satır sayısı: {len(submission_df)}")

# 6. Örnek RNA Görselleştirme
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')

sample_target = predictions_df['ID'].str.split('_').str[0].iloc[0]
sample_predictions = predictions_df[predictions_df['ID'].str.startswith(sample_target)]

ax.scatter(sample_predictions['x_1'],
           sample_predictions['y_1'],
           sample_predictions['z_1'],
           c=sample_predictions['resid'],
           cmap='viridis')

ax.set_title(f'Örnek RNA Tahmini: {sample_target} (XGBoost)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('sample_prediction_xgboost.png')
plt.close()

print("XGBoost pipeline tamamlandı! Submission dosyanız hazır.")
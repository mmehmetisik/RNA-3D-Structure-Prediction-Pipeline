# RNA 3D Folding - RibonanzaNet + Extreme Varyasyon Ensemble Stratejisi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
import torch
import os
import sys
import random
from tqdm import tqdm

# Zamanlayıcı başlat
start_time = time.time()

print("RNA 3D Yapı Tahmini Pipeline Başlatılıyor (RibonanzaNet + Extreme Varyasyon Ensemble)...")

# 1. RİBONANZANET KURULUMU
print("\n--- RİBONANZANET KURULUMU ---")

# Gerekli kütüphaneleri yükle
!pip
install - q
biopython
!pip
install - q
einops

# RibonanzaNet dataset'lerini kullan
input_dataset = "/kaggle/input/ribonanzanet-3d-finetune"
sys.path.append(input_dataset)

# Model sınıflarını import et (düzenlenmiş)
try:
    from Network import RibonanzaNet  # Eğer Network.py dosyası mevcutsa

    # Bu import için alternatif
    model_exists = True
    print("RibonanzaNet dosyası başarıyla yüklendi!")
except ImportError as e:
    model_exists = False
    print(f"RibonanzaNet import edilemedi: {e}")
    print("RandomForest modeli ile devam edilecek.")


# RibonanzaNet model yükleme fonksiyonu
def load_ribonanzanet_model():
    if not model_exists:
        return None

    try:
        model_paths = [
            "/kaggle/input/ribonanzanet-3d-finetune/RibonanzaNet-3D.pt",
            "/kaggle/input/ribonanzanet-3d-finetune/RibonanzaNet-3D-final.pt",
            "/kaggle/input/ribonanzanet-3d-inference/RibonanzaNet-3D.pt"
        ]

        # Mevcut model dosyasını kontrol et
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path:
            print(f"RibonanzaNet model ağırlıkları bulundu: {model_path}")

            # Model sınıflarını import et
            import yaml

            class Config:
                def __init__(self, **entries):
                    self.__dict__.update(entries)
                    self.entries = entries

            def load_config_from_yaml(file_path):
                with open(file_path, 'r') as file:
                    config = yaml.safe_load(file)
                return Config(**config)

            class finetuned_RibonanzaNet(RibonanzaNet):
                def __init__(self, config, pretrained=False):
                    config.dropout = 0.1
                    super(finetuned_RibonanzaNet, self).__init__(config)
                    if pretrained:
                        self.load_state_dict(torch.load("/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt",
                                                        map_location='cpu'))
                    self.dropout = nn.Dropout(0.0)
                    self.xyz_predictor = nn.Linear(256, 3)

                def forward(self, src):
                    sequence_features, pairwise_features = self.get_embeddings(src,
                                                                               torch.ones_like(src).long().to(
                                                                                   src.device))
                    xyz = self.xyz_predictor(sequence_features)
                    return xyz

            # Mevcut config dosyasını kontrol et
            config_paths = [
                "/kaggle/input/ribonanzanet-3d-inference/configs/pairwise.yaml",
                "/kaggle/input/ribonanzanet-3d-finetune/configs/pairwise.yaml",
                "/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml"
            ]

            config_path = None
            for path in config_paths:
                if os.path.exists(path):
                    config_path = path
                    break

            if config_path:
                # Modeli yükle
                print(f"Config dosyası bulundu: {config_path}")
                import torch.nn as nn
                model = finetuned_RibonanzaNet(load_config_from_yaml(config_path), pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()

                if torch.cuda.is_available():
                    model = model.cuda()
                    print("Model GPU'ya taşındı.")
                else:
                    print("GPU bulunamadı, CPU kullanılıyor.")

                return model
            else:
                print("Config dosyası bulunamadı!")
                return None
        else:
            print("RibonanzaNet model ağırlıkları bulunamadı!")
            return None
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None


# Tokenizer ve tahmin fonksiyonları
def tokenize_sequence(sequence):
    """RNA dizisini modele girdi olarak token'lar"""
    token_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    tokens = [token_dict.get(nt, 0) for nt in sequence]
    return torch.tensor([tokens]).long()


def predict_with_ribonanzanet(model, sequences):
    """RibonanzaNet modeli ile 3D koordinat tahmini yapar"""
    if model is None:
        return None

    try:
        all_predictions = []

        for seq_idx, sequence in enumerate(tqdm(sequences, desc="RibonanzaNet Tahminleri")):
            # Diziyi tokenize et
            tokens = tokenize_sequence(sequence)

            if torch.cuda.is_available():
                tokens = tokens.cuda()

            # Tahmin yap
            with torch.no_grad():
                predicted_coords = model(tokens).squeeze(0)

            # Koordinatları numpy array'ine dönüştür
            coords_np = predicted_coords.cpu().numpy()

            # Her nükleotid için tahmini kaydet
            for i in range(len(sequence)):
                all_predictions.append({
                    'sequence_idx': seq_idx,
                    'resid': i + 1,
                    'x_ribo': coords_np[i, 0],
                    'y_ribo': coords_np[i, 1],
                    'z_ribo': coords_np[i, 2]
                })

        return pd.DataFrame(all_predictions)
    except Exception as e:
        print(f"RibonanzaNet tahmin hatası: {e}")
        return None

# 2. VERİ YÜKLEME
print("\n--- VERİ DOSYALARI YÜKLENIYOR ---")
# Kaggle dosya yolları
train_sequences_path = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
train_labels_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
test_sequences_path = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
sample_submission_path = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"

# Veri dosyalarını yükle
train_sequences = pd.read_csv(train_sequences_path)
train_labels = pd.read_csv(train_labels_path)
test_sequences = pd.read_csv(test_sequences_path)
sample_submission = pd.read_csv(sample_submission_path)

print(f"Train sequences: {train_sequences.shape}")
print(f"Train labels: {train_labels.shape}")
print(f"Test sequences: {test_sequences.shape}")


# 3. PRE-PROCESSING
print("\n--- PRE-PROCESSING BAŞLATILIYOR ---")

# 3.1. Temporal Filtreleme
test_earliest_date = pd.to_datetime(test_sequences['temporal_cutoff']).min()
print(f"Test setindeki en erken tarih: {test_earliest_date}")

train_sequences['temporal_cutoff'] = pd.to_datetime(train_sequences['temporal_cutoff'])
filtered_train_sequences = train_sequences[train_sequences['temporal_cutoff'] < test_earliest_date]
print(f"Filtreleme sonrası eğitim seti boyutu: {len(filtered_train_sequences)} / {len(train_sequences)}")

# 3.2. Özellik çıkarma fonksiyonu
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

# 3.3. Eğitim etiketlerini düzenle
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

# 3.4. Koordinatları normalize et
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

# 3.5. Benzersiz nükleotid tipleri
unique_resnames = normalized_train_labels['resname'].unique()
print(f"Benzersiz nükleotid tipleri: {unique_resnames}")


# 4. VERİ HAZIRLAMA
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

# 5. MODEL EĞİTİMİ - RandomForest
print("\n--- RANDOMFOREST MODEL EĞİTİMİ BAŞLATILIYOR ---")

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

# RandomForest modelleri
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

# 6. RİBONANZANET ENTEGRASYONU
print("\n--- RIBONANZANET ENTEGRASYONU ---")

# RibonanzaNet modelini yükle
print("RibonanzaNet modelini yüklüyorum...")
ribonanza_model = load_ribonanzanet_model()

# Test RNA'ları için RibonanzaNet tahminleri yap
if ribonanza_model is not None:
    print("RibonanzaNet ile test dizileri için tahmin yapılıyor...")
    test_seqs = test_sequences['sequence'].tolist()
    ribonanza_predictions = predict_with_ribonanzanet(ribonanza_model, test_seqs)

    if ribonanza_predictions is not None:
        print(f"RibonanzaNet tahminleri başarıyla yapıldı: {len(ribonanza_predictions)} nükleotid için")
else:
    print("RibonanzaNet modeli kullanılamıyor, sadece RandomForest tahminleri kullanılacak.")
    ribonanza_predictions = None

# 7. TAHMİN OLUŞTURMA VE ENSEMBLE
print("\n--- TEST TAHMİNLERİ OLUŞTURULUYOR ---")


# RandomForest ile temel yapı tahmini
def predict_base_structure(test_features, test_sequences):
    """RandomForest ile ilk yapıyı (1. yapı) tahmin eder"""
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


# RandomForest ile temel yapıları tahmin et
base_predictions = predict_base_structure(test_features, test_sequences)
print(f"Temel yapı tahminleri oluşturuldu: {len(base_predictions)} RNA için")


# ENSEMBLE + EXTREME VARYASYON STRATEJİSİ
def generate_ensemble_extreme_strategy(prediction, target_id, ribo_preds=None):
    """RandomForest ve RibonanzaNet modellerini birleştirerek çoklu yapı tahmini yapar"""
    result = []

    # Ana yapı
    structure1 = prediction['coords']
    sequence_length = len(prediction['sequence'])

    # RibonanzaNet tahminlerini hazırla
    has_ribo = (ribo_preds is not None) and len(ribo_preds) > 0
    ribo_dict = {}

    # RibonanzaNet tahminlerini sözlüğe dönüştür
    if has_ribo:
        target_idx = next((i for i, row in enumerate(test_sequences.iterrows())
                           if row[1]['target_id'] == target_id), None)

        if target_idx is not None:
            for r in ribo_preds[ribo_preds['sequence_idx'] == target_idx].itertuples():
                ribo_dict[r.resid] = (r.x_ribo, r.y_ribo, r.z_ribo)

    # Her nükleotid için işlem yap
    for coord in structure1:
        resid = coord['resid']

        # RandomForest tahminleri
        rf_x = coord['x']
        rf_y = coord['y']
        rf_z = coord['z']

        # RibonanzaNet + RandomForest ensemble (Yapı 1)
        if has_ribo and resid in ribo_dict:
            ribo_x, ribo_y, ribo_z = ribo_dict[resid]
            # RibonanzaNet %70 + RandomForest %30 ağırlıklı ensemble
            ensemble_x = 0.3 * rf_x + 0.7 * ribo_x
            ensemble_y = 0.3 * rf_y + 0.7 * ribo_y
            ensemble_z = 0.3 * rf_z + 0.7 * ribo_z
        else:
            # Sadece RandomForest tahminleri
            ensemble_x, ensemble_y, ensemble_z = rf_x, rf_y, rf_z

        # Ana yapıyı kaydet
        entry = {
            'ID': coord['ID'],
            'resname': coord['resname'],
            'resid': resid,
            'x_1': ensemble_x,
            'y_1': ensemble_y,
            'z_1': ensemble_z
        }

        result.append(entry)

    # OPTIMIZED SEED 123 + LAPLACE DAĞILIMI + EXTREME VARYASYON

    # Yapı 2: Düşük gürültü (0.25 * 10) + Laplace dağılımı
    np.random.seed(123)
    for entry in result:
        resid = entry['resid']
        rel_pos = resid / sequence_length

        # Nükleotid tipine göre faktör
        base_factor = 1.5 if entry['resname'] in ['A', 'U'] else 0.8

        # RNA uzunluğuna göre faktör
        length_factor = max(0.8, min(1.5, 40 / sequence_length))

        # Uçlarda daha fazla esneklik
        edge_factor = 1.8 if rel_pos < 0.15 or rel_pos > 0.85 else 1.0

        # Gürültü seviyesi hesapla
        noise_scale = 0.25 * 10 * base_factor * length_factor * edge_factor

        # Laplace dağılımı (ağır kuyruklu - daha geniş konformasyon taraması)
        entry['x_2'] = entry['x_1'] + np.random.laplace(0, noise_scale)
        entry['y_2'] = entry['y_1'] + np.random.laplace(0, noise_scale)
        entry['z_2'] = entry['y_1'] + np.random.laplace(0, noise_scale)

    # Yapı 3: Orta-yüksek gürültü (0.50 * 10) - Gaussian
    np.random.seed(123)
    for entry in result:
        resid = entry['resid']
        rel_pos = resid / sequence_length

        # Faktörler
        base_factor = 1.5 if entry['resname'] in ['A', 'U'] else 0.8
        length_factor = max(0.8, min(1.5, 40 / sequence_length))
        edge_factor = 1.8 if rel_pos < 0.15 or rel_pos > 0.85 else 1.0

        # Gürültü seviyesi
        noise_scale = 0.50 * 10 * base_factor * length_factor * edge_factor

        # Normal dağılım
        entry['x_3'] = entry['x_1'] + np.random.normal(0, noise_scale)
        entry['y_3'] = entry['y_1'] + np.random.normal(0, noise_scale)
        entry['z_3'] = entry['z_1'] + np.random.normal(0, noise_scale)

    # Yapı 4: Çok yüksek gürültü (0.80 * 10) - Gaussian
    np.random.seed(123)
    for entry in result:
        resid = entry['resid']
        rel_pos = resid / sequence_length

        # Faktörler
        base_factor = 1.2 if entry['resname'] in ['A', 'U'] else 0.9
        length_factor = max(0.8, min(1.5, 40 / sequence_length))
        edge_factor = 2.0 if rel_pos < 0.15 or rel_pos > 0.85 else 1.0

        # Gürültü seviyesi
        noise_scale = 0.80 * 10 * base_factor * length_factor * edge_factor

        # Normal dağılım
        entry['x_4'] = entry['x_1'] + np.random.normal(0, noise_scale)
        entry['y_4'] = entry['y_1'] + np.random.normal(0, noise_scale)
        entry['z_4'] = entry['z_1'] + np.random.normal(0, noise_scale)

    # Yapı 5: Ultra yüksek gürültü + RNA uçlarına özel odak
    np.random.seed(123)
    for entry in result:
        resid = entry['resid']
        rel_pos = resid / sequence_length

        # RNA uçlarında 3x daha fazla esneklik
        edge_factor = 3.0 if rel_pos < 0.15 or rel_pos > 0.85 else 1.0

        # Nükleotid tipine göre farklı esneklik
        base_factor = 1.2 if entry['resname'] in ['A', 'U'] else 0.9

        # Gürültü seviyesi
        noise = 0.70 * 10 * edge_factor * base_factor

        # Normal dağılım
        entry['x_5'] = entry['x_1'] + np.random.normal(0, noise)
        entry['y_5'] = entry['y_1'] + np.random.normal(0, noise)
        entry['z_5'] = entry['z_1'] + np.random.normal(0, noise)

    return result


# RibonanzaNet tahminlerini işle
ribo_preds_processed = None
if ribonanza_predictions is not None:
    # RibonanzaNet tahminlerini hazırla
    print("RibonanzaNet tahminleri işleniyor...")
    ribo_preds_processed = ribonanza_predictions

# Her RNA için Ensemble + Extreme Varyasyon stratejisi uygula
print("Ensemble + Extreme Varyasyon stratejisi uygulanıyor...")
all_predictions = []

for prediction in tqdm(base_predictions, desc="RNA Yapıları İşleniyor"):
    target_id = prediction['target_id']
    structures = generate_ensemble_extreme_strategy(prediction, target_id, ribo_preds_processed)
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
    "Yapı 1: RibonanzaNet+RF Ensemble",
    "Yapı 2: Düşük Gürültü - Laplace (0.25*10)",
    "Yapı 3: Orta-Yüksek Gürültü (0.50*10)",
    "Yapı 4: Çok Yüksek Gürültü (0.80*10)",
    "Yapı 5: Ultra Gürültü + RNA Uçları"
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
plt.savefig('ribonanzanet_ensemble_strategy.png')
plt.close()

# Stratejileri karşılaştır
if ribonanza_predictions is not None:
    print("\n--- MODELLERİN KARŞILAŞTIRMASI ---")
    print("Bu submission, RibonanzaNet ve RandomForest modellerinin birleştirildiği güçlü bir ensemble kullanmaktadır.")
    print("Ensemble ağırlıkları: RibonanzaNet %70, RandomForest %30")
    print("Ayrıca Geliştirilmiş Extreme Varyasyon stratejisi uygulanmıştır:")
    print("  - Yapı 1: RibonanzaNet + RandomForest Ensemble")
    print("  - Yapı 2: Laplace dağılımı ile geniş konformasyon taraması")
    print("  - Yapı 3-5: Geniş gürültü spektrumu (0.25-0.80 * 10)")
    print("  - RNA uçlarında 1.8-3.0x esneklik artışı")
    print("  - Nükleotid tipine göre adaptif faktörler (A-U: x1.2-1.5, G-C: x0.8-0.9)")
    print("  - RNA uzunluğuna göre adaptif faktörler")

    print("\nBu kombinasyon, önceki 0.190 skorunuzu 0.200+ seviyelerine çıkarabilir.")
else:
    print("\nRibonanzaNet entegrasyonu olmadan da Extreme Varyasyon stratejisi uygulanmıştır.")
    print("Daha yüksek skor için, RibonanzaNet entegrasyonunu tamamlamayı deneyin.")

# Toplam çalışma süresi
end_time = time.time()
print(
    f"\nRibonanzaNet Ensemble + Extreme Varyasyon Pipeline tamamlandı! Toplam süre: {(end_time - start_time):.2f} saniye")































import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Dosya yolları
data_dir = r"C:\Users\ASUS\Desktop\comp3\data"
processed_dir = r"C:\Users\ASUS\Desktop\comp3\Pre-Processing\processed_02"
output_dir = r"C:\Users\ASUS\Desktop\comp3"


def load_data():
    """Veri dosyalarını yükler"""
    print("Veri dosyaları yükleniyor...")

    # İşlenmiş veriler
    train_features = pd.read_csv(os.path.join(processed_dir, "train_features.csv"))
    test_features = pd.read_csv(os.path.join(processed_dir, "test_features.csv"))
    normalized_train_labels = pd.read_csv(os.path.join(processed_dir, "normalized_train_labels.csv"))

    # Orijinal veri setleri
    test_sequences = pd.read_csv(os.path.join(data_dir, "test_sequences.csv"))
    sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    # Normalizasyon parametrelerini yükle
    with open(os.path.join(processed_dir, "normalization_params.json"), 'r') as f:
        norm_params_str = json.load(f)

    print(f"Eğitim özellikleri: {train_features.shape}")
    print(f"Eğitim etiketleri: {normalized_train_labels.shape}")
    print(f"Test özellikleri: {test_features.shape}")

    return {
        'train_features': train_features,
        'test_features': test_features,
        'train_labels': normalized_train_labels,
        'test_sequences': test_sequences,
        'sample_submission': sample_submission,
        'norm_params': norm_params_str
    }


def prepare_training_data(train_features, train_labels):
    """Eğitim verileri hazırlar"""
    print("Eğitim verileri hazırlanıyor...")

    # NaN değer kontrolü
    nan_count_x = train_labels['x_1'].isna().sum()
    nan_count_y = train_labels['y_1'].isna().sum()
    nan_count_z = train_labels['z_1'].isna().sum()
    print(f"Etiketlerdeki NaN değer sayısı - X: {nan_count_x}, Y: {nan_count_y}, Z: {nan_count_z}")

    # NaN değerleri temizle
    train_labels_clean = train_labels.dropna(subset=['x_1', 'y_1', 'z_1'])
    print(f"NaN temizleme sonrası etiket sayısı: {len(train_labels_clean)} / {len(train_labels)}")

    # Benzersiz nükleotid tiplerini saklayalım
    unique_resnames = train_labels_clean['resname'].unique()
    print(f"Benzersiz nükleotid tipleri: {unique_resnames}")

    # Train_features ve train_labels'ı birleştir
    feature_cols = ['length', 'gc_content', 'a_content', 'u_content',
                    'gu_pairs', 'au_pairs', 'gc_pairs']

    # Her bir nükleotid için eğitim verisi oluştur
    train_data = []

    # Tüm hedefler için
    for target_id in train_features['target_id'].unique():
        # Bu hedef için özellikler
        if target_id in train_features['target_id'].values:
            target_features = train_features[train_features['target_id'] == target_id][feature_cols].iloc[0].to_dict()
        else:
            continue  # Hedef bulunamazsa atla

        # Bu hedef için etiketler
        target_labels = train_labels_clean[train_labels_clean['target_id'] == target_id]
        if len(target_labels) == 0:
            continue  # Etiket yoksa atla

        # Her bir nükleotid için bir satır oluştur
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

            # Pozisyon bilgisini ekle (nükleotidin dizideki pozisyonu önemli bir özellik)
            entry['position_ratio'] = row['resid'] / target_features['length']

            train_data.append(entry)

    train_df = pd.DataFrame(train_data)
    print(f"Hazırlanan eğitim verisi: {train_df.shape}")

    return train_df, unique_resnames


def train_baseline_models(train_data):
    """Baseline koordinat tahmin modellerini eğitir"""
    print("Baseline modeller eğitiliyor...")

    # NaN değerleri son bir kez kontrol et
    x_nan_count = train_data['x_1'].isna().sum()
    y_nan_count = train_data['y_1'].isna().sum()
    z_nan_count = train_data['z_1'].isna().sum()

    if x_nan_count > 0 or y_nan_count > 0 or z_nan_count > 0:
        print(f"UYARI: Hala NaN değerler var - X: {x_nan_count}, Y: {y_nan_count}, Z: {z_nan_count}")
        print("NaN değerler temizleniyor...")
        train_data = train_data.dropna(subset=['x_1', 'y_1', 'z_1'])

    # Özellik sütunları
    feature_cols = ['length', 'gc_content', 'a_content', 'u_content',
                    'gu_pairs', 'au_pairs', 'gc_pairs', 'position_ratio', 'resid']

    # One-hot encoding için nükleotid tipini ekle
    train_data_encoded = pd.get_dummies(train_data, columns=['resname'], prefix='resname')
    feature_cols += [col for col in train_data_encoded.columns if col.startswith('resname_')]

    # Eğitim ve doğrulama setleri ayır
    X = train_data_encoded[feature_cols]
    y_x = train_data_encoded['x_1']
    y_y = train_data_encoded['y_1']
    y_z = train_data_encoded['z_1']

    X_train, X_val, y_x_train, y_x_val = train_test_split(X, y_x, test_size=0.2, random_state=42)
    _, _, y_y_train, y_y_val = train_test_split(X, y_y, test_size=0.2, random_state=42)
    _, _, y_z_train, y_z_val = train_test_split(X, y_z, test_size=0.2, random_state=42)

    # Modelleri oluştur
    print("X koordinatı için model eğitiliyor...")
    model_x = RandomForestRegressor(n_estimators=100, random_state=42)
    model_x.fit(X_train, y_x_train)

    print("Y koordinatı için model eğitiliyor...")
    model_y = RandomForestRegressor(n_estimators=100, random_state=42)
    model_y.fit(X_train, y_y_train)

    print("Z koordinatı için model eğitiliyor...")
    model_z = RandomForestRegressor(n_estimators=100, random_state=42)
    model_z.fit(X_train, y_z_train)

    # Modelleri değerlendir
    x_pred = model_x.predict(X_val)
    y_pred = model_y.predict(X_val)
    z_pred = model_z.predict(X_val)

    x_rmse = np.sqrt(mean_squared_error(y_x_val, x_pred))
    y_rmse = np.sqrt(mean_squared_error(y_y_val, y_pred))
    z_rmse = np.sqrt(mean_squared_error(y_z_val, z_pred))

    print(f"Doğrulama RMSE - X: {x_rmse:.4f}, Y: {y_rmse:.4f}, Z: {z_rmse:.4f}")

    return {
        'model_x': model_x,
        'model_y': model_y,
        'model_z': model_z,
        'feature_cols': feature_cols
    }


def generate_predictions(models, test_features, test_sequences, norm_params, unique_resnames):
    """Test verileri için tahminler oluşturur"""
    print("Test tahminleri oluşturuluyor...")

    # Normalizasyon parametrelerini parse et
    norm_params_dict = eval(norm_params['train_normalization_params'])
    x_min, x_max = norm_params_dict['x_1']
    y_min, y_max = norm_params_dict['y_1']
    z_min, z_max = norm_params_dict['z_1']

    # Test veri seti hazırlama
    predictions = []

    # Her bir test dizisi için
    for idx, test_row in test_features.iterrows():
        target_id = test_row['target_id']
        sequence = test_sequences[test_sequences['target_id'] == target_id]['sequence'].iloc[0]

        # Dizi uzunluğu
        seq_length = len(sequence)

        # Her bir nükleotid için tahmin yap
        for i, nucleotide in enumerate(sequence):
            resid = i + 1  # 1-tabanlı indeksleme

            # Özellik vektörü oluştur
            features = test_row[['length', 'gc_content', 'a_content', 'u_content',
                                 'gu_pairs', 'au_pairs', 'gc_pairs']].to_dict()
            features['position_ratio'] = resid / seq_length
            features['resid'] = resid

            # Tüm nükleotid tipleri için one-hot encoding (eğitim setindeki tüm tipler dahil)
            for resname in unique_resnames:
                features[f'resname_{resname}'] = 1 if nucleotide == resname else 0

            # Model özellik sütunlarını kontrol et ve eksik olanları 0 olarak doldur
            feature_df = pd.DataFrame([features])
            for col in models['feature_cols']:
                if col not in feature_df.columns:
                    feature_df[col] = 0

            # Tahmin yap
            feature_vector = feature_df[models['feature_cols']]

            x_pred_norm = models['model_x'].predict(feature_vector)[0]
            y_pred_norm = models['model_y'].predict(feature_vector)[0]
            z_pred_norm = models['model_z'].predict(feature_vector)[0]

            # Normalize edilmiş tahminleri gerçek koordinatlara dönüştür
            x_pred = x_pred_norm * (x_max - x_min) + x_min
            y_pred = y_pred_norm * (y_max - y_min) + y_min
            z_pred = z_pred_norm * (z_max - z_min) + z_min

            predictions.append({
                'ID': f"{target_id}_{resid}",
                'resname': nucleotide,
                'resid': resid,
                'x_1': x_pred,
                'y_1': y_pred,
                'z_1': z_pred
            })

    return pd.DataFrame(predictions)


def generate_multiple_structures(base_predictions, num_structures=5):
    """Tek yapı tahmininden çoklu yapılar oluşturur"""
    print(f"{num_structures} farklı yapı oluşturuluyor...")

    # Tüm tahminler
    all_predictions = []

    # İlk yapı olarak temel tahmini kullan
    for _, row in base_predictions.iterrows():
        entry = {
            'ID': row['ID'],
            'resname': row['resname'],
            'resid': row['resid'],
            'x_1': row['x_1'],
            'y_1': row['y_1'],
            'z_1': row['z_1']
        }

        # Diğer 4 yapıyı oluştur (farklı varyasyonlar)
        for i in range(2, num_structures + 1):
            # Rastgele varyasyon ekle (kontrollü)
            noise_scale = 0.1 * (i - 1)  # Her yapı için artan oranda varyasyon

            entry[f'x_{i}'] = row['x_1'] + np.random.normal(0, noise_scale * 10)
            entry[f'y_{i}'] = row['y_1'] + np.random.normal(0, noise_scale * 10)
            entry[f'z_{i}'] = row['z_1'] + np.random.normal(0, noise_scale * 10)

        all_predictions.append(entry)

    return pd.DataFrame(all_predictions)


def save_submission(predictions, sample_submission):
    """Tahminleri submission formatında kaydeder"""
    print("Submission dosyası oluşturuluyor...")

    # Sütun sırasını sample_submission ile aynı yap
    submission = predictions[sample_submission.columns]

    # Submission dosyasını kaydet
    submission_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Submission dosyası kaydedildi: {submission_path}")
    return submission_path


def run_pipeline():
    """Tüm model pipeline'ını çalıştırır"""
    print("RNA 3D Yapı Tahmini Pipeline Başlatılıyor...")

    # 1. Verileri yükle
    data = load_data()

    # 2. Eğitim verilerini hazırla
    train_data, unique_resnames = prepare_training_data(data['train_features'], data['train_labels'])

    # 3. Baseline modelleri eğit
    models = train_baseline_models(train_data)

    # 4. Test tahminlerini oluştur
    base_predictions = generate_predictions(
        models,
        data['test_features'],
        data['test_sequences'],
        data['norm_params'],
        unique_resnames
    )

    # 5. Çoklu yapılar oluştur
    final_predictions = generate_multiple_structures(base_predictions)

    # 6. Submission dosyasını kaydet
    submission_path = save_submission(final_predictions, data['sample_submission'])

    print("Pipeline tamamlandı!")
    return submission_path


if __name__ == "__main__":
    # Pipeline'ı çalıştır
    submission_path = run_pipeline()
    print(f"Final submission dosyası: {submission_path}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# Zamanlayıcı başlat
start_time = time.time()

# Dosya yolları
train_sequences_path = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
train_labels_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
test_sequences_path = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
sample_submission_path = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"

print("RNA 3D Yapı Tahmini Pipeline Başlatılıyor (Temel Termal Ensemble Kabsch)...")

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

# 4. MODEL EĞİTİMİ
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

X_train, X_val, y_x_train, y_x_val = train_test_split(X, y_x, test_size=0.2, random_state=123)
_, _, y_y_train, y_y_val = train_test_split(X, y_y, test_size=0.2, random_state=123)
_, _, y_z_train, y_z_val = train_test_split(X, y_z, test_size=0.2, random_state=123)

# RANDOM FOREST MODELLERİ
print("X koordinatı için RF model eğitiliyor...")
model_x = RandomForestRegressor(n_estimators=150, random_state=123)
model_x.fit(X_train, y_x_train)

print("Y koordinatı için RF model eğitiliyor...")
model_y = RandomForestRegressor(n_estimators=150, random_state=123)
model_y.fit(X_train, y_y_train)

print("Z koordinatı için RF model eğitiliyor...")
model_z = RandomForestRegressor(n_estimators=150, random_state=123)
model_z.fit(X_train, y_z_train)

# Doğrulama değerlendirmesi
x_pred_val = model_x.predict(X_val)
y_pred_val = model_y.predict(X_val)
z_pred_val = model_z.predict(X_val)

x_rmse = np.sqrt(mean_squared_error(y_x_val, x_pred_val))
y_rmse = np.sqrt(mean_squared_error(y_y_val, y_pred_val))
z_rmse = np.sqrt(mean_squared_error(y_z_val, z_pred_val))

print(f"Doğrulama RMSE - X: {x_rmse:.4f}, Y: {y_rmse:.4f}, Z: {z_rmse:.4f}")


# 5. YARDIMCI FONKSİYONLAR

# Kabsch algoritması ile 3D rotasyon matrisi hesaplama
def kabsch_rotation(P, Q):
    """Kabsch algoritması ile rotasyon matrisi hesaplar"""
    # Centroid hesapla
    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)

    # Merkezleri çıkar
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    # Korelasyon matrisi
    H = P_centered.T @ Q_centered

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Rotasyon matrisi
    R = Vt.T @ U.T

    # Yansıma durumunu kontrol et
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R, P_mean, Q_mean


# Basit Watson-Crick baz eşleşmesi tahmini
def predict_secondary_structure(sequence):
    """RNA dizisinden basit bir sekonder yapı tahmini yapar"""
    # Boş parantez notasyonu oluştur
    sec_structure = ['.' for _ in range(len(sequence))]

    # Watson-Crick ve G-U wobble eşleşmelerini kontrol et
    i = 0
    while i < len(sequence):
        for j in range(len(sequence) - 1, i + 3, -1):  # En az 4 nükleotid uzaklıkta arama yap
            # Watson-Crick eşleşmeleri
            if (sequence[i] == 'A' and sequence[j] == 'U') or \
                    (sequence[i] == 'U' and sequence[j] == 'A') or \
                    (sequence[i] == 'G' and sequence[j] == 'C') or \
                    (sequence[i] == 'C' and sequence[j] == 'G') or \
                    (sequence[i] == 'G' and sequence[j] == 'U') or \
                    (sequence[i] == 'U' and sequence[j] == 'G'):  # G-U wobble

                # Aralarında çakışan baz çiftleri var mı kontrol et
                if all(sec_structure[k] == '.' for k in range(i + 1, j)):
                    sec_structure[i] = '('
                    sec_structure[j] = ')'
                    break
        i += 1

    return ''.join(sec_structure)


# RNA motiflerini tahmin et
def identify_motifs(sequence, sec_structure):
    """Sekonder yapıdan RNA motiflerini tanımlar"""
    motifs = ['.' for _ in range(len(sequence))]

    # Stem, Loop, Bulge, Hairpin, Terminal bölgeleri tanımla
    for i in range(len(sequence)):
        if sec_structure[i] == '(':
            # Stem başlangıcı
            if i == 0 or sec_structure[i - 1] != '(':
                motifs[i] = 'stem_edge'
            else:
                motifs[i] = 'stem'
        elif sec_structure[i] == ')':
            # Stem sonu
            if i == len(sequence) - 1 or sec_structure[i + 1] != ')':
                motifs[i] = 'stem_edge'
            else:
                motifs[i] = 'stem'
        else:  # '.'
            # Loop bölgesi
            if i > 0 and i < len(sequence) - 1:
                if sec_structure[i - 1] == '(' and sec_structure[i + 1] == ')':
                    motifs[i] = 'hairpin'  # Hairpin loop
                elif sec_structure[i - 1] == '(' or sec_structure[i + 1] == ')':
                    motifs[i] = 'bulge'  # Bulge veya internal loop
                else:
                    motifs[i] = 'loop'  # Diğer loop'lar
            else:
                motifs[i] = 'terminal'  # Uç bölgeler

    return motifs


# 6. TAHMİN OLUŞTURMA FONKSİYONLARI

# Test veri seti hazırlama - Temel yapı
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
                'z': z_pred,
            })

        # Sekonder yapı ve motif tahmini
        sec_structure = predict_secondary_structure(sequence)
        motifs = identify_motifs(sequence, sec_structure)

        # Motif ve sekonder yapı bilgilerini koordinatlara ekle
        for i, coord in enumerate(coords):
            coord['sec_structure'] = sec_structure[i]
            coord['motif'] = motifs[i]

        # Tüm dizi için tahminleri kaydet
        predictions.append({
            'target_id': target_id,
            'sequence': sequence,
            'coords': coords
        })

    return predictions


# TERMAL ENSEMBLE KABSCH STRATEJİSİ - ANA MODÜL
def generate_thermal_ensemble_kabsch(prediction):
    """Termal Ensemble konsepti ile geliştirilmiş Kabsch stratejisi"""
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

    # Koordinat matrisi oluştur
    coords = np.array([[entry['x_1'], entry['y_1'], entry['z_1']] for entry in result])

    # Sequence uzunluğu
    sequence_length = len(prediction['sequence'])
    sequence = prediction['sequence']

    # Başarılı seed değerini koru
    seed_base = 123

    # RNA molekülündeki her nükleotid için ortak faktörler
    np.random.seed(seed_base)
    residue_factors = {}

    for entry in result:
        resid = entry['resid']
        residue_factors[resid] = {
            'common_x': np.random.normal(0, 1.0),
            'common_y': np.random.normal(0, 1.0),
            'common_z': np.random.normal(0, 1.0),
            'corr_factor_2': np.random.normal(0, 1.0),
            'corr_factor_3': np.random.normal(0, 1.0),
            'corr_factor_4': np.random.normal(0, 1.0),
            'corr_factor_5': np.random.normal(0, 1.0)
        }

    # Sekonder yapı ve motif bilgilerini alın
    motifs = [coord['motif'] for coord in structure1]

    # --------------------------------------------------
    # Yapı 2: DÜŞÜK SICAKLIK (0.25 * 10) - Laplace dağılımı
    # --------------------------------------------------
    np.random.seed(seed_base)
    noise_scale_2 = 0.25 * 10

    # Tüm yapı için küçük rastgele rotasyon
    angle_degrees = np.random.uniform(-15, 15)
    axis = np.random.rand(3)
    axis = axis / np.linalg.norm(axis)  # Birim vektör

    rotation = Rotation.from_rotvec(np.radians(angle_degrees) * axis)
    rot_matrix = rotation.as_matrix()

    # Yapının merkezini bul
    centroid = np.mean(coords, axis=0)

    # Merkeze göre koordinatları kaydır, rotasyonu uygula ve geri kaydır
    coords_centered = coords - centroid
    rotated_coords = (rot_matrix @ coords_centered.T).T + centroid

    # Nükleotid başına korelasyonlu gürültü ekle
    for i, entry in enumerate(result):
        resid = entry['resid']
        rel_pos = resid / sequence_length
        nucleotide = entry['resname']

        # Student's t dağılımı parametreleri
        df = 5  # Varsayılan serbestlik derecesi

        # Adaptif faktörler - İYİLEŞTİRİLMİŞ NÜKLEOTİD FAKTÖRÜ
        if nucleotide == 'A':
            base_factor = 1.6  # A en esnek
        elif nucleotide == 'U':
            base_factor = 1.4  # U oldukça esnek
        elif nucleotide == 'G':
            base_factor = 0.75  # G oldukça stabil
        else:  # C
            base_factor = 0.85  # C stabil

        length_factor = max(0.8, min(1.5, 40 / sequence_length))

        # Kenar faktörü
        if rel_pos < 0.15 or rel_pos > 0.85:
            edge_factor = 1.8  # Uçlarda daha esnek
        else:
            edge_factor = 1.0  # Orta bölgelerde daha stabil

        # Motif bilgisine göre edge faktörünü ayarla
        motif = motifs[i]
        if motif == 'stem':
            edge_factor *= 0.7  # Stem bölgeleri daha az hareket eder
        elif motif == 'loop':
            edge_factor *= 1.2  # Loop bölgeleri daha fazla hareket eder
        elif motif == 'bulge':
            edge_factor *= 1.5  # Bulge bölgeleri çok daha fazla hareket eder
        elif motif == 'hairpin':
            edge_factor *= 1.3  # Hairpin bölgeleri daha fazla hareket eder
        elif motif == 'terminal':
            edge_factor *= 1.8  # Terminal bölgeler en fazla hareket eder
        elif motif == 'stem_edge':
            edge_factor *= 0.85  # Stem kenarları orta seviye hareket eder

        # Gürültü seviyesi
        noise = noise_scale_2 * base_factor * length_factor * edge_factor

        # Korelasyonlu gürültü faktörleri
        corr_factor = residue_factors[resid]['corr_factor_2']
        corr_weight = 0.72  # İnce ayarlanmış korelasyon ağırlığı
        indiv_weight = 0.6

        # Laplace dağılımı ile gürültü
        entry['x_2'] = rotated_coords[i, 0] + corr_factor * noise * corr_weight + np.random.laplace(0,
                                                                                                    noise * indiv_weight)
        entry['y_2'] = rotated_coords[i, 1] + corr_factor * noise * corr_weight + np.random.laplace(0,
                                                                                                    noise * indiv_weight)
        entry['z_2'] = rotated_coords[i, 2] + corr_factor * noise * corr_weight + np.random.laplace(0,
                                                                                                    noise * indiv_weight)

    # --------------------------------------------------
    # Yapı 3: ORTA SICAKLIK (0.45 * 10) - Student's t + Normal karışımı dağılımı
    # --------------------------------------------------
    np.random.seed(seed_base + 1)
    noise_scale_3 = 0.45 * 10  # Orta derecede gürültü

    # Uzama faktörleri - her eksende farklı
    stretch_x = np.random.uniform(0.9, 1.1)
    stretch_y = np.random.uniform(0.9, 1.1)
    stretch_z = np.random.uniform(0.9, 1.1)

    # Centroid hesapla
    centroid_3 = np.mean(coords, axis=0)

    # Merkezinden kaydır, uzat/sıkıştır ve geri kaydır
    coords_3_centered = coords - centroid_3
    coords_3_stretched = np.copy(coords_3_centered)
    coords_3_stretched[:, 0] *= stretch_x
    coords_3_stretched[:, 1] *= stretch_y
    coords_3_stretched[:, 2] *= stretch_z
    coords_3_stretched += centroid_3

    # Nükleotid başına korelasyonlu gürültü ekle
    for i, entry in enumerate(result):
        resid = entry['resid']
        rel_pos = resid / sequence_length
        nucleotide = entry['resname']

        # Student's t dağılımı parametreleri
        df = 5  # Varsayılan serbestlik derecesi
        motif = motifs[i]
        if motif == 'stem':
            df = 6  # Daha az aşırı hareket
        elif motif == 'loop' or motif == 'bulge':
            df = 4  # Daha aşırı hareket
        elif motif == 'hairpin':
            df = 3  # En aşırı hareket
        elif motif == 'terminal':
            df = 3  # En aşırı hareket

        # Adaptif faktörler - İYİLEŞTİRİLMİŞ NÜKLEOTİD FAKTÖRÜ
        if nucleotide == 'A':
            base_factor = 1.6
        elif nucleotide == 'U':
            base_factor = 1.4
        elif nucleotide == 'G':
            base_factor = 0.75
        else:  # C
            base_factor = 0.85

        length_factor = max(0.8, min(1.5, 40 / sequence_length))

        # Kenar faktörü - daha agresif
        if rel_pos < 0.15 or rel_pos > 0.85:
            edge_factor = 2.0  # Uçlarda çok daha esnek
        else:
            edge_factor = 1.0

        # Motif bilgisine göre edge faktörünü ayarla
        if motif == 'stem':
            edge_factor *= 0.7  # Stem bölgeleri daha az hareket eder
        elif motif == 'loop':
            edge_factor *= 1.2  # Loop bölgeleri daha fazla hareket eder
        elif motif == 'bulge':
            edge_factor *= 1.5  # Bulge bölgeleri çok daha fazla hareket eder
        elif motif == 'hairpin':
            edge_factor *= 1.3  # Hairpin bölgeleri daha fazla hareket eder
        elif motif == 'terminal':
            edge_factor *= 1.8  # Terminal bölgeler en fazla hareket eder
        elif motif == 'stem_edge':
            edge_factor *= 0.85  # Stem kenarları orta seviye hareket eder

        # Gürültü seviyesi
        noise = noise_scale_3 * base_factor * length_factor * edge_factor

        # Korelasyonlu gürültü faktörleri
        corr_factor = residue_factors[resid]['corr_factor_3']
        corr_weight = 0.74  # İnce ayarlanmış korelasyon ağırlığı
        indiv_weight = 0.6

        # Student's t ve Normal karışımı dağılımı ile gürültü
        if np.random.rand() < 0.7:  # %70 Student's t
            noise_x = np.random.standard_t(df) * (noise * indiv_weight / np.sqrt(df / (df - 2)))
            noise_y = np.random.standard_t(df) * (noise * indiv_weight / np.sqrt(df / (df - 2)))
            noise_z = np.random.standard_t(df) * (noise * indiv_weight / np.sqrt(df / (df - 2)))
        else:  # %30 Normal
            noise_x = np.random.normal(0, noise * indiv_weight)
            noise_y = np.random.normal(0, noise * indiv_weight)
            noise_z = np.random.normal(0, noise * indiv_weight)

        # Yapı 2'den hafif etkilenme (%15)
        influence_factor = 0.15
        prev_dx = entry['x_2'] - entry['x_1']
        prev_dy = entry['y_2'] - entry['y_1']
        prev_dz = entry['z_2'] - entry['z_1']

        entry['x_3'] = coords_3_stretched[
                           i, 0] + corr_factor * noise * corr_weight + noise_x + prev_dx * influence_factor
        entry['y_3'] = coords_3_stretched[
                           i, 1] + corr_factor * noise * corr_weight + noise_y + prev_dy * influence_factor
        entry['z_3'] = coords_3_stretched[
                           i, 2] + corr_factor * noise * corr_weight + noise_z + prev_dz * influence_factor

    # --------------------------------------------------
    # Yapı 4: YÜKSEK SICAKLIK (0.75 * 10) - Student's t dağılımı
    # --------------------------------------------------
    np.random.seed(seed_base + 2)
    noise_scale_4 = 0.75 * 10  # Yüksek derecede gürültü

    # Birleşik transformasyon: Rotasyon + Makaslama
    # Önce rotasyon
    angle_degrees_4 = np.random.uniform(-20, 20)
    axis_4 = np.random.rand(3)
    axis_4 = axis_4 / np.linalg.norm(axis_4)

    rotation_4 = Rotation.from_rotvec(np.radians(angle_degrees_4) * axis_4)
    rot_matrix_4 = rotation_4.as_matrix()

    # Makaslama (shear) matrisi
    shear_matrix = np.eye(3)
    shear_matrix[0, 1] = np.random.uniform(-0.2, 0.2)  # x yönünde y'ye bağlı makaslama

    # Centroid hesapla
    centroid_4 = np.mean(coords, axis=0)

    # Dönüşümleri uygula: merkeze kaydır, rotasyon, makaslama, geri kaydır
    coords_4_centered = coords - centroid_4
    coords_4_rotated = (rot_matrix_4 @ coords_4_centered.T).T
    coords_4_transformed = (shear_matrix @ coords_4_rotated.T).T + centroid_4

    # Nükleotid başına korelasyonlu gürültü ekle
    for i, entry in enumerate(result):
        resid = entry['resid']
        rel_pos = resid / sequence_length
        nucleotide = entry['resname']

        # Student's t dağılımı parametreleri
        df = 4  # Daha düşük serbestlik derecesi (daha ağır kuyruk)
        motif = motifs[i]
        if motif == 'stem':
            df = 5  # Daha az aşırı hareket
        elif motif == 'loop' or motif == 'bulge':
            df = 3  # Daha aşırı hareket
        elif motif == 'hairpin':
            df = 2  # En aşırı hareket
        elif motif == 'terminal':
            df = 2  # En aşırı hareket

        # Adaptif faktörler - İYİLEŞTİRİLMİŞ NÜKLEOTİD FAKTÖRÜ
        if nucleotide == 'A':
            base_factor = 1.7  # Yüksek sıcaklıkta daha esnek
        elif nucleotide == 'U':
            base_factor = 1.5  # Yüksek sıcaklıkta daha esnek
        elif nucleotide == 'G':
            base_factor = 0.8  # Nispeten daha stabil
        else:  # C
            base_factor = 0.9  # Nispeten daha stabil

        length_factor = max(0.7, min(1.6, 40 / sequence_length))

        # Kenar faktörü - daha agresif
        if rel_pos < 0.1 or rel_pos > 0.9:
            edge_factor = 2.5  # Uçlarda çok çok daha esnek
        elif rel_pos < 0.2 or rel_pos > 0.8:
            edge_factor = 1.8  # Uçlara yakın daha esnek
        else:
            edge_factor = 1.0

        # Motif bilgisine göre edge faktörünü ayarla
        if motif == 'stem':
            edge_factor *= 0.7  # Stem bölgeleri daha az hareket eder
        elif motif == 'loop':
            edge_factor *= 1.2  # Loop bölgeleri daha fazla hareket eder
        elif motif == 'bulge':
            edge_factor *= 1.5  # Bulge bölgeleri çok daha fazla hareket eder
        elif motif == 'hairpin':
            edge_factor *= 1.3  # Hairpin bölgeleri daha fazla hareket eder
        elif motif == 'terminal':
            edge_factor *= 1.8  # Terminal bölgeler en fazla hareket eder
        elif motif == 'stem_edge':
            edge_factor *= 0.85  # Stem kenarları orta seviye hareket eder

        # Gürültü seviyesi
        noise = noise_scale_4 * base_factor * length_factor * edge_factor

        # Korelasyonlu gürültü faktörleri
        corr_factor = residue_factors[resid]['corr_factor_4']
        corr_weight = 0.77  # İnce ayarlanmış korelasyon ağırlığı
        indiv_weight = 0.55

        # Student's t dağılımı ile gürültü
        if df <= 2:
            # df <= 2 için normal dağılım kullan
            noise_x = np.random.normal(0, noise * indiv_weight)
            noise_y = np.random.normal(0, noise * indiv_weight)
            noise_z = np.random.normal(0, noise * indiv_weight)
        else:
            # df > 2 için Student's t dağılımı kullan
            noise_x = np.random.standard_t(df) * (noise * indiv_weight / np.sqrt(df / (df - 2)))
            noise_y = np.random.standard_t(df) * (noise * indiv_weight / np.sqrt(df / (df - 2)))
            noise_z = np.random.standard_t(df) * (noise * indiv_weight / np.sqrt(df / (df - 2)))

        # Yapı 2 ve 3'ten etkilenme
        influence_2 = 0.10  # Yapı 2'den etki
        influence_3 = 0.20  # Yapı 3'ten etki

        dx2 = entry['x_2'] - entry['x_1']
        dy2 = entry['y_2'] - entry['y_1']
        dz2 = entry['z_2'] - entry['z_1']

        dx3 = entry['x_3'] - entry['x_1']
        dy3 = entry['y_3'] - entry['y_1']
        dz3 = entry['z_3'] - entry['z_1']

        entry['x_4'] = coords_4_transformed[
                           i, 0] + corr_factor * noise * corr_weight + noise_x + dx2 * influence_2 + dx3 * influence_3
        entry['y_4'] = coords_4_transformed[
                           i, 1] + corr_factor * noise * corr_weight + noise_y + dy2 * influence_2 + dy3 * influence_3
        entry['z_4'] = coords_4_transformed[
                           i, 2] + corr_factor * noise * corr_weight + noise_z + dz2 * influence_2 + dz3 * influence_3

    # --------------------------------------------------
    # Yapı 5: KABSCH HİZALAMA - Yapı 1, 2 ve 3'ün en iyi kombinasyonu
    # --------------------------------------------------
    np.random.seed(seed_base + 3)
    noise_scale_5 = 0.65 * 10  # Orta-yüksek gürültü

    # Yapı 5 için temel olarak yapı 1, 2 ve 3'ün ortalama konformasyon alanını al
    coords_1 = np.array([[entry['x_1'], entry['y_1'], entry['z_1']] for entry in result])
    coords_2 = np.array([[entry['x_2'], entry['y_2'], entry['z_2']] for entry in result])
    coords_3 = np.array([[entry['x_3'], entry['y_3'], entry['z_3']] for entry in result])

    # Konformasyonları ağırlıklı olarak birleştir
    # Yapı 1 temel yapı, 2 ve 3 ise 3 farklı "sıcaklık" seviyesinde varyasyonlar
    ref_coords = coords_1 * 0.5 + coords_2 * 0.25 + coords_3 * 0.25

    # Kabsch rotasyonu uygula
    valid_indices = ~np.isnan(coords_1).any(axis=1) & ~np.isnan(ref_coords).any(axis=1)
    if np.sum(valid_indices) >= 3:  # En az 3 nokta gerekli
        rot_matrix, centroid_1, centroid_ref = kabsch_rotation(coords_1[valid_indices], ref_coords[valid_indices])
        aligned_coords = (rot_matrix @ (coords_1 - centroid_1).T).T + centroid_ref
    else:
        # Hizalama yapılamıyorsa orijinal yapıyı kullan
        aligned_coords = coords_1

    # Nükleotid başına korelasyonlu gürültü ekle - Karma gürültü (Laplace ve Normal)
    for i, entry in enumerate(result):
        resid = entry['resid']
        rel_pos = resid / sequence_length
        nucleotide = entry['resname']

        # Student's t dağılımı parametreleri
        df = 4  # Varsayılan serbestlik derecesi
        motif = motifs[i]
        if motif == 'stem':
            df = 5  # Daha az aşırı hareket
        elif motif == 'loop' or motif == 'bulge':
            df = 4  # Daha aşırı hareket
        elif motif == 'hairpin':
            df = 3  # En aşırı hareket
        elif motif == 'terminal':
            df = 3  # En aşırı hareket

        # Adaptif faktörler - İYİLEŞTİRİLMİŞ NÜKLEOTİD FAKTÖRÜ
        if nucleotide == 'A':
            base_factor = 1.6
        elif nucleotide == 'U':
            base_factor = 1.4
        elif nucleotide == 'G':
            base_factor = 0.75
        else:  # C
            base_factor = 0.85

        length_factor = max(0.7, min(1.8, 35 / sequence_length))

        # Kenar faktörü - daha agresif
        if rel_pos < 0.1 or rel_pos > 0.9:
            edge_factor = 3.0  # Uçlarda çok çok çok daha esnek
        elif rel_pos < 0.2 or rel_pos > 0.8:
            edge_factor = 2.0  # Uçlara yakın çok daha esnek
        else:
            edge_factor = 1.0

        # Motif bilgisine göre edge faktörünü ayarla
        if motif == 'stem':
            edge_factor *= 0.7  # Stem bölgeleri daha az hareket eder
        elif motif == 'loop':
            edge_factor *= 1.2  # Loop bölgeleri daha fazla hareket eder
        elif motif == 'bulge':
            edge_factor *= 1.5  # Bulge bölgeleri çok daha fazla hareket eder
        elif motif == 'hairpin':
            edge_factor *= 1.3  # Hairpin bölgeleri daha fazla hareket eder
        elif motif == 'terminal':
            edge_factor *= 1.8  # Terminal bölgeler en fazla hareket eder
        elif motif == 'stem_edge':
            edge_factor *= 0.85  # Stem kenarları orta seviye hareket eder

        # Gürültü seviyesi
        noise = noise_scale_5 * edge_factor * base_factor * length_factor

        # Korelasyonlu gürültü faktörleri
        corr_factor = residue_factors[resid]['corr_factor_5']
        corr_weight = 0.82  # İnce ayarlanmış korelasyon ağırlığı
        indiv_weight = 0.5

        # Karma gürültü: Student's t ve Laplace karışımı
        if np.random.rand() < 0.7:  # %70 Student's t
            noise_x = np.random.standard_t(df) * (noise * indiv_weight / np.sqrt(df / (df - 2)))
            noise_y = np.random.standard_t(df) * (noise * indiv_weight / np.sqrt(df / (df - 2)))
            noise_z = np.random.standard_t(df) * (noise * indiv_weight / np.sqrt(df / (df - 2)))
        else:  # %30 Laplace
            noise_x = np.random.laplace(0, noise * indiv_weight * 0.7)
            noise_y = np.random.laplace(0, noise * indiv_weight * 0.7)
            noise_z = np.random.laplace(0, noise * indiv_weight * 0.7)

        # Önceki yapılardan etkilenme
        influence_2 = 0.05  # Yapı 2'den etki
        influence_3 = 0.10  # Yapı 3'ten etki
        influence_4 = 0.15  # Yapı 4'ten etki

        dx2 = entry['x_2'] - entry['x_1']
        dy2 = entry['y_2'] - entry['y_1']
        dz2 = entry['z_2'] - entry['z_1']

        dx3 = entry['x_3'] - entry['x_1']
        dy3 = entry['y_3'] - entry['y_1']
        dz3 = entry['z_3'] - entry['z_1']

        dx4 = entry['x_4'] - entry['x_1']
        dy4 = entry['y_4'] - entry['y_1']
        dz4 = entry['z_4'] - entry['z_1']

        entry['x_5'] = aligned_coords[i, 0] + corr_factor * noise * corr_weight + noise_x + \
                       dx2 * influence_2 + dx3 * influence_3 + dx4 * influence_4
        entry['y_5'] = aligned_coords[i, 1] + corr_factor * noise * corr_weight + noise_y + \
                       dy2 * influence_2 + dy3 * influence_3 + dy4 * influence_4
        entry['z_5'] = aligned_coords[i, 2] + corr_factor * noise * corr_weight + noise_z + \
                       dz2 * influence_2 + dz3 * influence_3 + dz4 * influence_4

    return result


# 7. ANA TAHMİN FONKSİYONU
def process_prediction(prediction):
    """Verilen tahmini işleyerek optimize edilmiş sonuçlar üretir"""
    # Temel Termal Ensemble Kabsch stratejisi
    structures = generate_thermal_ensemble_kabsch(prediction)
    return structures


# 8. TAHMİN OLUŞTURMA VE ÇALIŞTIRMA
print("\n--- TEST TAHMİNLERİ OLUŞTURULUYOR ---")

# İlk yapıları tahmin et
base_predictions = predict_base_structure(test_features, test_sequences)
print(f"Temel yapı tahminleri oluşturuldu: {len(base_predictions)} RNA için")

# Her RNA için Termal Ensemble Kabsch stratejisini uygula
print("Termal Ensemble Kabsch stratejisi uygulanıyor...")
all_predictions = []

for prediction in tqdm(base_predictions, desc="RNA Yapıları İşleniyor"):
    structures = process_prediction(prediction)
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
    "Yapı 2: Düşük Sıcaklık (0.25*10) - Laplace",
    "Yapı 3: Orta Sıcaklık (0.45*10) - Student's t + Normal",
    "Yapı 4: Yüksek Sıcaklık (0.75*10) - Student's t",
    "Yapı 5: Kabsch Hibrit (0.65*10) - Karma"
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
plt.savefig('basic_thermal_ensemble_kabsch.png')
plt.close()

# Toplam çalışma süresi
end_time = time.time()
print(f"\nTemel Termal Ensemble Kabsch Pipeline tamamlandı! Toplam süre: {(end_time - start_time):.2f} saniye")
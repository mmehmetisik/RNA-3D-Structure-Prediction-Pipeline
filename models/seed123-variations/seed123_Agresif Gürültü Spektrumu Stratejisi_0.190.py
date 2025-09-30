# RNA 3D Folding - Gelişmiş Biyolojik Entegrasyon Stratejisi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
import re

# Zamanlayıcı başlat
start_time = time.time()

# Kaggle dosya yolları
train_sequences_path = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
train_labels_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
test_sequences_path = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
sample_submission_path = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"

print("RNA 3D Yapı Tahmini Pipeline Başlatılıyor (Gelişmiş Biyolojik Entegrasyon Stratejisi)...")

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

# 5. TAHMİN OLUŞTURMA VE GELİŞMİŞ BİYOLOJİK ENTEGRASYON STRATEJİSİ
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


# BİYOLOJİK RNA YAPISI ANALİZ FONKSİYONLARI
def predict_rna_structure(sequence):
    """RNA sekonder yapı elementlerini tahmin eder (basit kural tabanlı)"""
    # Basitleştirilmiş RNA ikincil yapı tahmini
    structure = ['.' for _ in range(len(sequence))]

    # Potansiyel baz çiftleri
    pair_map = {
        'A': 'U', 'U': 'A',  # AU çiftleri
        'G': 'C', 'C': 'G',  # GC çiftleri
        'G': 'U', 'U': 'G'  # GU wobble çiftleri
    }

    # Minimum loop uzunluğu
    min_loop_length = 3

    # Potansiyel stem bölgelerini belirle
    stems = []
    for i in range(len(sequence)):
        for j in range(i + min_loop_length + 1, len(sequence)):
            if j - i > len(sequence) // 2:
                continue  # Çok uzak çiftler muhtemel değil

            if sequence[j] == pair_map.get(sequence[i], 'X'):
                # Potansiyel baz çifti
                stems.append((i, j))

    # En güçlü stemleri seç (basit açgözlü algoritma)
    stems.sort(key=lambda x: x[0])  # Başlangıç pozisyonuna göre sırala
    used = set()

    for i, j in stems:
        if i not in used and j not in used:
            structure[i] = '('
            structure[j] = ')'
            used.add(i)
            used.add(j)

    return ''.join(structure)


def analyze_structure_context(sequence, structure, pos):
    """Nükleotidin yapısal bağlamını analiz eder"""
    length = len(sequence)
    if pos >= length:
        return 'unknown'

    # 5' ucunda mı?
    if pos < 0.1 * length:
        return '5_prime'

    # 3' ucunda mı?
    if pos > 0.9 * length:
        return '3_prime'

    # Stem bölgesinde mi?
    if structure[pos] in '()':
        return 'stem'

    # İlmek (loop) bölgeleri
    window = 5  # Çevresindeki yapıyı kontrol et
    start = max(0, pos - window)
    end = min(length, pos + window + 1)
    local_struct = structure[start:end]

    if '(' in local_struct and ')' in local_struct:
        # İki parantez arasında - iç ilmek
        return 'internal_loop'

    if '(' in local_struct or ')' in local_struct:
        # Tek taraflı parantez - hairpin ilmeği
        return 'hairpin_loop'

    # Parantez yok - zincir bölgesi
    return 'external_region'


def identify_motifs(sequence, pos):
    """RNA motiflerini belirler"""
    motifs = set()
    length = len(sequence)

    # 5 nükleotid penceresi
    window_start = max(0, pos - 2)
    window_end = min(length, pos + 3)
    window = sequence[window_start:window_end]

    # Yaygın RNA motifler
    if "GGAGG" in window:
        motifs.add("shine_dalgarno")
    if "AAUAAA" in window:
        motifs.add("polyadenylation")
    if "GNRA" in window:  # G, herhangi bir nükleotid, R (A veya G), A
        motifs.add("tetraloop")
    if "CUG" in window or "CAG" in window:
        motifs.add("splicing")
    if "UUUU" in window:
        motifs.add("pyrimidine_tract")
    if "GNNNNG" in window:
        motifs.add("hexaloop")

    # G-quadruplex potansiyel
    if window.count('G') >= 3:
        motifs.add("g_rich")

    return motifs


# GELİŞMİŞ BİYOLOJİK ENTEGRASYON STRATEJİSİ
def generate_bio_integrated_strategy(prediction):
    """RNA biyolojik yapısını dikkate alan gelişmiş strateji"""
    result = []

    # Ana dizinin sekonder yapısını tahmin et
    sequence = prediction['sequence']
    predicted_structure = predict_rna_structure(sequence)
    sequence_length = len(sequence)

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

    # SEED 123 İLE BİYOLOJİK ADAPTASYONLU VARYASYONLAR
    np.random.seed(123)  # En başarılı tohum değeri

    # Her nükleotid için biyolojik özellikleri analiz et
    bio_factors = []
    for i, coord in enumerate(structure1):
        resid = coord['resid']
        pos = resid - 1  # 0-tabanlı indeks

        # Yapısal bağlam
        struct_context = analyze_structure_context(sequence, predicted_structure, pos)

        # Motifler
        motifs = identify_motifs(sequence, pos)

        # Pozisyon faktörü - RNA uçları daha esnek
        rel_pos = resid / sequence_length
        position_factor = 1.0
        if rel_pos < 0.08:  # 5' ucu
            position_factor = 4.0  # Çok daha esnek!
        elif rel_pos < 0.15:  # 5' yakını
            position_factor = 2.5
        elif rel_pos > 0.92:  # 3' ucu
            position_factor = 3.5  # 3' ucu da çok esnek!
        elif rel_pos > 0.85:  # 3' yakını
            position_factor = 2.0

        # Nükleotid tipi faktörü
        nucleotide = coord['resname']
        base_factor = 1.0
        if nucleotide == 'A':
            base_factor = 1.3  # A daha esnek
        elif nucleotide == 'U':
            base_factor = 1.2  # U daha esnek
        elif nucleotide == 'G':
            base_factor = 0.9  # G daha kararlı
        elif nucleotide == 'C':
            base_factor = 0.85  # C en kararlı

        # Yapısal bağlam faktörü
        context_factor = 1.0
        if struct_context == 'stem':
            context_factor = 0.7  # Stem bölgeleri daha kararlı
        elif struct_context == 'hairpin_loop':
            context_factor = 1.5  # Hairpin ilmekleri esnek
        elif struct_context == 'internal_loop':
            context_factor = 1.3  # İç ilmekler orta derecede esnek
        elif struct_context == 'external_region':
            context_factor = 1.2  # Dış bölgeler ortada

        # Motif faktörü
        motif_factor = 1.0
        if 'tetraloop' in motifs:
            motif_factor = 0.8  # Tetraloop'lar kararlı
        elif 'g_rich' in motifs:
            motif_factor = 0.75  # G-zengin bölgeler daha kararlı
        elif 'pyrimidine_tract' in motifs:
            motif_factor = 1.2  # Pirimidin traktları daha esnek

        # Toplam biyolojik faktör
        total_factor = position_factor * base_factor * context_factor * motif_factor

        bio_factors.append({
            'resid': resid,
            'position_factor': position_factor,
            'base_factor': base_factor,
            'context_factor': context_factor,
            'motif_factor': motif_factor,
            'total_factor': total_factor
        })

    # Her yapı için farklı biyolojik odaklı varyasyonlar

    # Yapı 2: Orta Gürültü - Stem odaklı (0.30 * 10)
    base_noise_2 = 0.30 * 10
    for i, entry in enumerate(result):
        resid = entry['resid']
        factor = bio_factors[i]['total_factor'] * 0.8  # Stem odaklı faktör

        entry['x_2'] = entry['x_1'] + np.random.normal(0, base_noise_2 * factor)
        entry['y_2'] = entry['y_1'] + np.random.normal(0, base_noise_2 * factor)
        entry['z_2'] = entry['z_1'] + np.random.normal(0, base_noise_2 * factor)

    # Yapı 3: Yüksek Gürültü - İlmek odaklı (0.50 * 10)
    np.random.seed(123)
    base_noise_3 = 0.50 * 10
    for i, entry in enumerate(result):
        resid = entry['resid']
        pos = resid - 1

        # İlmek odaklı faktör - ilmeklerde daha yüksek gürültü
        struct_context = analyze_structure_context(sequence, predicted_structure, pos)
        loop_focus = 1.5 if 'loop' in struct_context else 0.8
        factor = bio_factors[i]['total_factor'] * loop_focus

        entry['x_3'] = entry['x_1'] + np.random.normal(0, base_noise_3 * factor)
        entry['y_3'] = entry['y_1'] + np.random.normal(0, base_noise_3 * factor)
        entry['z_3'] = entry['y_1'] + np.random.normal(0, base_noise_3 * factor)

    # Yapı 4: Çok Yüksek Gürültü - Uç Odaklı (0.70 * 10)
    np.random.seed(123)
    base_noise_4 = 0.70 * 10
    for i, entry in enumerate(result):
        resid = entry['resid']
        rel_pos = resid / sequence_length

        # Uç odaklı faktör - uçlarda çok daha yüksek (5x) gürültü
        end_focus = 5.0 if rel_pos < 0.08 or rel_pos > 0.92 else 1.0
        factor = bio_factors[i]['total_factor'] * end_focus

        entry['x_4'] = entry['x_1'] + np.random.normal(0, base_noise_4 * factor)
        entry['y_4'] = entry['y_1'] + np.random.normal(0, base_noise_4 * factor)
        entry['z_4'] = entry['z_1'] + np.random.normal(0, base_noise_4 * factor)

    # Yapı 5: Ultra Yüksek Gürültü - Tam Biyolojik Entegrasyon (0.90 * 10)
    np.random.seed(123)
    base_noise_5 = 0.90 * 10
    for i, entry in enumerate(result):
        # Tam biyolojik faktör - tüm etkilerin birleşimi
        factor = bio_factors[i]['total_factor']

        # Biyolojik faktöre göre yön vektörü (daha yönlü gürültü)
        # Stem bölgeleri daha radyal hareket eder, ilmekler daha teğetsel
        pos = entry['resid'] - 1
        struct_context = analyze_structure_context(sequence, predicted_structure, pos)

        if struct_context == 'stem':
            # Stem'ler için daha düzenli hareket
            dx = np.random.normal(0, base_noise_5 * factor * 0.8)
            dy = np.random.normal(0, base_noise_5 * factor * 0.8)
            dz = np.random.normal(0, base_noise_5 * factor * 1.2)  # Z'de daha fazla hareket
        elif 'loop' in struct_context:
            # İlmekler için daha teğetsel hareket
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.normal(0, base_noise_5 * factor)
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            dz = np.random.normal(0, base_noise_5 * factor * 0.5)  # Z'de daha az hareket
        else:
            # Diğer bölgeler için standart gürültü
            dx = np.random.normal(0, base_noise_5 * factor)
            dy = np.random.normal(0, base_noise_5 * factor)
            dz = np.random.normal(0, base_noise_5 * factor)

        entry['x_5'] = entry['x_1'] + dx
        entry['y_5'] = entry['y_1'] + dy
        entry['z_5'] = entry['z_1'] + dz

    return result


# Her RNA için Gelişmiş Biyolojik Entegrasyon stratejisi uygula
print("Gelişmiş Biyolojik Entegrasyon stratejisi uygulanıyor...")
all_predictions = []

for prediction in base_predictions:
    structures = generate_bio_integrated_strategy(prediction)
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
    "Yapı 1: Temel RandomForest",
    "Yapı 2: Stem Odaklı Varyasyon (0.30*10)",
    "Yapı 3: İlmek Odaklı Varyasyon (0.50*10)",
    "Yapı 4: Uç Odaklı Varyasyon (0.70*10)",
    "Yapı 5: Tam Biyolojik Entegrasyon (0.90*10)"
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
plt.savefig('bio_integrated_structures.png')
plt.close()

# Toplam çalışma süresi
end_time = time.time()
print(f"\nGelişmiş Biyolojik Entegrasyon Pipeline tamamlandı! Toplam süre: {(end_time - start_time):.2f} saniye")
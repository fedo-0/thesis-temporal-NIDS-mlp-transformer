import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

def load_dataset_config(config_path="config/dataset.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['dataset']

def frequency_encoding_fit(df_train, categorical_columns, max_levels=32):
    freq_mappings = {}
    for col in categorical_columns:
        if col in df_train.columns:
            value_counts = df_train[col].value_counts()
            top_values = value_counts.nlargest(max_levels).index.tolist()
            freq_map = {val: i + 1 for i, val in enumerate(top_values)}
            freq_mappings[col] = freq_map
            print(f"Frequency encoding mappatura creata per colonna: {col} ({len(freq_map)} valori)")
        else:
            print(f"Attenzione: colonna {col} non trovata nel dataset")
    return freq_mappings

def frequency_encoding_transform(df, freq_mappings):
    df_encoded = df.copy()
    for col, freq_map in freq_mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(lambda x: freq_map.get(x, 0))
        else:
            print(f"Attenzione: colonna {col} non trovata nel dataset")
    return df_encoded

def log1p_transform(df, numeric_columns):
    df_transformed = df.copy()
    for col in numeric_columns:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))
        else:
            print(f"Attenzione: colonna {col} non trovata nel dataset")
    return df_transformed

def minmax_scaling_fit(df_train, numeric_columns):
    scaler = MinMaxScaler()
    numeric_cols_present = [col for col in numeric_columns if col in df_train.columns]
    if numeric_cols_present:
        scaler.fit(df_train[numeric_cols_present])
        print(f"MinMax scaler fitted su {len(numeric_cols_present)} colonne numeriche del training set")
    return scaler, numeric_cols_present

def minmax_scaling_transform(df, scaler, numeric_cols_present):
    df_scaled = df.copy()
    if numeric_cols_present:
        df_scaled[numeric_cols_present] = scaler.transform(df_scaled[numeric_cols_present])
    return df_scaled

def analyze_packet_distribution(df, target_column):
    print(f"\n--- ANALISI DISTRIBUZIONE PACCHETTI ---")
    if df[target_column].dtype == 'object':
        attack_mask = df[target_column] != 'Benign'
    else:
        attack_mask = df[target_column] != 0
    total_packets = len(df)
    attack_packets = attack_mask.sum()
    benign_packets = total_packets - attack_packets
    print(f"Pacchetti totali: {total_packets:,}")
    print(f"Pacchetti di attacco: {attack_packets:,} ({attack_packets/total_packets*100:.2f}%)")
    print(f"Pacchetti benigni: {benign_packets:,} ({benign_packets/total_packets*100:.2f}%)")
    if attack_packets > 0:
        attack_positions = df[attack_mask].index.tolist()
        first_attack = attack_positions[0]
        last_attack = attack_positions[-1]
        attack_span = (last_attack + 1) / total_packets
        print(f"Primo attacco: posizione {first_attack:,}")
        print(f"Ultimo attacco: posizione {last_attack:,}")
        print(f"Span degli attacchi: {attack_span*100:.1f}% del dataset")
        return {
            'total_packets': total_packets,
            'attack_packets': attack_packets,
            'last_attack_position': last_attack,
            'attack_span': attack_span
        }
    else:
        print("Nessun attacco trovato nel dataset!")
        return {
            'total_packets': total_packets,
            'attack_packets': 0,
            'last_attack_position': 0,
            'attack_span': 0
        }

def create_micro_windows(df, target_column, min_window_size=10, max_window_size=30):
    print(f"\n--- CREAZIONE MICRO-FINESTRE ({min_window_size}-{max_window_size} pacchetti) ---")
    if df[target_column].dtype == 'object':
        attack_mask = df[target_column] != 'Benign'
    else:
        attack_mask = df[target_column] != 0
    micro_windows = []
    current_pos = 0
    window_id = 0
    while current_pos < len(df):
        remaining_packets = len(df) - current_pos
        max_possible_size = min(max_window_size, remaining_packets)
        window_size = min_window_size
        lookahead_window = df.iloc[current_pos:current_pos + max_possible_size]
        lookahead_attack_ratio = attack_mask.iloc[current_pos:current_pos + max_possible_size].mean()
        if lookahead_attack_ratio == 0:
            window_size = max_possible_size
        elif lookahead_attack_ratio < 0.1:
            window_size = min(max_window_size, max_possible_size)
        else:
            window_size = min(min_window_size + 5, max_possible_size)
        end_pos = min(current_pos + window_size, len(df))
        micro_window_data = df.iloc[current_pos:end_pos].copy()
        window_attack_mask = attack_mask.iloc[current_pos:end_pos]
        attack_count = window_attack_mask.sum()
        attack_ratio = attack_count / len(micro_window_data)
        window_info = {
            'id': window_id,
            'data': micro_window_data,
            'start_pos': current_pos,
            'end_pos': end_pos,
            'size': len(micro_window_data),
            'attack_count': attack_count,
            'attack_ratio': attack_ratio,
            'benign_count': len(micro_window_data) - attack_count
        }
        micro_windows.append(window_info)
        current_pos = end_pos
        window_id += 1
        if window_id % 1000 == 0:
            print(f"  Processate {window_id} micro-finestre...")
    print(f"Totale micro-finestre create: {len(micro_windows)}")
    sizes = [w['size'] for w in micro_windows]
    attack_ratios = [w['attack_ratio'] for w in micro_windows]
    print(f"\nStatistiche Micro-Finestre:")
    print(f"  Dimensione media: {np.mean(sizes):.1f} pacchetti")
    print(f"  Range dimensioni: {min(sizes)}-{max(sizes)} pacchetti")
    print(f"  Ratio attacchi medio: {np.mean(attack_ratios):.3f}")
    print(f"  Finestre con attacchi: {sum(1 for r in attack_ratios if r > 0)} ({sum(1 for r in attack_ratios if r > 0)/len(attack_ratios)*100:.1f}%)")
    print(f"  Finestre solo benigne: {sum(1 for r in attack_ratios if r == 0)} ({sum(1 for r in attack_ratios if r == 0)/len(attack_ratios)*100:.1f}%)")
    return micro_windows

def stratified_window_assignment(micro_windows, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    print(f"\n--- ASSEGNAZIONE STRATIFICATA MICRO-FINESTRE ({train_ratio*100:.0f}-{val_ratio*100:.0f}-{test_ratio*100:.0f}) ---")
    pure_benign = [w for w in micro_windows if w['attack_ratio'] == 0]
    low_attack = [w for w in micro_windows if 0 < w['attack_ratio'] <= 0.2]
    medium_attack = [w for w in micro_windows if 0.2 < w['attack_ratio'] <= 0.6]
    high_attack = [w for w in micro_windows if w['attack_ratio'] > 0.6]
    categories = {
        'pure_benign': pure_benign,
        'low_attack': low_attack,
        'medium_attack': medium_attack,
        'high_attack': high_attack
    }
    print("Distribuzione categorie micro-finestre:")
    for name, windows in categories.items():
        print(f"  {name}: {len(windows)} finestre ({len(windows)/len(micro_windows)*100:.1f}%)")
    train_windows, val_windows, test_windows = [], [], []
    for category_name, windows in categories.items():
        if len(windows) == 0:
            continue
        for i, window in enumerate(windows):
            cycle_pos = i % 20
            if cycle_pos < 14:
                train_windows.append(window)
            elif cycle_pos < 17:
                val_windows.append(window)
            else:
                test_windows.append(window)
        print(f"  {category_name}: Train={len([w for w in windows if w in train_windows])}, Val={len([w for w in windows if w in val_windows])}, Test={len([w for w in windows if w in test_windows])}")
    train_windows.sort(key=lambda x: x['start_pos'])
    val_windows.sort(key=lambda x: x['start_pos'])
    test_windows.sort(key=lambda x: x['start_pos'])
    print(f"\nTotale finestre assegnate:")
    print(f"  Train: {len(train_windows)} finestre")
    print(f"  Validation: {len(val_windows)} finestre")
    print(f"  Test: {len(test_windows)} finestre")
    return train_windows, val_windows, test_windows

def reconstruct_datasets(train_windows, val_windows, test_windows):
    print(f"\n--- RICOSTRUZIONE DATASET FINALI ---")
    train_data = pd.concat([w['data'] for w in train_windows], ignore_index=True)
    val_data = pd.concat([w['data'] for w in val_windows], ignore_index=True)
    test_data = pd.concat([w['data'] for w in test_windows], ignore_index=True)
    print(f"Dataset ricostruiti:")
    print(f"  Train: {len(train_data)} pacchetti")
    print(f"  Validation: {len(val_data)} pacchetti")
    print(f"  Test: {len(test_data)} pacchetti")
    return train_data, val_data, test_data

def micro_window_split(df, target_column, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, 
                      min_window_size=10, max_window_size=30):
    print("MICRO-WINDOW STRATIFIED TEMPORAL SPLIT")
    print("=" * 60)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    packet_stats = analyze_packet_distribution(df, target_column)
    micro_windows = create_micro_windows(df, target_column, min_window_size, max_window_size)
    if len(micro_windows) < 3:
        raise ValueError("Troppo poche micro-finestre per creare train/val/test set")
    train_windows, val_windows, test_windows = stratified_window_assignment(
        micro_windows, train_ratio, val_ratio, test_ratio
    )
    train_data, val_data, test_data = reconstruct_datasets(train_windows, val_windows, test_windows)
    return train_data, val_data, test_data

def preprocess_dataset_binary(dataset_path, config_path, output_dir, 
                      train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                      min_window_size=10, max_window_size=30):
    config = load_dataset_config(config_path)
    numeric_columns = config['numeric_columns']
    categorical_columns = config['categorical_columns']
    target_column = config['target_column']
    print(f"Configurazione caricata:")
    print(f"- Colonne numeriche: {len(numeric_columns)}")
    print(f"- Colonne categoriche: {len(categorical_columns)}")
    print(f"- Colonna target: {target_column}")
    print(f"\nCaricamento dataset da: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset originale: {df.shape[0]} righe, {df.shape[1]} colonne")
    initial_rows = len(df)
    numeric_cols_present = [col for col in numeric_columns if col in df.columns]
    if numeric_cols_present:
        inf_count = 0
        for col in numeric_cols_present:
            col_inf_count = np.isinf(df[col]).sum()
            inf_count += col_inf_count
            if col_inf_count > 0:
                print(f"⚠️  Trovati {col_inf_count} valori infiniti in colonna: {col}")
        df[numeric_cols_present] = df[numeric_cols_present].replace([np.inf, -np.inf], np.nan)
        print(f"Sostituiti {inf_count} valori infiniti con NaN")
    df = df.dropna().reset_index(drop=True)
    removed_rows = initial_rows - len(df)
    print(f"Rimosse {removed_rows} righe contenenti valori NaN o infiniti")
    print(f"Dataset dopo pulizia: {df.shape[0]} righe, {df.shape[1]} colonne")
    if len(df) == 0:
        raise ValueError("Il dataset risulta vuoto dopo la rimozione dei valori NaN!")
    if target_column not in df.columns:
        raise ValueError(f"Colonna target '{target_column}' non trovata nel dataset!")
    feature_columns = [col for col in df.columns if col != target_column]
    train_data, val_data, test_data = micro_window_split(
        df, target_column, train_ratio, val_ratio, test_ratio, 
        min_window_size, max_window_size
    )
    X_train = train_data[feature_columns].copy()
    y_train = train_data[target_column].copy()
    X_val = val_data[feature_columns].copy()
    y_val = val_data[target_column].copy()
    X_test = test_data[feature_columns].copy()
    y_test = test_data[target_column].copy()
    total_original = len(df)
    print(f"\nVerifica totale pacchetti: {len(train_data) + len(val_data) + len(test_data)} / {total_original}")
    print(f"\nInizio preprocessing delle features...")
    print("\n--- FREQUENCY ENCODING ---")
    freq_mappings = frequency_encoding_fit(X_train, categorical_columns)
    X_train_encoded = frequency_encoding_transform(X_train, freq_mappings)
    X_val_encoded = frequency_encoding_transform(X_val, freq_mappings)
    X_test_encoded = frequency_encoding_transform(X_test, freq_mappings)
    print("Frequency encoding applicato a train, validation e test set")
    print("\n--- TRASFORMAZIONE LOG1P ---")
    X_train_log = log1p_transform(X_train_encoded, numeric_columns)
    X_val_log = log1p_transform(X_val_encoded, numeric_columns)
    X_test_log = log1p_transform(X_test_encoded, numeric_columns)
    print("Trasformazione log1p applicata a train, validation e test set")
    print("\n--- MINMAX SCALING ---")
    scaler, numeric_cols_present = minmax_scaling_fit(X_train_log, numeric_columns)
    X_train_scaled = minmax_scaling_transform(X_train_log, scaler, numeric_cols_present)
    X_val_scaled = minmax_scaling_transform(X_val_log, scaler, numeric_cols_present)
    X_test_scaled = minmax_scaling_transform(X_test_log, scaler, numeric_cols_present)
    print("MinMax scaling applicato a train, validation e test set")
    df_train = X_train_scaled.copy()
    df_train[target_column] = y_train.reset_index(drop=True)
    df_val = X_val_scaled.copy()
    df_val[target_column] = y_val.reset_index(drop=True)
    df_test = X_test_scaled.copy()
    df_test[target_column] = y_test.reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    print(f"\nSalvataggio dataset preprocessati in: {output_dir}")
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"Dataset salvati con successo!")
    print(f"- Training: {train_path} ({df_train.shape[0]} righe, {df_train.shape[1]} colonne)")
    print(f"- Validation: {val_path} ({df_val.shape[0]} righe, {df_val.shape[1]} colonne)")
    print(f"- Test: {test_path} ({df_test.shape[0]} righe, {df_test.shape[1]} colonne)")
    print(f"\n--- STATISTICHE FINALI DISTRIBUZIONE ATTACCHI ---")
    for name, dataset in [("Validation", df_val), ("Test", df_test), ("Training", df_train)]:
        total = len(dataset)
        if dataset[target_column].dtype == 'object':
            attacks = (dataset[target_column] != 'Benign').sum()
        else:
            attacks = (dataset[target_column] != 0).sum()
        benign = total - attacks
        print(f"\n{name} Set:")
        print(f"  Totale: {total:,} pacchetti ({total/(len(df_train)+len(df_val)+len(df_test))*100:.1f}%)")
        print(f"  Attacchi: {attacks:,} ({attacks/total*100:.2f}%)")
        print(f"  Benigni: {benign:,} ({benign/total*100:.2f}%)")
        print(f"  Distribuzione dettagliata:")
        value_counts = dataset[target_column].value_counts().sort_index()
        for value, count in value_counts.items():
            print(f"    {value}: {count:,} ({count/total*100:.2f}%)")
    return df_train, df_val, df_test, scaler, freq_mappings

if __name__ == "__main__":
    preprocess_dataset_binary(
        dataset_path="resources/datasets/NF-UNSW-NB15-v3.csv",
        config_path="config/dataset.json",
        output_dir="resources/datasets",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        min_window_size=10,
        max_window_size=30
    )

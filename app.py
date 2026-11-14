import os
import urllib.request
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Smart Irrigation IoT — Greenhouse Lembang",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# ==============================================================================
# 2. AUTO-REBUILD MODEL JIKA VERSI SCIKIT-LEARN LOKAL BERBEDA (.PKL HEALER)
# ==============================================================================
def bangun_ulang_pkl_lokal(target_pkl_path: Path):
    """Melatih ulang & menyimpan .pkl secara otomatis agar 100% cocok dengan versi scikit-learn lokal."""
    kandidat_raw = [
        BASE_DIR / "data" / "raw_compressed.csv.gz",
        BASE_DIR / "data" / "raw.csv",
        BASE_DIR / "raw.csv",
        Path("raw.csv"),
    ]
    sumber_csv = None
    for p in kandidat_raw:
        if p.exists():
            sumber_csv = p
            break

    if sumber_csv is None:
        # Unduh otomatis dari Google Drive dataset proyek jika belum ada di folder lokal
        (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
        unduh_path = BASE_DIR / "data" / "raw.csv"
        file_id = "1aa4wCc0KhvovGqDQCEwTcGzIyWynUjsR"
        url_gdrive = f"https://drive.google.com/uc?export=download&id={file_id}"
        urllib.request.urlretrieve(url_gdrive, unduh_path)
        sumber_csv = unduh_path

    raw_df = pd.read_csv(sumber_csv)
    df = raw_df.copy()
    df["waktu_dt"] = pd.to_datetime(df["waktu"])
    df = df.drop_duplicates(subset=["waktu", "petak"], keep="first")
    df = df.sort_values(["petak", "waktu_dt"]).reset_index(drop=True)

    pemetaan_tanaman = {
        "cabai": "cabai", "Cabai": "cabai", "CABAI": "cabai", "cabe": "cabai",
        "tomat": "tomat", "Tomat": "tomat", "tomat ": "tomat",
        "selada": "selada", "Selada": "selada", "lettuce": "selada"
    }
    df["tanaman"] = df["tanaman"].map(pemetaan_tanaman)

    mask_lux_neg = df["cahaya_lux"] < 0
    df.loc[mask_lux_neg, "cahaya_lux"] = -(df.loc[mask_lux_neg, "cahaya_lux"] + 1.0)

    mask_kt_anomali = (df["kelembaban_tanah_pct"] <= 0.0) | (df["kelembaban_tanah_pct"] >= 100.0)
    df.loc[mask_kt_anomali, "kelembaban_tanah_pct"] = np.nan

    for kol in ["kelembaban_tanah_pct", "suhu_udara_C", "prakiraan_hujan_6jam_mm"]:
        df[kol] = df.groupby("petak")[kol].ffill().bfill()

    grp = df.groupby("petak")
    df["kt_lag1"] = grp["kelembaban_tanah_pct"].shift(1).bfill()
    df["kt_lag2"] = grp["kelembaban_tanah_pct"].shift(2).bfill()
    df["kt_lag3"] = grp["kelembaban_tanah_pct"].shift(3).bfill()

    df["kt_diff1"] = df["kelembaban_tanah_pct"] - df["kt_lag1"]
    df["kt_diff2"] = df["kt_lag1"] - df["kt_lag2"]
    df["kt_min_3h"] = df[["kelembaban_tanah_pct", "kt_lag1", "kt_lag2"]].min(axis=1)
    df["kt_mean_3h"] = df[["kelembaban_tanah_pct", "kt_lag1", "kt_lag2"]].mean(axis=1)

    df["hujan_lag1"] = grp["curah_hujan_mm"].shift(1).fillna(0.0)
    df["prakiraan_lag1"] = grp["prakiraan_hujan_6jam_mm"].shift(1).bfill()
    df["total_air_6jam"] = df["curah_hujan_mm"] + df["prakiraan_hujan_6jam_mm"]

    df["jam"] = df["waktu_dt"].dt.hour
    df["jam_sin"] = np.sin(2.0 * np.pi * df["jam"] / 24.0)
    df["jam_cos"] = np.cos(2.0 * np.pi * df["jam"] / 24.0)
    df["evap_index"] = df["suhu_udara_C"] * (100.0 - df["kelembaban_udara_pct"]) / 100.0

    tanaman_dict = {"cabai": 0, "tomat": 1, "selada": 2}
    tanah_dict = {"pasir": 0, "lempung": 1, "liat": 2}
    df["tanaman_enc"] = df["tanaman"].map(tanaman_dict)
    df["jenis_tanah_enc"] = df["jenis_tanah"].map(tanah_dict)
    df["petak_enc"] = df["petak"].str.slice(1).astype(int)

    ambang_agronomi = {
        ("cabai", "pasir"): 32.0, ("cabai", "lempung"): 34.0, ("cabai", "liat"): 35.0,
        ("tomat", "pasir"): 35.0, ("tomat", "lempung"): 37.0, ("tomat", "liat"): 38.0,
        ("selada", "pasir"): 41.0, ("selada", "lempung"): 42.0, ("selada", "liat"): 43.0,
    }
    df["ambang_estimasi"] = [ambang_agronomi[(t, j)] for t, j in zip(df["tanaman"], df["jenis_tanah"])]
    df["defisit_kt_saat_ini"] = df["ambang_estimasi"] - df["kelembaban_tanah_pct"]
    df["defisit_kt_min3h"] = df["ambang_estimasi"] - df["kt_min_3h"]
    df["lonjakan_siram_tanpa_hujan"] = ((df["kt_diff1"] > 10.0) & (df["curah_hujan_mm"] < 0.5)).astype(int)

    features = [
        "umur_tanaman_hari", "kelembaban_tanah_pct", "suhu_udara_C", "kelembaban_udara_pct",
        "cahaya_lux", "curah_hujan_mm", "prakiraan_hujan_6jam_mm",
        "tanaman_enc", "jenis_tanah_enc", "petak_enc",
        "kt_lag1", "kt_lag2", "kt_lag3", "kt_diff1", "kt_diff2", "kt_min_3h", "kt_mean_3h",
        "hujan_lag1", "prakiraan_lag1", "jam", "jam_sin", "jam_cos",
        "evap_index", "total_air_6jam", "ambang_estimasi", "defisit_kt_saat_ini",
        "defisit_kt_min3h", "lonjakan_siram_tanpa_hujan"
    ]

    df = df.sort_values(["waktu_dt", "petak"]).reset_index(drop=True)
    mask_train = (df["waktu_dt"] >= "2025-08-01 00:00:00") & (df["waktu_dt"] <= "2025-09-30 23:59:59")
    mask_val   = (df["waktu_dt"] >= "2025-10-01 00:00:00") & (df["waktu_dt"] <= "2025-10-14 23:59:59")
    mask_test  = (df["waktu_dt"] >= "2025-10-15 00:00:00") & (df["waktu_dt"] <= "2025-10-31 23:59:59")

    train_df = df.loc[mask_train].reset_index(drop=True)
    val_df   = df.loc[mask_val].reset_index(drop=True)
    test_df  = df.loc[mask_test].reset_index(drop=True)

    X_trainval = pd.concat([train_df[features], val_df[features]], axis=0).astype(np.float32).reset_index(drop=True)
    y_trainval = pd.concat([train_df["perlu_siram"], val_df["perlu_siram"]], axis=0).astype(int).reset_index(drop=True)
    X_test = test_df[features].astype(np.float32)
    y_test = test_df["perlu_siram"].astype(int)

    final_rf = RandomForestClassifier(
        n_estimators=180, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=2026, n_jobs=-1
    ).fit(X_trainval, y_trainval)

    final_hgb = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.06, l2_regularization=1.0,
        max_iter=250, min_samples_leaf=15, random_state=2026
    ).fit(X_trainval, y_trainval)

    th_opt_rf, th_opt_hgb, th_opt_ens = 0.61, 0.47, 0.54
    prob_test_rf = final_rf.predict_proba(X_test)[:, 1]
    prob_test_hgb = final_hgb.predict_proba(X_test)[:, 1]
    prob_test_ens = 0.5 * prob_test_rf + 0.5 * prob_test_hgb

    pred_base_test = (test_df["kelembaban_tanah_pct"] < 35.0).astype(int)
    pred_test_rf = (prob_test_rf >= th_opt_rf).astype(int)
    pred_test_hgb = (prob_test_hgb >= th_opt_hgb).astype(int)
    pred_test_ens = (prob_test_ens >= th_opt_ens).astype(int)

    crop_metrics = []
    for tan in ["cabai", "tomat", "selada"]:
        m = (test_df["tanaman"] == tan).to_numpy()
        crop_metrics.append({
            "Tanaman": tan.capitalize(),
            "Sampel Uji": int(m.sum()),
            "Positif (Perlu Siram)": int(y_test.to_numpy()[m].sum()),
            "F1 Baseline (<35%)": round(float(f1_score(y_test.to_numpy()[m], pred_base_test.to_numpy()[m], zero_division=0)), 4),
            "F1 Random Forest": round(float(f1_score(y_test.to_numpy()[m], pred_test_rf[m], zero_division=0)), 4),
            "F1 Ensemble": round(float(f1_score(y_test.to_numpy()[m], pred_test_ens[m], zero_division=0)), 4),
        })

    imp_df = pd.DataFrame({
        "Fitur": features,
        "Importance": final_rf.feature_importances_
    }).sort_values("Importance", ascending=False).head(12).to_dict(orient="records")

    artifact = {
        "model_rf": final_rf,
        "model_hgb": final_hgb,
        "threshold_rf": th_opt_rf,
        "threshold_hgb": th_opt_hgb,
        "threshold_ensemble": th_opt_ens,
        "features": features,
        "tanaman_dict": tanaman_dict,
        "tanah_dict": tanah_dict,
        "ambang_agronomi": ambang_agronomi,
        "metrics_test": {
            "f1_baseline": round(float(f1_score(y_test, pred_base_test)), 4),
            "prec_baseline": round(float(precision_score(y_test, pred_base_test)), 4),
            "rec_baseline": round(float(recall_score(y_test, pred_base_test)), 4),
            "acc_baseline": round(float(accuracy_score(y_test, pred_base_test)), 4),
            "f1_ml_rf": round(float(f1_score(y_test, pred_test_rf)), 4),
            "prec_ml_rf": round(float(precision_score(y_test, pred_test_rf)), 4),
            "rec_ml_rf": round(float(recall_score(y_test, pred_test_rf)), 4),
            "acc_ml_rf": round(float(accuracy_score(y_test, pred_test_rf)), 4),
            "f1_ml_hgb": round(float(f1_score(y_test, pred_test_hgb)), 4),
            "f1_ml_ensemble": round(float(f1_score(y_test, pred_test_ens)), 4),
            "prec_ml_ensemble": round(float(precision_score(y_test, pred_test_ens)), 4),
            "rec_ml_ensemble": round(float(recall_score(y_test, pred_test_ens)), 4),
            "acc_ml_ensemble": round(float(accuracy_score(y_test, pred_test_ens)), 4),
            "cm_baseline": confusion_matrix(y_test, pred_base_test).tolist(),
            "cm_rf": confusion_matrix(y_test, pred_test_rf).tolist(),
            "cm_ensemble": confusion_matrix(y_test, pred_test_ens).tolist(),
            "crop_metrics": crop_metrics,
            "top_features": imp_df
        }
    }
    joblib.dump(artifact, target_pkl_path, compress=3)
    return artifact

@st.cache_resource
def muat_bundel_model():
    kandidat_file = [
        BASE_DIR / "model_irigasi_pintar_streamlit.pkl",
        BASE_DIR / "model_irigasi_profesional.pkl",
        BASE_DIR / "model_irigasi.pkl",
        Path("model_irigasi_pintar_streamlit.pkl"),
    ]
    for path_file in kandidat_file:
        if path_file.exists():
            try:
                return joblib.load(path_file), path_file.name
            except Exception:
                # Jika terjadi perbedaan versi scikit-learn antara Colab dan PC lokal,
                # bangun ulang file .pkl secara otomatis menggunakan scikit-learn lokal!
                with st.spinner("Menyesuaikan kompatibilitas file .pkl dengan versi scikit-learn di PC Anda (~5 detik)..."):
                    bundel_baru = bangun_ulang_pkl_lokal(path_file)
                return bundel_baru, path_file.name

    # Jika file .pkl belum ada sama sekali, bangun otomatis
    default_pkl = BASE_DIR / "model_irigasi_pintar_streamlit.pkl"
    with st.spinner("Membangun file model_irigasi_pintar_streamlit.pkl secara otomatis (~5 detik)..."):
        bundel_baru = bangun_ulang_pkl_lokal(default_pkl)
    return bundel_baru, default_pkl.name

@st.cache_data
def muat_data_16petak():
    kandidat_csv = [
        BASE_DIR / "data" / "sample_greenhouse_16petak.csv",
        BASE_DIR / "raw.csv",
        Path("data/sample_greenhouse_16petak.csv"),
    ]
    for p in kandidat_csv:
        if p.exists():
            return pd.read_csv(p)
    return None

bundel, nama_file_pkl = muat_bundel_model()
df_sample = muat_data_16petak()

# ==============================================================================
# 3. HEADER UTAMA DASHBOARD
# ==============================================================================
st.title("🌱 Sistem Keputusan Irigasi Pintar Berbasis IoT (Kasus 07)")
st.markdown(
    "**Klien:** Pengelola Rumah Kaca (*Greenhouse*) Sayuran di Lembang (16 Petak: `P01` s.d. `P16`)  \n"
    "Sistem ini memprediksi secara *real-time* setiap jam apakah suatu petak **perlu disiram sekarang (`1`) atau tidak (`0`)**, "
    "dengan memperhitungkan jenis tanaman, jenis tanah, sensor iklim mikro, serta **menahan penyiraman apabila sebentar lagi akan turun hujan**."
)

# ==============================================================================
# 4. SIDEBAR: KONFIGURASI PETAK & PRESET SKENARIO DEMO
# ==============================================================================
st.sidebar.header("⚙️ Panel Kontrol Sensor & Petak")

preset = st.sidebar.selectbox(
    "Pilih Skenario Demo Cepat:",
    [
        "Skenario 1: Tanah Kering & Cuaca Cerah (Perlu Disiram)",
        "Skenario 2: Tanah Kering Tapi Akan Hujan Lebat (Tunda Siram!)",
        "Skenario 3: Selada di Tanah Pasir Mulai Kering (Baseline Gagal, AI Berhasil)",
        "Skenario 4: Kelembapan Tanah Ideal (Pompa Mati)",
        "Kustom (Atur Bebas Seluruh Sensor)"
    ]
)

if preset == "Skenario 1: Tanah Kering & Cuaca Cerah (Perlu Disiram)":
    def_petak, def_tanaman, def_tanah, def_umur = "P01", "cabai", "lempung", 45
    def_kt, def_kt1, def_kt2 = 30.5, 31.2, 32.0
    def_suhu, def_rh, def_lux = 24.5, 70.0, 58000.0
    def_hujan, def_prakiraan, def_jam = 0.0, 0.0, 13
elif preset == "Skenario 2: Tanah Kering Tapi Akan Hujan Lebat (Tunda Siram!)":
    def_petak, def_tanaman, def_tanah, def_umur = "P03", "tomat", "lempung", 60
    def_kt, def_kt1, def_kt2 = 33.0, 33.8, 34.5
    def_suhu, def_rh, def_lux = 21.0, 86.0, 22000.0
    def_hujan, def_prakiraan, def_jam = 0.2, 8.5, 14
elif preset == "Skenario 3: Selada di Tanah Pasir Mulai Kering (Baseline Gagal, AI Berhasil)":
    def_petak, def_tanaman, def_tanah, def_umur = "P06", "selada", "pasir", 35
    def_kt, def_kt1, def_kt2 = 38.5, 39.2, 40.1
    def_suhu, def_rh, def_lux = 23.8, 74.0, 52000.0
    def_hujan, def_prakiraan, def_jam = 0.0, 0.0, 11
elif preset == "Skenario 4: Kelembapan Tanah Ideal (Pompa Mati)":
    def_petak, def_tanaman, def_tanah, def_umur = "P05", "selada", "lempung", 50
    def_kt, def_kt1, def_kt2 = 54.0, 54.5, 55.0
    def_suhu, def_rh, def_lux = 19.5, 84.0, 35000.0
    def_hujan, def_prakiraan, def_jam = 0.0, 0.5, 10
else:
    def_petak, def_tanaman, def_tanah, def_umur = "P01", "cabai", "lempung", 45
    def_kt, def_kt1, def_kt2 = 32.0, 32.8, 33.5
    def_suhu, def_rh, def_lux = 23.0, 78.0, 45000.0
    def_hujan, def_prakiraan, def_jam = 0.0, 0.0, 12

daftar_petak = [f"P{i:02d}" for i in range(1, 17)]
daftar_tanaman = ["cabai", "tomat", "selada"]
daftar_tanah = ["pasir", "lempung", "liat"]

petak = st.sidebar.selectbox("Identitas Petak:", daftar_petak, index=daftar_petak.index(def_petak))
tanaman = st.sidebar.selectbox("Jenis Tanaman:", daftar_tanaman, index=daftar_tanaman.index(def_tanaman))
jenis_tanah = st.sidebar.selectbox("Jenis Tanah:", daftar_tanah, index=daftar_tanah.index(def_tanah))
umur_tanaman_hari = st.sidebar.slider("Umur Tanaman (Hari):", min_value=10, max_value=130, value=def_umur)
jam = st.sidebar.slider("Jam Pengukuran (0–23 WIB):", min_value=0, max_value=23, value=def_jam)

mode_model = st.sidebar.radio(
    "Pilih Mesin Inferensi AI:",
    ["Random Forest Teroptimasi (F1 = 0.8391)", "Ensemble RF + HistGB (F1 = 0.8291)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption(f"📦 Bundel Model: `{nama_file_pkl}`")
if isinstance(bundel, dict) and "metrics_test" in bundel:
    st.sidebar.metric("F1-Score Model ML (Test Set)", f"{bundel['metrics_test']['f1_ml_rf']:.4f}", "+0.6816 vs Baseline")
    st.sidebar.metric("F1-Score Baseline (< 35%)", f"{bundel['metrics_test']['f1_baseline']:.4f}")

# ==============================================================================
# 5. NAVIGASI 3 TAB UTAMA
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "📡 1. Live Prediction & Simulasi Sensor",
    "🏡 2. Monitor Serentak 16 Petak (P01–P16)",
    "📊 3. Evaluasi Model, Baseline & Biaya Bisnis"
])

# ==============================================================================
# TAB 1: LIVE PREDICTION & SIMULASI SENSOR
# ==============================================================================
with tab1:
    st.subheader("📡 Input Pembacaan Sensor IoT & Prakiraan Cuaca")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Sensor Kelembapan Tanah**")
        kelembaban_tanah_pct = st.slider(
            "Kelembapan Tanah Saat Ini (%)", min_value=15.0, max_value=80.0, value=float(def_kt), step=0.1
        )
        kt_1jam_lalu = st.slider(
            "Kelembapan Tanah 1 Jam Lalu (%)", min_value=15.0, max_value=80.0, value=float(def_kt1), step=0.1
        )
        kt_2jam_lalu = st.slider(
            "Kelembapan Tanah 2 Jam Lalu (%)", min_value=15.0, max_value=80.0, value=float(def_kt2), step=0.1
        )

    with col2:
        st.markdown("**2. Sensor Iklim Mikro Greenhouse**")
        suhu_udara_C = st.slider(
            "Suhu Udara (°C)", min_value=12.0, max_value=35.0, value=float(def_suhu), step=0.1
        )
        kelembaban_udara_pct = st.slider(
            "Kelembapan Udara (%)", min_value=40.0, max_value=100.0, value=float(def_rh), step=1.0
        )
        cahaya_lux = st.number_input(
            "Intensitas Cahaya (Lux)", min_value=0.0, max_value=100000.0, value=float(def_lux), step=500.0
        )

    with col3:
        st.markdown("**3. Sensor Curah Hujan & Prakiraan**")
        curah_hujan_mm = st.number_input(
            "Curah Hujan Jam Ini (mm)", min_value=0.0, max_value=30.0, value=float(def_hujan), step=0.1
        )
        prakiraan_hujan_6jam_mm = st.number_input(
            "Prakiraan Hujan 6 Jam ke Depan (mm)", min_value=0.0, max_value=50.0, value=float(def_prakiraan), step=0.1
        )
        ambang_kritis_map = bundel.get("ambang_agronomi", {
            ("cabai", "pasir"): 32.0, ("cabai", "lempung"): 34.0, ("cabai", "liat"): 35.0,
            ("tomat", "pasir"): 35.0, ("tomat", "lempung"): 37.0, ("tomat", "liat"): 38.0,
            ("selada", "pasir"): 41.0, ("selada", "lempung"): 42.0, ("selada", "liat"): 43.0,
        })
        ambang_tanaman = ambang_kritis_map[(tanaman, jenis_tanah)]
        st.info(
            f"💡 **Referensi Agronomi:**  \n"
            f"Tanaman **{tanaman.capitalize()}** pada tanah **{jenis_tanah}** memiliki ambang kritis "
            f"kelembapan tanah sekitar **`{ambang_tanaman:.0f}%`**."
        )

    # Hitung 28 fitur kausal
    kt_min3 = min(kelembaban_tanah_pct, kt_1jam_lalu, kt_2jam_lalu)
    kt_mean3 = (kelembaban_tanah_pct + kt_1jam_lalu + kt_2jam_lalu) / 3.0
    kt_diff1 = kelembaban_tanah_pct - kt_1jam_lalu
    kt_diff2 = kt_1jam_lalu - kt_2jam_lalu
    evap_index = suhu_udara_C * (100.0 - kelembaban_udara_pct) / 100.0
    total_air_6jam = curah_hujan_mm + prakiraan_hujan_6jam_mm
    defisit_saat_ini = ambang_tanaman - kelembaban_tanah_pct
    defisit_min3h = ambang_tanaman - kt_min3
    lonjakan_siram = int((kt_diff1 > 10.0) and (curah_hujan_mm < 0.5))

    tanaman_dict = bundel.get("tanaman_dict", {"cabai": 0, "tomat": 1, "selada": 2})
    tanah_dict = bundel.get("tanah_dict", {"pasir": 0, "lempung": 1, "liat": 2})

    fitur_input = {
        "umur_tanaman_hari": umur_tanaman_hari,
        "kelembaban_tanah_pct": kelembaban_tanah_pct,
        "suhu_udara_C": suhu_udara_C,
        "kelembaban_udara_pct": kelembaban_udara_pct,
        "cahaya_lux": cahaya_lux,
        "curah_hujan_mm": curah_hujan_mm,
        "prakiraan_hujan_6jam_mm": prakiraan_hujan_6jam_mm,
        "tanaman_enc": tanaman_dict[tanaman],
        "jenis_tanah_enc": tanah_dict[jenis_tanah],
        "petak_enc": int(petak.replace("P", "")),
        "kt_lag1": kt_1jam_lalu,
        "kt_lag2": kt_2jam_lalu,
        "kt_lag3": kt_2jam_lalu,
        "kt_diff1": kt_diff1,
        "kt_diff2": kt_diff2,
        "kt_min_3h": kt_min3,
        "kt_mean_3h": kt_mean3,
        "hujan_lag1": curah_hujan_mm,
        "prakiraan_lag1": prakiraan_hujan_6jam_mm,
        "jam": jam,
        "jam_sin": np.sin(2.0 * np.pi * jam / 24.0),
        "jam_cos": np.cos(2.0 * np.pi * jam / 24.0),
        "evap_index": evap_index,
        "total_air_6jam": total_air_6jam,
        "ambang_estimasi": ambang_tanaman,
        "defisit_kt_saat_ini": defisit_saat_ini,
        "defisit_kt_min3h": defisit_min3h,
        "lonjakan_siram_tanpa_hujan": lonjakan_siram,
    }

    df_row = pd.DataFrame([fitur_input])[bundel["features"]].astype(np.float32)
    prob_rf = float(bundel["model_rf"].predict_proba(df_row)[0, 1])
    prob_hgb = float(bundel["model_hgb"].predict_proba(df_row)[0, 1])

    if mode_model.startswith("Random Forest"):
        prob_ml = prob_rf
        threshold_dipakai = float(bundel["threshold_rf"])
    else:
        prob_ml = 0.5 * prob_rf + 0.5 * prob_hgb
        threshold_dipakai = float(bundel["threshold_ensemble"])

    keputusan_ml = int(prob_ml >= threshold_dipakai)
    keputusan_baseline = int(kelembaban_tanah_pct < 35.0)

    st.divider()
    st.subheader(f"🔍 Hasil Keputusan Irigasi — Petak `{petak}` ({tanaman.capitalize()} · Tanah {jenis_tanah.capitalize()})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kelembapan Tanah", f"{kelembaban_tanah_pct:.1f}%", f"{kt_diff1:+.1f}% vs 1 jam lalu")
    m2.metric("Selisih dari Ambang Tanaman", f"{kelembaban_tanah_pct - ambang_tanaman:+.1f}%", f"Ambang: {ambang_tanaman:.0f}%")
    m3.metric("Prakiraan Hujan 6 Jam", f"{prakiraan_hujan_6jam_mm:.1f} mm", f"Saat ini: {curah_hujan_mm:.1f} mm")
    m4.metric("Skor Keyakinan AI", f"{prob_ml * 100:.1f}%", f"Threshold: {threshold_dipakai * 100:.0f}%")

    col_res1, col_res2 = st.columns([3, 2])

    with col_res1:
        if keputusan_ml == 1:
            st.error("### 🚨 KEPUTUSAN AI: PERLU DISIRAM SEKARANG (POMPA MENYALA)")
            st.markdown(
                f"**Alasan Keputusan Agronomi:**  \n"
                f"1. **Defisit Kelembapan Tanah:** Kelembapan tanah pada petak **{petak}** saat ini (**{kelembaban_tanah_pct:.1f}%**, "
                f"minimum 3 jam terakhir: **{kt_min3:.1f}%**) telah turun di bawah ambang nyaman tanaman **{tanaman}** "
                f"pada tanah **{jenis_tanah}** (~**{ambang_tanaman:.0f}%**).  \n"
                f"2. **Prakiraan Cuaca Aman untuk Menyiram:** Curah hujan saat ini (**{curah_hujan_mm:.1f} mm**) dan prakiraan hujan "
                f"6 jam ke depan (**{prakiraan_hujan_6jam_mm:.1f} mm**) sangat rendah, sehingga penyiraman tidak akan membuang air.  \n"
                f"3. **Mitigasi Risiko Bisnis:** Menyiram sekarang mencegah tanaman **{tanaman}** mengalami stres kekeringan dan layu."
            )
        else:
            st.success("### ✅ KEPUTUSAN AI: TIDAK PERLU DISIRAM (POMPA MATI / DITAHAN)")
            if prakiraan_hujan_6jam_mm >= 1.5 or curah_hujan_mm >= 0.5:
                st.markdown(
                    f"**Alasan Keputusan Agronomi (Penundaan karena Hujan):**  \n"
                    f"1. **Antisipasi Hujan:** Meskipun kelembapan tanah berada di angka **{kelembaban_tanah_pct:.1f}%**, "
                    f"terdeteksi curah hujan/prakiraan hujan 6 jam ke depan sebesar **{prakiraan_hujan_6jam_mm:.1f} mm** "
                    f"(total suplai air hujan: **{total_air_6jam:.1f} mm**).  \n"
                    f"2. **Penghematan Air & Pencegahan Busuk Akar:** Sesuai instruksi klien, menyiram sesaat sebelum hujan turun "
                    f"akan membuang-buang air dan membuat tanah terlalu jenuh sehingga akar **{tanaman}** berisiko membusuk."
                )
            else:
                st.markdown(
                    f"**Alasan Keputusan Agronomi (Kelembapan Mencukupi):**  \n"
                    f"1. **Cadangan Air Tanah Cukup:** Kelembapan tanah saat ini (**{kelembaban_tanah_pct:.1f}%**) masih mencukupi "
                    f"kebutuhan tanaman **{tanaman}** pada tanah **{jenis_tanah}** (ambang kritis ~**{ambang_tanaman:.0f}%**).  \n"
                    f"2. **Efisiensi Operasional:** Menahan pompa tetap mati menghemat air serta menjaga aerasi akar tetap sehat."
                )

    with col_res2:
        st.markdown("#### ⚖️ Perbandingan terhadap Baseline Wajib (`< 35%`)")
        label_base = "🚨 SIRAM (1)" if keputusan_baseline == 1 else "✅ JANGAN SIRAM (0)"
        label_ai = "🚨 SIRAM (1)" if keputusan_ml == 1 else "✅ JANGAN SIRAM (0)"

        st.table(pd.DataFrame([
            {"Metode": "Aturan Sederhana (Kelembapan < 35%)", "Keputusan": label_base},
            {"Metode": "Model Machine Learning (IoT Pintar)", "Keputusan": label_ai},
        ]))

        if keputusan_ml != keputusan_baseline:
            if keputusan_baseline == 1 and keputusan_ml == 0:
                st.warning(
                    "🔍 **Keunggulan AI atas Baseline:**  \n"
                    "Aturan `< 35%` menyalakan pompa secara kaku hanya karena kelembapan di bawah 35%, "
                    "padahal **akan turun hujan** atau tanah masih aman bagi tanaman. AI mencegah pemborosan air!"
                )
            else:
                st.warning(
                    "🔍 **Keunggulan AI atas Baseline:**  \n"
                    f"Aturan `< 35%` gagal mendeteksi bahwa tanaman **{tanaman}** memiliki ambang kritis lebih tinggi "
                    f"(~**{ambang_tanaman:.0f}%**). AI menyalakan penyiraman lebih tepat waktu sebelum tanaman stres!"
                )
        else:
            st.caption("Pada kondisi sensor ini, keputusan Model Machine Learning dan aturan sederhana (< 35%) sejalan.")

# ==============================================================================
# TAB 2: MONITOR SERENTAK 16 PETAK RUMAH KACA (P01 - P16)
# ==============================================================================
with tab2:
    st.subheader("🏡 Pantauan Serentak 16 Petak Rumah Kaca (`P01` – `P16`)")
    if df_sample is None or "kt_lag1" not in df_sample.columns:
        st.info("Data sampel 16 petak tidak tersedia.")
    else:
        daftar_waktu = sorted(df_sample["waktu"].unique(), reverse=True)
        waktu_pilih = st.selectbox("Pilih Jam Rekaman Sensor Historis (Data Uji Oktober 2025):", daftar_waktu, index=0)

        sub_df = df_sample[df_sample["waktu"] == waktu_pilih].sort_values("petak").reset_index(drop=True)
        X_sub = sub_df[bundel["features"]].astype(np.float32)

        p_rf_sub = bundel["model_rf"].predict_proba(X_sub)[:, 1]
        p_hgb_sub = bundel["model_hgb"].predict_proba(X_sub)[:, 1]
        if mode_model.startswith("Random Forest"):
            prob_sub = p_rf_sub
            th_sub = float(bundel["threshold_rf"])
        else:
            prob_sub = 0.5 * p_rf_sub + 0.5 * p_hgb_sub
            th_sub = float(bundel["threshold_ensemble"])

        sub_df["Probabilitas AI (%)"] = (prob_sub * 100).round(1)
        sub_df["Prediksi AI"] = (prob_sub >= th_sub).astype(int)
        sub_df["Baseline (<35%)"] = (sub_df["kelembaban_tanah_pct"] < 35.0).astype(int)

        jml_siram_ai = int(sub_df["Prediksi AI"].sum())
        jml_siram_asli = int(sub_df["perlu_siram"].sum())
        curah_jam_ini = float(sub_df["curah_hujan_mm"].iloc[0])
        prakiraan_6j = float(sub_df["prakiraan_hujan_6jam_mm"].iloc[0])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Waktu Pantauan", waktu_pilih)
        c2.metric("Petak Perlu Disiram (AI)", f"{jml_siram_ai} / 16 Petak")
        c3.metric("Keputusan Asli Agronom", f"{jml_siram_asli} / 16 Petak")
        c4.metric("Prakiraan Hujan 6 Jam", f"{prakiraan_6j:.1f} mm", f"Hujan Jam Ini: {curah_jam_ini:.1f} mm")

        st.markdown("#### Peta Status 16 Petak Rumah Kaca")
        for row_idx in range(4):
            cols_grid = st.columns(4)
            for col_idx in range(4):
                idx = row_idx * 4 + col_idx
                if idx < len(sub_df):
                    r = sub_df.iloc[idx]
                    status_ikon = "🚨 SIRAM" if r["Prediksi AI"] == 1 else "✅ AMAN"
                    with cols_grid[col_idx]:
                        st.markdown(
                            f"**{r['petak']} · {r['tanaman'].capitalize()} ({r['jenis_tanah']})**  \n"
                            f"Status AI: **{status_ikon}** (`{r['Probabilitas AI (%)']:.1f}%`)  \n"
                            f"Kelembapan: `{r['kelembaban_tanah_pct']:.1f}%` (Ambang: `{r['ambang_estimasi']:.0f}%`)  \n"
                            f"Label Agronom: `{'Siram (1)' if r['perlu_siram']==1 else 'Tidak (0)'}`"
                        )

        st.markdown("#### Tabel Detail Pembacaan Sensor 16 Petak")
        tampil_cols = [
            "petak", "tanaman", "jenis_tanah", "umur_tanaman_hari",
            "kelembaban_tanah_pct", "kt_lag1", "suhu_udara_C",
            "prakiraan_hujan_6jam_mm", "Probabilitas AI (%)", "Prediksi AI", "Baseline (<35%)", "perlu_siram"
        ]
        st.dataframe(sub_df[tampil_cols], use_container_width=True)

# ==============================================================================
# TAB 3: EVALUASI MODEL, BASELINE & SIMULASI BIAYA BISNIS
# ==============================================================================
with tab3:
    st.subheader("📊 Evaluasi Performa Model pada Test Set Resmi (15–31 Oktober 2025)")
    mt = bundel["metrics_test"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("F1-Score Random Forest", f"{mt['f1_ml_rf']:.4f}", f"+{mt['f1_ml_rf'] - mt['f1_baseline']:.4f} vs Baseline")
    k2.metric("Precision Random Forest", f"{mt['prec_ml_rf']:.4f}", "Minim Pemborosan Air")
    k3.metric("Recall Random Forest", f"{mt['rec_ml_rf']:.4f}", "Minim Tanaman Stres")
    k4.metric("F1-Score Baseline (< 35%)", f"{mt['f1_baseline']:.4f}", "Aturan Tanpa ML")

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.markdown("#### Perbandingan F1-Score per Jenis Tanaman (Test Set)")
        df_crop = pd.DataFrame(mt["crop_metrics"])
        st.table(df_crop)
        st.caption(
            "Perhatikan pada tanaman **Selada**: aturan Baseline (`< 35%`) memperoleh **F1 = 0.0000** karena "
            "selada membutuhkan penyiraman pada kelembapan ~41–43%. Model Machine Learning berhasil mencapai **F1 > 0.80**."
        )

    with col_ev2:
        st.markdown("#### Top 10 Feature Importance (Random Forest)")
        df_imp = pd.DataFrame(mt["top_features"]).head(10)
        fig_imp, ax_imp = plt.subplots(figsize=(6, 3.8))
        sns.barplot(data=df_imp, x="Importance", y="Fitur", color="#2A9D8F", ax=ax_imp)
        ax_imp.set_title("Fitur Penentu Keputusan Irigasi Pintar")
        plt.tight_layout()
        st.pyplot(fig_imp)

    st.divider()
    st.subheader("💰 Kalkulator Pertimbangan Biaya Operasional & Agronomi (Cost Trade-off)")
    st.markdown(
        "Sesuai *brief*: **Terlambat menyiram (*False Negative*)** membuat tanaman stres kekeringan, "
        "sedangkan **menyiram tanpa perlu (*False Positive*)** membuang air/listrik dan dapat membuat akar membusuk."
    )

    cb1, cb2 = st.columns(2)
    with cb1:
        biaya_fp = st.number_input(
            "Estimasi Biaya False Positive (Boros Air & Risiko Busuk Akar per Kejadian - Rp):",
            min_value=1000, max_value=500000, value=15000, step=1000
        )
    with cb2:
        biaya_fn = st.number_input(
            "Estimasi Biaya False Negative (Tanaman Stres Kekeringan per Kejadian - Rp):",
            min_value=1000, max_value=500000, value=45000, step=1000
        )

    cm_b = mt["cm_baseline"]
    cm_r = mt["cm_rf"]
    cm_e = mt["cm_ensemble"]

    fp_b, fn_b = cm_b[0][1], cm_b[1][0]
    fp_r, fn_r = cm_r[0][1], cm_r[1][0]
    fp_e, fn_e = cm_e[0][1], cm_e[1][0]

    rug_b = fp_b * biaya_fp + fn_b * biaya_fn
    rug_r = fp_r * biaya_fp + fn_r * biaya_fn
    rug_e = fp_e * biaya_fp + fn_e * biaya_fn

    df_cost = pd.DataFrame([
        {
            "Metode / Model": "1. Baseline Wajib (kelembaban_tanah < 35%)",
            "False Positive (Boros Air / Busuk Akar)": fp_b,
            "False Negative (Tanaman Stres)": fn_b,
            "Total Estimasi Kerugian (17 Hari Uji)": f"Rp {rug_b:,.0f}",
            "Penghematan vs Baseline": "Rp 0 (Patokan Dasar)"
        },
        {
            "Metode / Model": "2. Ensemble RF + HistGB",
            "False Positive (Boros Air / Busuk Akar)": fp_e,
            "False Negative (Tanaman Stres)": fn_e,
            "Total Estimasi Kerugian (17 Hari Uji)": f"Rp {rug_e:,.0f}",
            "Penghematan vs Baseline": f"Rp {rug_b - rug_e:,.0f}"
        },
        {
            "Metode / Model": "3. Random Forest Teroptimasi (Model Terbaik)",
            "False Positive (Boros Air / Busuk Akar)": fp_r,
            "False Negative (Tanaman Stres)": fn_r,
            "Total Estimasi Kerugian (17 Hari Uji)": f"Rp {rug_r:,.0f}",
            "Penghematan vs Baseline": f"Rp {rug_b - rug_r:,.0f}"
        }
    ])
    st.table(df_cost)

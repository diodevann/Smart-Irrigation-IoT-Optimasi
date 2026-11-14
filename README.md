# 🌱 Smart Irrigation IoT — Streamlit App (Siap Pakai di VSCode)

Folder proyek ini sudah berisi **seluruh file lengkap** termasuk model Machine Learning yang telah dilatih (`model_irigasi_pintar_streamlit.pkl` dengan **F1-Score Test = 0.8391** vs **Baseline = 0.1575**), data sampel 16 petak rumah kaca, kode antarmuka `app.py`, serta skrip otomatisasi `venv`.

---

## 📁 Struktur Folder Proyek

```text
smart_irrigation_vscode/
├── .streamlit/
│   └── config.toml                                 # Tema warna agronomi untuk Streamlit
├── .vscode/
│   ├── launch.json                                 # Konfigurasi Run/Debug F5 di VSCode
│   ├── settings.json                               # Pengaturan interpreter Python otomatis
│   └── tasks.json                                  # Task otomatis di VSCode
├── data/
│   └── sample_greenhouse_16petak.csv               # Data uji 16 petak (P01-P16) untuk Tab Monitor
├── notebooks/
│   └── Kasus_07_Smart_Irrigation_IoT_Optimasi_Lengkap.ipynb  # Notebook Google Colab lengkap
├── app.py                                          # Aplikasi utama Streamlit (3 Tab Interaktif)
├── model_irigasi_pintar_streamlit.pkl              # Bundel model ML siap pakai (sudah terlatih)
├── requirements.txt                                # Daftar library Python untuk lokal & cloud
├── run_windows.bat                                 # Klik 2x (atau jalankan di terminal) untuk Windows
├── run_mac_linux.sh                                # Launcher otomatis untuk macOS / Linux
└── README.md                                       # Panduan penggunaan
```

---

## 🚀 Cara Langsung Menjalankan di VSCode (Hanya 1 Langkah!)

### Opsi A — Cara Paling Cepat (Otomatis Buat `venv` + Install + Run)
1. Ekstrak file `.zip` ini, lalu buka folder **`smart_irrigation_vscode`** di **VSCode** (`File -> Open Folder`).
2. Buka Terminal di VSCode (`Ctrl + ~`), lalu ketik perintah berikut:
   - **Windows:**
     ```cmd
     .\run_windows.bat
     ```
   - **macOS / Linux:**
     ```bash
     bash run_mac_linux.sh
     ```
   *(Skrip ini akan otomatis membuat folder `venv`, menginstal isi `requirements.txt`, dan membuka website Streamlit di browser Anda!)*

### Opsi B — Cara Manual di Terminal VSCode
```bash
# 1. Buat virtual environment
python -m venv venv

# 2. Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install library
pip install -r requirements.txt

# 4. Jalankan aplikasi Streamlit
streamlit run app.py
```

---

## 🌐 Cara Deploy ke Internet Publik (Streamlit Community Cloud)
1. Buat repository baru di **GitHub** Anda.
2. Upload isi folder ini (`app.py`, `model_irigasi_pintar_streamlit.pkl`, `requirements.txt`, folder `.streamlit/`, dan folder `data/`) ke repository GitHub tersebut *(catatan: folder `venv/` tidak perlu ikut di-upload)*.
3. Buka [share.streamlit.io](https://share.streamlit.io/), klik **New app**, pilih repository GitHub Anda, pastikan Main file path diisi **`app.py`**, lalu klik **Deploy!**

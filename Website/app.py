# WEBSITE REKOMENDASI MENU MAKANAN MINGGUAN DENGAN OPTIMASI

# Algoritma yang dibuat akan menyusun menu makan mingguan menggunakan optimasi dengan 5 objektif menggunakan solver C-TAEA
# 5 objektif yang digunakan mencakup :
#   Makronutrisi : kalori, protein, lemak, karbohidrat, dan serat
# Menu makan mingguan yang dihasilkan mencakup data
#   1. Menu makan per waktu makan  : sarapan, snack pagi, makan siang, snack sore, dan makan malam
#   2. Data selisih dan persentasi selisih nutrisi menu makan harian terhadap AKG

# Algoritma menerima input berupa
#   1. Dataset AKG
#   2. Dataset makanan
#   3. Input tahun standar AKG
#   4. Input usia anak balita user
#   5. Input alergi makanan anak user

# Algoritma menghasilkan output berupa
#   menu makanan mingguan

# Aloritma terhubung dengan website melalui Flask
# Terdapat 5 halaman pada website
#   1. Halaman Utama (landing.html)
#   2. Halaman Input Data (index.html)
#   3. Halaman Target Gizi Harian (akg.html) -> termasuk Halaman Penyusunan Menu Makanan yang ditampilkan ketika menu mingguan sedang dibuat
#   4. Halaman Output Menu Makanan (result.html)
#   5. Halaman Error (error.html)

# Menu makanan mingguan yang dihasilkan oleh algoritma akan ditampilkan pada Halaman Output Menu Makanan

# Algoritma :

# Import library yang akan digunakan
from flask import Flask, render_template, request   # flask untuk menghubungkan algoritma dengan website
import numpy as np  # Library untuk fungsi matematika
import pandas as pd # Library untuk mengolah dataset
import random, secrets  # Library untuk membuat nilai random
from collections import Counter # Import library untuk menghitung

# Import library yang akan digunakan untuk Optimasi MaOO
# Library yang akan digunakna adalah pymoo
# Solver yang digunakan adalah C-TAEA
from pymoo.algorithms.moo.ctaea import CTAEA    # Import solver C-TAEA
from pymoo.optimize import minimize # Import minimize untuk optimasi dengan fungsi untuk mencari nilai terkecil
from pymoo.util.ref_dirs import get_reference_directions    # Import reference direction untuk optimasi
from pymoo.core.problem import ElementwiseProblem   # Import Element Wise Problem untuk mendefinisikan penyusunan menu makanan sebagai masalah optimisasi
from pymoo.operators.crossover.sbx import SBX   # Import crossover yang digunakan dalam optimasi
from pymoo.operators.mutation.pm import PM  # Import mutasi yang digunakan dalam optimasi

# Deklarasi app
app = Flask(__name__)

# Membaca Dataset yang digunakan
# Digunakan dua dataset pada optimasi
#   1. Dataset AKG yang akan menyimpan data AKG berdasarkan usia anak dan tahun standar AKG yang digunakan
#   2. Dataset makanan yang menyimpan data makanan dan informasi-informasi seperti kandungan nutrisi, porsi, dan tipe makanan
try:
    data_AKG = pd.read_excel("AKG.xlsx")    # Membaca dataset AKG target
    data_makanan = pd.read_excel("Dataset_Makanan.xlsx")    # Membaca dataset makanan
    
    # Membersihkan nama kolom (Hapus spasi di awal/akhir nama kolom)
    data_AKG.columns = data_AKG.columns.str.strip()
    data_makanan.columns = data_makanan.columns.str.strip()
    
    # Jika berhasil maka, print
    print("✅ Dataset berhasil dimuat!")

# Jika dataset tidak ditemukan maka, beri peringatan
except FileNotFoundError:
    print("❌ Error: Pastikan file 'AKG.xlsx' dan 'Dataset_Makanan_baru.xlsx' ada di folder yang sama!")

# Mendeklarasikan fungsi-fungsi yang akan digunakan dalam kode utama

# Membuat fungsi format porsi makanan untuk menampilkan porsi tiap makanan pada menu
def format_nama_urt(row):
    # Menggabungkan Nama Makanan dengan Porsi (URT) dan Berat (Gram).
    # Output: "Nasi Putih (1 Piring, 100 gram)"

    # Ambil nama makanan
    nama = row['Nama Makanan']
    
    # Ambil data URT
    urt_text = ""
    try:
        urt_val = row.get('URT_nominal')    # Ambil data nominal URT
        urt_unit = row.get('URT_ukuran')    # Ambil data ukuran URT yang digunakan
        # Gabungkan menjadi satu besaran URT
        # Contoh : 1 piring
        if pd.notna(urt_val) and pd.notna(urt_unit):
            urt_text = f"{urt_val} {urt_unit}"       
    except:
        pass

    # Ambil data berat
    berat_text = ""
    try:
        berat_val = row.get('gram')  # Ambil data berat makanan 
        # Tambahkan kata gram dibelakang
        # Contoh : 100 gram
        if pd.notna(berat_val):
            berat_text = f"{round(berat_val, 2)} gram"
    except:
        pass

    # Gabungkan besar URT dengan berat makanan
    besaran_list = []    # deklarasi besaran list untuk menyimpan data URT dan berat
    if urt_text: 
        besaran_list.append(urt_text)   # tambahkan data URT makanan
    if berat_text: 
        besaran_list.append(berat_text)  #tambahkan berat makanan
    
    # Menmbentuk kalimat yang menyatakan nama makanan dan besarannya
    if besaran_list:
        # Menggabungkan URT dengan berat makanan, dan memisahkannya dengan "koma"
        gabungan = ", ".join(besaran_list) # Hasil: "1 Piring, 100 gram"
        # Membentuk kalimat dengan nama makanan yang diikuti dengan besarannya
        return f"{nama} ({gabungan})"   # Hasil : "Nasi Putih (1 Piring, 100 gram)"
    else:
        return nama # Jika tidak ada besaran, maka kembalikan nama makanan saja
    
# Membuat fungsi untuk mencari nilai AKG target
# Pencarian AKG target terdiri ata dua tahap
#   1. Mencari AKG targer berdasarkan usia, terdapat 5 pilihan yaitu 1, 2, 3, 4, dan 5 tahun
#   2. Mencari AKG target berdasarkan tahun standar yang digunakan, terdapat 2 pilihan yaitu tahun 2014 dan 2019
# Membuat fungsi untuk mencari nilai AKG target berdasarkan tahun
# Fungsi akan mencari nilai target AKG sesuai dengan tahun standar yang diinput oleh user
def cari_Tahun_AKG(tahun_AKG, df_AKG):
    row_data = df_AKG[df_AKG["Tahun"] == tahun_AKG] # Mencari tahun pada kolom "tahun" di dataset yang sama dengan tahun standar yang diinput
    if row_data.empty:
        return None # Jika tidak ditemukan, maka mengembalikan none
    # Jika usia ditemukan, maka akan dikembalikan data AKG yang sesuai dengan usia input
    return row_data.drop(columns=["Tahun"]).reset_index(drop=True) # AKG dengan tahun yang tidak sesuai akan didrop

# Membuat fungsi untuk mencari nilai AKG target berdasarkan usia
# Fungsi akan mencari nilai target AKG sesuai dengan usia yang diinput oleh user
def cari_Target_AKG(umur_user, df_AKG):
    row_data = df_AKG[df_AKG["umur"] == umur_user] # Mencari usia pada kolom "umur" di dataset yang sama dengan usia user
    if row_data.empty:
        return None # Jika tidak ditemukan, maka mengembalikan none
    # jika usia ditemukan, maka akan dikembalikan data AKG yang sesuai dengan usia input
    return row_data.drop(columns=["umur"]).reset_index(drop=True)   # AKG untuk usia yang tidak sesuai akan didrop

# Beberapa fungsi tambahan yang akan digunakan untuk menyusun menu mingguan
# Membuat fungsi untuk reindex dataframe menjadi 6 index
def pad_df_to_six(df):
    return df.reindex(range(6))

# Membuat fungsi untuk drop index yang kosong
def drop_empty(x):
    return [i for i in x if pd.notna(i)]

# Mendefinisikan pencarian rekomendasi makanan menjadi sebuah masalah optimasi 
# Optimasi dilakukan dengan memilih makanan-makanan dan mengkombinasikannya ke dalam suatu menu harian sebagai calon solusi
# Optimasi akan memilih solusi berdasarkan calon solusi dnegan niali selisih nutrisi terhadap target AKG terkecil
class Meal_Planning(ElementwiseProblem):
    # Input dari fungsi adalah target AKG, jumlah makanan untuk 1 hari,dan jumlah maksimal snack
    def __init__(self, Target_AKG_MaOO, jumlah_makanan, n5, umur, dm):
        self.akg = Target_AKG_MaOO.reset_index(drop=True)   # Mendeklarasi Target AKG
        self.n5 = n5 # Mendeklarasi jumlah maksimal snack
        self.data_makanan = dm # Mendeklarasi data makanan
        
        # Mendeklarasi variabel optimasi
        super().__init__(n_var=jumlah_makanan,  # Mendeklarasi jumlah variabel per 1 solusi sebagai jumlah makanan untuk 1 hari
                         n_obj=5,   # Mendeklarasi objektif optimasi sebagai Target AKG 
                         n_ieq_constr=1,    # Mendeklarasi jumlah constraint 
                         xl=0,  # Mendeklarasi index pertama untuk calon solusi
                         xu=len(data_makanan) - 1,  # Mendeklarasi index terakhir untuk calon solusi
                         vtype=int) # Mendeklarasi tipe variabel untuk calon solusi yaitu integer

    def _evaluate(self, x, out, *args, **kwargs):
        # Mendeklarasi calon solusi
        # Calon solusi dipilih dari dataset makanan, dan dipanggil sebagai indeks makanan terpilih pada dataset makanan
        x = [int(i) for i in x] 
        pilih = data_makanan.iloc[x]

        # Menghilangkan kolom pada dataset diluar kolom-kolom yang menyimpan data nutrisi
        # Kolom yang didrop antara lain adalah kolom "No", "Kode", "Nama Makanan", "Jenis", "Tipe", "gram", "URT_nominal", dan "URT_ukuran"
        kolom_nutrisi = ["Kalori (kkal)", "Protein (g)", "Lemak (g)", "Karbohidrat (g)", "Serat (g)"]
        # Mendeklarasi variabel total untuk menotalkan nutrisi makanan-makanan yang terdapat dalam satu calon solusi 
        total = pilih[kolom_nutrisi].sum()
        # Mendeklarasi target sebagai AKG target
        Target = self.akg.iloc[0]

        # Memberikan pembobotan pada objektif
        # Pembobotan dilakukan agak optimasi menekankan pencarian selisih terkecil pada nutrisi-nutrisi tertentu
        # Nutrisi portein memiliki pembobotan tertinggi mengingat kecenderungan hasil solusi yang terlalu banyak memberikan protein
        nutrient_weights = {
            "Kalori (kkal)": 2.0,
            "Protein (g)":5.0,
            "Lemak (g)": 3.0,
            "Karbohidrat (g)": 2.0,
            "Serat (g)": 2.0,
        }

        # Mendeklarasi variabel yang menyimpan nilai objektif calon solusi
        obj = []
        
        # Menghitung nilai selisih nutrisi dari calon solusi dengan target AKG
        for i in self.akg.columns:
            nilai_total = total[i]
            nilai_target = Target[i]
            
            # Menghitung selisih nutrisi di calon solusi dengan target
            persen_selisih = (abs(nilai_total - nilai_target) / nilai_target)*100
            weight = nutrient_weights.get(i, 1.0)   # Memberikan pembobotan pada nilai objektif
            obj.append(persen_selisih * weight) # Menyimpan nilai objektif            

        # Output nilai objektif
        out["F"] = obj 
        
        # Mendeklarasi constrain
        # Contraint dibuat untuk memastikan bahwa solusi yang dikeluarkan memiliki
        #   1. Setidaknya 2 makanan pokok
        #   2. Setidaknya 1 makanan lauk-pauk, sayur, dan buah
        #   3. Jumlah snack tidka melebihi ambang batas
        # Hal ini dilakukan untuk memastikan bahwa solusi menu makanan berisikan semua tipe-tipe makanan yang seimbang
        kategori_semua = pilih["Jenis"].tolist()    # Pengecekan tipe makanan dilihat melalui kolom "Jenis"
        jumlah_pokok = kategori_semua.count("Makanan Pokok")    # Menghitung jumlah makanan pokok dalam calon solusi
        jumlah_lauk = kategori_semua.count("Lauk-pauk") # Menghitung jumlah laup-pauk dalam calon solusi
        jumlah_sayur = kategori_semua.count("Sayur-mayur")  # Menghitung jumlah sayur-mayur dalam calon solusi
        jumlah_buah = kategori_semua.count("Buah")  # Menghitung buah dalam calon solusi
        jumlah_snack = kategori_semua.count("Snack")    # Menghitung jumlah snack dalam calon solusi

        # Menghitung constraint
        # Jika calon solusi tidak memenuhi jumlah batas tiap tipe makanan, maka nilai constraint violation ditambah 1
        c1 = 0  # Mendeklarasi variabel yang menyimpan nilai constraint violation
        if jumlah_pokok < 2 : c1 += 1   # Mengecek jumlah makanan pokok dalam calon solusi
        if jumlah_lauk < 1 : c1 += 1    # Mengecek jumlah lauk-pauk dalam calon solusi
        if jumlah_sayur < 1 : c1 += 1   # Mengecek jumlah sayur-mayur dalam calon solusi
        if jumlah_buah < 1 : c1 += 1    # Mengecek jumlah buah dalam calon solusi
        if jumlah_snack > self.n5 : c1 += 1 # Mengecek jumlah snack dalam calon solusi
        
        # Output nilai constraint violation
        out["G"] = [c1]


# Fungsi untuk menyusuk menu makanan mingguan
def generate_menu_logic(input_umur, input_tahun, input_alergi_str):
    # Mengkopi data makanan ke lokal
    local_data_makanan = data_makanan.copy()
    
    # Mencari AKG yang sesuai dengan tahun standar input
    Tahun_AKG = cari_Tahun_AKG(input_tahun, data_AKG)
    # Mencari AKG yang sesuai dengan usia input
    Target_AKG = cari_Target_AKG(input_umur, Tahun_AKG)
    # Memilih hanya 5 target AKG sebagai objektif (kalori, protein, lemak, karbohidrat, dan serat)
    Target_AKG_obj = Target_AKG[["Kalori (kkal)", "Protein (g)", "Lemak (g)", "Karbohidrat (g)", "Serat (g)"]].reset_index(drop=True)

    # Menghilangkan data makanan alergi user data dataset makanan
    if input_alergi_str and input_alergi_str.lower() != 'none':
        alergi_list = [a.strip().lower() for a in input_alergi_str.split(",")] # Membuat input data makanan menjadi list

        # Memfilter makanan alergi dari dataset
        # Mencari data makanan alergi di dataset makanan
        # Pencarian makanan dilakukan berdasarkan kolom "Nama makanan" dan "Tipe"

        # Mendeklarasi kondisi nama dan kondisi tipe yang mendefinisikan alergi makanan
        kondisi_nama = local_data_makanan["Nama Makanan"].str.lower().str.contains("|".join(alergi_list))
        kondisi_tipe = local_data_makanan["Tipe"].str.lower().str.contains("|".join(alergi_list))

        # Menyimpan kondisi alergi makanan jika ia terdapat di kolom "Nama Makanan" atau "Tipe" sebagai kondisi gabungan
        kondisi_gabungan = kondisi_nama | kondisi_tipe

        # Drop alergi makanan dari dataset berdasarkan kondisi gabungan
        local_data_makanan = local_data_makanan[~kondisi_gabungan].reset_index(drop=True)

    # Anak dibawah 1 tahun gak boleh makan kacang karena ukurannya yang kecil dapat mengakibatkan anak tersedak
    # Jika anak di bawah 1 tahun, maka makanan kacang akan didrop dari dataset
    # Makanan kacang terdapat pada dataset sebagai jenis makanan "snack" dengan tipe makanan "kacang"
    if input_umur == 1:
        # Mencari kacang pada dataset makanan

        # Mendeklarasi kondisi suatu makanan sebagai mendefinisikan kacang
        kondisi_snack_kacang = (local_data_makanan["Jenis"].str.lower() == "snack") & \
                                (local_data_makanan["Tipe"].str.lower() == "kacang")

        # Menghapus kacang dari dataset makanan
        local_data_makanan = local_data_makanan[~kondisi_snack_kacang].reset_index(drop=True)

    # Susu akan selalu hadir pada menu makanan harian sesuai dengan rekomendari Pedoman Makanan bergizi
    # Kecuali anak alergi terhadap susu
    # Jika tidak alergi, maka nilai nutrisi susu akan dikurangi dari nilai target AKG
    kolom_nutrisi = [kol for kol in Target_AKG.columns if kol in local_data_makanan.columns]  # Memilih kolom nutrisi
    susu_df = local_data_makanan[local_data_makanan["Jenis"] == "Susu"] # Memastikan susu ada dalam dataset (tidak alergi)
    ada_susu = False  # Mendeklarasi keadaan ada susu atau tidak

    # Jika ada susu (tidak alergi), maka
    if not susu_df.empty:
        ada_susu = True # keadaan dideklarasikan menjadi ada susu
        total_nutrisi_susu = susu_df[kolom_nutrisi].sum()   # Memilih kolom nutrisi susu
        Target_nutrisi = Target_AKG.iloc[0][kolom_nutrisi].astype(float) # Memilih kolom nutrisi pada AKG
        # Kurangi nutrisi susu dari target dan clip minimal 0
        Target_AKG.iloc[0][kolom_nutrisi] = (Target_nutrisi - total_nutrisi_susu).clip(lower=0)

    # Jumlah makanan untuk setiap usia, dengan usia termuda memiliki jumlah makanan terkecil dan usia tertua dnegan jumlah terbanyak
    # Hal ini dilakukan mengingat semakin tinggi usia, semakin tinggi juga AKG yang perlu dicapai

    # Berdasarkan standar AKG 2014, berikut adalah jumlah makanan untuk setiap usia
    if input_tahun == 2014 :
        # Menjumlahkan makanan yang dipilih per hari
        if input_umur== 1 :
            jumlah_n = 11
        elif input_umur == 2 or input_umur == 3 :
            jumlah_n = 12
        else  : # input_umur == 4 and input_umur == 5 
            jumlah_n = 14

    # Bedasarkan standar AKG 2019, berikut adalah jumlah makanan untuk setiap usia
    else :
        if input_umur == 4 or input_umur == 5:
            jumlah_n = 15
        else : # input_umur == 1 or input_umur == 2 or input_umur == 3 
            jumlah_n = 14

    # Mendeklarasi jumlah maksimal snack dalam menu harian
    if input_umur == 4 or input_umur == 5:
        n5 = 3
    else : # input_umur == 1 or input_umur == 2 or input_umur == 3 
        n5 = 2

    # Jika anak alergi dengan susu, maka
    #   1. Jumlah makanan akan ditambah 1, mengingat susu tidak akan ditambahkan ke menu makanan
    #   2. Batas maksimal snack pada menu makanan ditambah 1
    if ada_susu == False :
        jumlah_n += 1
        n5 += 1

    # Mendefinisikan algoritma solving optimasi
    # Optimasi dilakukan dengan solver C-TAEA

    # Mendefinisikan reference direction yang digunakna dalam optimasi
    # Reference direction dibangkitkan dengan "das-dennis" dan 5 partisi
    # Jumlah reference direction adalah 126
    ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=5)

    # Algoritma CTAEA
    # Definisikan algoritma C-TAEA
    algorithm = CTAEA(op_size=126,  # besar populasi dibuat sesuai dengan nilai reference directionnnya
                        ref_dirs=ref_dirs, # Mendeklarasi reference direction
                        crossover=SBX(prob=0.9, eta=20), # Mengatur kondisi crossover
                        mutation=PM(prob=0.5, eta=15)) # Mengatur kondisi mutasi
        
    # Melakukan optimasi
    # Optimasi dilakukan dengan mencari nilai objektif terkecil, dengan demikian digunakan minimize
    # Mendeklarasi optimasi
    Hasil = minimize(
            problem = Meal_Planning(Target_AKG_obj, jumlah_n, n5, input_umur, local_data_makanan), # Mendeklarasi masalah optimasi rekomendasi menu makanan sebagai mana yang telah didefinisikan
            algorithm = algorithm, # Mendeklarasi algoritma solver
            termination=('n_gen', 500), # Mendeklarasi bahwa optimasi dihentikan pada iterasi ke 500
            verbose=True,
            copy_algorithm=False
        )
    # Hasil optimasi akan disimpan ke dalam variabel Hasil

    # Menyimpan solusi optimasi
    # Mendeklarasi variabel untuk menyimpan solusi optimasi
    list_solusi = []

    # Mendeklarasi kolom nutrisi
    kolom_nutrisi_lengkap = ["Kalori (kkal)", "Protein (g)", "Lemak (g)", "Karbohidrat (g)", "Serat (g)"]

    # Mendeklarasi target AKG
    target = Target_AKG.iloc[0]
    # Mendeklarasi target AKG sebagai data AKG dengan kolom nutrisi
    target_values = target[kolom_nutrisi].values

    # Memisahkan solusi menu makanan harian optimasi ke dalam 5 waktu makan yang berbeda
    #   1. Sarapan : terdiri atas makanan pokok, lauk-pauk, dan sayur-mayur
    #   2. Snack pagi : terdiri atas snack dan buah
    #   3. Makan siang : terdiri atas makanan pokok, lauk-pauk, dan sayur-mayur
    #   4. Snack sore : terdiri atas snack dan buah
    #   5. Makan malam : terdiri atas makanan pokok, lauk-pauk, dan sayur-mayur
    for idx, solusi in enumerate(Hasil.X):
        # Indeks menyimpan data indeks-indeks makanan solusi optimasi
        indeks = [int(i) for i in solusi] 

        # Memisahkan makanan pada solusi berdasarkan jenis (makanan pokok, lauk-pauk, sayur-mayur, buah, dan snack)
        # Makanan terpilih merupakan dataset dari makanan-makanan yang ada dalam menu harian solusi optimasi
        makanan_terpilih = local_data_makanan.iloc[indeks].reset_index(drop=True)

        # Memisahkan makanan-makanan ke dalam dataset-dataset yang menyimpan makanan solusi optimasi berdasarkan jenisnya
        pokok = makanan_terpilih[makanan_terpilih["Jenis"] == "Makanan Pokok"].reset_index(drop=True) # membuat dataset makanan pokok
        lauk = makanan_terpilih[makanan_terpilih["Jenis"] == "Lauk-pauk"].reset_index(drop=True)    # membuat dataset lauk-pauk
        sayur = makanan_terpilih[makanan_terpilih["Jenis"] == "Sayur-mayur"].reset_index(drop=True) # membuat dataset sayur-mayur
        buah = makanan_terpilih[makanan_terpilih["Jenis"] == "Buah"].reset_index(drop=True) # membuat dataset buah
        snack = makanan_terpilih[makanan_terpilih["Jenis"] == "Snack"].reset_index(drop=True)   # membuat dataset snack

        # Membuat dataset susu yang disimpan dalam susu_item
        # Hal ini dilakukan jika user tidak alergi susu
        if ada_susu == True : # Jika user tidak alergi susu
            index_susu = local_data_makanan.index[local_data_makanan["Jenis"] == "Susu"][-1] # Mengambil susu dari dataset makanan
            susu_item = local_data_makanan.iloc[index_susu]   # Menyimpan susu ke dalam susu_item
        else : # Jika user alergi susu
            susu_item = buah.iloc[-1] # susu_item akan menyimpan buah pada index terakhir
            buah = buah.drop(buah.index[-1]).reset_index(drop=True) # Mendrop buah dari dataset buah

        # Membuat masing-masing dataset menjadi dataset dengan jumlah indeks 6.
        pokok = pad_df_to_six(pokok)
        lauk = pad_df_to_six(lauk)
        sayur = pad_df_to_six(sayur)
        buah = pad_df_to_six(buah)
        snack = pad_df_to_six(snack)

        # Menyusun menu harian
        # Memisahkan makanan-makanan dari solusi optimasi ke dalam 5 waktu makan yang berbeda
        # Untuk setiap makanan pada menu makanan yang terpilih, diterapkan format porsi makanan
        sarapan = [
            format_nama_urt(pokok.iloc[0]), format_nama_urt(pokok.iloc[3]), 
            format_nama_urt(lauk.iloc[2]), format_nama_urt(lauk.iloc[5]), 
            format_nama_urt(sayur.iloc[1]), format_nama_urt(sayur.iloc[4]), 
            format_nama_urt(buah.iloc[4])
        ]
        snack_pagi = [
            format_nama_urt(snack.iloc[0]), format_nama_urt(snack.iloc[2]), 
            format_nama_urt(snack.iloc[4]), format_nama_urt(buah.iloc[0])
        ]
        makan_siang = [
            format_nama_urt(pokok.iloc[1]), format_nama_urt(pokok.iloc[4]), 
            format_nama_urt(lauk.iloc[0]), format_nama_urt(lauk.iloc[3]), 
            format_nama_urt(sayur.iloc[2]), format_nama_urt(sayur.iloc[5]), 
            format_nama_urt(buah.iloc[1])
        ]
        snack_sore = [
            format_nama_urt(susu_item), 
            format_nama_urt(snack.iloc[1]), format_nama_urt(snack.iloc[3]), format_nama_urt(snack.iloc[5]), 
            format_nama_urt(buah.iloc[3]), format_nama_urt(buah.iloc[5])
        ]
        makan_malam = [
            format_nama_urt(pokok.iloc[2]), format_nama_urt(pokok.iloc[5]),  
            format_nama_urt(lauk.iloc[1]), format_nama_urt(lauk.iloc[4]), 
            format_nama_urt(buah.iloc[2]), 
            format_nama_urt(sayur.iloc[0]), format_nama_urt(sayur.iloc[3])
        ]

        # Mendeklarasi data untuk setiap kolom dari dataset
    # deklarasi data pada kolom data menu per waktu makan
        row = {
            "Sarapan": drop_empty(sarapan),
            "Snack Pagi": drop_empty(snack_pagi),
            "Makan Siang": drop_empty(makan_siang),
            "Snack Sore": drop_empty(snack_sore),
            "Makan Malam": drop_empty(makan_malam),
        }

        # Menghitung total nutrisi pada menu harian solusi optimasi
        total_nutrisi = makanan_terpilih[kolom_nutrisi_lengkap].sum() + susu_item[kolom_nutrisi_lengkap]
        
        # Membulatkan data kalori, protein, lemak, karbohidrat, dan serat menjadi 2 angka dibelakang desimal
        row["total_kalori"] = round(total_nutrisi["Kalori (kkal)"], 2)
        row["total_protein"] = round(total_nutrisi["Protein (g)"], 2)
        row["total_lemak"] = round(total_nutrisi["Lemak (g)"], 2)
        row["total_karbo"] = round(total_nutrisi["Karbohidrat (g)"], 2)
        row["total_serat"] = round(total_nutrisi["Serat (g)"], 2)
        # ---------------------------

        # Menghitung selisih kandungan nutrisi pada solusi optimasi dnegan target AKG
        target_values = Target_AKG.iloc[0][kolom_nutrisi_lengkap].values
        total_val = total_nutrisi.values
        bukan_persen = total_val - target_values
        persen = (bukan_persen / target_values) * 100   # Membuat nilai selisih ke dalam persentase
        
        # deklarasi data pada kolom data selisih nutrisi
        for i, nut in enumerate(kolom_nutrisi_lengkap):
            row[f"Selisih % {nut}"] = persen[i]
        
        # Menyimpan data ke dalam baris
        list_solusi.append(row)

    data_solusi = pd.DataFrame(list_solusi) # Mengubah list_solusi menjadi dataframe pandas
    
    # Membentuk solusi menu mingguan

    # Memfilter solusi yang valid
    # Solusi dianggap valid jika
    #   1. Persentase selisih kalori dibawah 20% (lebih banyak atau kurang dari Target AKG)
    #   2. Persentase selisih protein dibawah 40% (lebih banyak atau kurang dari Target AKG)
    #   3. Persentase selisih lemak dibawah 40% (lebih banyak atau kurang dari Target AKG)
    #   4. Persentase selisih Karbohidrat dibawah 40% (lebih banyak atau kurang dari Target AKG)
    #   5. Persentase selisih serat dibawah 50% (lebih banyak atau kurang dari Target AKG)
    menu_valid = data_solusi[(data_solusi["Selisih % Kalori (kkal)"] >= -20) & (data_solusi["Selisih % Kalori (kkal)"] <= 20) & 
                            (data_solusi["Selisih % Protein (g)"] >= -40) & (data_solusi["Selisih % Protein (g)"] <= 40) & 
                            (data_solusi["Selisih % Lemak (g)"] >= -40) & (data_solusi["Selisih % Lemak (g)"] <= 40) & 
                            (data_solusi["Selisih % Karbohidrat (g)"] >= -40) & (data_solusi["Selisih % Karbohidrat (g)"] <= 40) & 
                            (data_solusi["Selisih % Serat (g)"] >= -50) & (data_solusi["Selisih % Serat (g)"] <= 50) 
    ].reset_index(drop=True)

    # Jika solusi menu yang valid kurang dari 7, dan tidak bisa membentuk menu mingguan maka
    # Maka kriteria valid dilonggarkan sebagai persentase selisih kalori dibawah 20% (lebih banyak atau kurang dari Target AKG) saja
    if len(menu_valid) < 7 :
        menu_valid = data_solusi[(data_solusi["Selisih % Kalori (kkal)"] >= -20) & (data_solusi["Selisih % Kalori (kkal)"] <= 20)
        ].reset_index(drop=True)
        print("menu yang valid_1 kurang dari 7")
        # Jika menu yang valid masih kurang dari 7, dan tidak bisa membentuk menu mingguan maka
        # Maka seluruh data solusi dianggap sebagai menu yang valid
        if len(menu_valid) < 7 :
            menu_valid = data_solusi
            print("menu yang valid_2 kurang dari 7")

    # Dalam pembentukan menu mingguan, 7 menu harian dipilih secara acak, kemudian diperiksa
    # Pemeriksaan dilakukan untuk memastikan bahwa dalam menu mingguan tidak terdapat terllau banyak makanan yang berulang
    # Untuk itu dilakukan pemeriksaan berupa
    #   1. Memastikan jumlah makanan pokok tidak lebih dari ambang batas
    #   2. Memastikan jumlah lauk-pauk tidak lebih dari ambang batas 
    #   3. Memastikan jumlah snack tidak lebih dari ambang batas 
    # Berikut adalah ambang batas yang ditetapkan untuk tidak jenis makanan
    if input_tahun == 2014 : # Jika standar yang digunakan adalah standar AKG tahun 2014
        if input_umur == 1 or input_umur == 2 :
            n2 = 5
        else :
            n2 = 6
        if input_umur == 1 or input_umur == 2 :
            n3 = 3
        else :
            n3 = 4

    else : # Jika standar yang digunakan adalah standar AKG tahun 2014
        if input_umur == 1 or input_umur == 2 :
            n2 = 12
        else :
            n2 = 12
        if input_umur == 1 or input_umur == 2 :
            n3 = 4
        else :
            n3 = 4
    # Ambang batas jumlah makanan yang sama untuk tiap jenis makanan
    Batas_maks_jumlah = {
        "Lauk-pauk": n3,    # Ambang batas jumlah lauk-pauk yang sama dalam menu mingguan
        "Snack": n3 + 1,    # Ambang batas jumlah snack yang sama dalam menu mingguan
        "Makanan Pokok" : n2    # Ambang batas jumlah makanan yang sama dalam menu mingguan
    }
    maks_jumlah = Batas_maks_jumlah.copy()

    # Jika tidak ditemukan menu makanan mingguan yang memenuhi starat ambang batas jumlah makanan yang sama, maka
    # Ambang batas makanan dilonggarkan dan ditambah 1
    # Batas maksimum pelonggaran adalah sebanyak 3 kali
    jumlah_longgar = 0
    maks_longgar = 3

    sukses = False

    menu_mingguan = []  # deklarasi variabel yang menyimpan menu mingguan
    
    # Loop Relaksasi: Mencoba menyusun 7 menu unik
    while jumlah_longgar <= maks_longgar :
        coba = 0    # Mengitung jumlah pelonggaran batas ambang
        hitung_lauk = Counter()    # Menghitung jumlah lauk-pauk yang sama
        hitung_snack = Counter()   # Menghitung jumlah snack yang sama
        hitung_pokok = Counter()   # Menghitung jumlah makanan pokok yang sama
        
        # Algoritma akan mencoba mengkombinasikan 7 menu makanan harian untuk membentuk menu makanan mingguan
        # Percobaan kombinasi akan dilakukan sebanyak maksimal 5000 kali
        while len(menu_mingguan) < 7 and coba < 5000:
            coba += 1
            # Memilih menu makanansecara acak
            calon = menu_valid.sample(1).iloc[0]
            
            # Menggabungkan semua menu per waktu makan menjadi menu harian
            makanan_harian = (calon["Sarapan"]
                            + calon["Snack Pagi"]
                            + calon["Makan Siang"]
                            + calon["Snack Sore"]
                            + calon["Makan Malam"])

            # Cek jumlah makanan yang sama dalam satu menu minggaun
            valid = True

            for m_full in makanan_harian:
                # Mengambil nama makanan saja tanpa data URT maupun berat makanan untuk mengecek jumlah
                nama_only = m_full.split('(')[0].strip() if '(' in m_full else m_full
                
                jenis = local_data_makanan.loc[local_data_makanan["Nama Makanan"] == nama_only, "Jenis"].values
                if len(jenis) > 0:
                    j = jenis[0]
                    # Cek counter
                    current_count = 0
                    # Cek jumlah lauk-pauk yang sama
                    if j=="Lauk-pauk": 
                        current_count = hitung_lauk[nama_only]
                    # Cek jumlah snack yang sama
                    elif j=="Snack": 
                        current_count = hitung_snack[nama_only]
                    # Cek jumlah makanan pokok yang sama
                    elif j=="Makanan Pokok": 
                        current_count = hitung_pokok[nama_only]
                    
                    if j in maks_jumlah and current_count >= maks_jumlah[j]:
                        valid = False; 
                        break
            
             # Kalau belum melanggar makan menu harian akan ditambahkan ke menu mingguan
            if valid:
                menu_mingguan.append(calon)
                for m_full in makanan_harian:
                    nama_only = m_full.split('(')[0].strip() if '(' in m_full else m_full
                    jenis = local_data_makanan.loc[local_data_makanan["Nama Makanan"] == nama_only, "Jenis"].values
                    if len(jenis) > 0:
                        if jenis[0] == "Lauk-pauk": 
                            hitung_lauk[nama_only] += 1    # Menambahkan jumlah lauk-pauk yang sama ke dalam counter lauk-pauk
                        elif jenis[0] == "Snack": 
                            hitung_snack[nama_only] += 1   # Menambahkan jumlah snack yang sama ke dalam counter snack
                        elif jenis[0] == "Makanan Pokok": 
                            hitung_pokok[nama_only] += 1
        
        # Jika berhasil ditemukan 7 menu mingguan, maka pencarian akan berhenti
        if len(menu_mingguan) == 7:
            sukses = True
            break 
        
        # Jika gagal, maka ambang batas makanan akan dilonggarkan
        jumlah_longgar += 1
        if jumlah_longgar >= maks_longgar :
            maks_jumlah["Lauk-pauk"] += 1
            maks_jumlah["Snack"] += 1
            maks_jumlah["Makanan Pokok"] += 1

    # Jika menu mingguan kurang dari 7, maka
    # menu makanan akan ditambahkan secara acak hingga tercapai 7 menu makanan
    if len(menu_mingguan) < 7:
        # Ambil index menu harian yang sudah terpakai
        used_idx = {m["index"] for m in menu_mingguan if "index" in m}

        # Buang menu yang sudah dipakai
        # Simpan sisanya ke dalam dataset pool
        pool = menu_valid.reset_index().loc[~menu_valid.reset_index()["index"].isin(used_idx)]

        #Menambahkan menu makanan sisanya
        if not pool.empty:
            needed = 7 - len(menu_mingguan) # Menghitung jumlah menu makanan yang masih kurang
            tambahan = pool.sample(min(len(pool), needed), replace=False).to_dict('records')    # Mengambil menu makanan tambahan sesuai jumlah yang masih kurang
            menu_mingguan.extend(tambahan)  # Menambahkan menu makanan tambahan
        sukses = True

    # Output menu mingguan
    if sukses == False :
        return None  # Kembalikan None jika gagal mendapatkan 7 hari
    
    # Output menu mingguan
    return menu_mingguan[:7]

    
# Routing Flask

# Membuka Halaman Utama
@app.route('/')
def home():
    return render_template('landing.html')

# Membukan Halaman Input Data
@app.route('/mulai')
def mulai():
    return render_template('index.html')

# Mengolah input user untuk menghitung kebutuhan gizi harian
@app.route('/cek_akg', methods=['POST'])
def cek_akg():
    try:
        # Mengambil data dari form
        umur = int(request.form['umur'])    # Mengambil data usia anak balita user
        tahun = int(request.form['tahun'])  # Mengambil data tahun standar AKG
        alergi = request.form['alergi'] # Mengambil data alergi

        # Mencari data AKG
        Tahun_AKG = cari_Tahun_AKG(tahun, data_AKG) # Berdasarkan tahun
        Target_AKG_row = cari_Target_AKG(umur, Tahun_AKG)   # Berdasarkan usia
        
        # Menyiapkan data AKG yang akan ditampilkan
        # Membulatkannya ke dua angka dibelakang desimal
        data_tampil = {
            "Kalori": round(Target_AKG_row.iloc[0]["Kalori (kkal)"], 2),
            "Protein": round(Target_AKG_row.iloc[0]["Protein (g)"], 2),
            "Lemak": round(Target_AKG_row.iloc[0]["Lemak (g)"], 2),
            "Karbohidrat": round(Target_AKG_row.iloc[0]["Karbohidrat (g)"], 2),
            "Serat": round(Target_AKG_row.iloc[0]["Serat (g)"], 2)
        }
        
        # Menyiapkan input user
        user_input = {'umur': umur, 'tahun': tahun, 'alergi': alergi}

        # Menampilkan Target Gizi Harian
        return render_template('akg.html', akg=data_tampil, user=user_input)
    
    # Jika terjadi error maka akan menampilkan pesan error ke user
    except Exception as e:
        return f"<h3>Terjadi Kesalahan Data:</h3><p>{str(e)}</p><a href='/'>Kembali</a>"

# Menghasilkan menu makan mingguan untuk user
@app.route('/generate', methods=['POST'])
def generate():
    # Mengambil data yang dimasukkan oleh user
    umur = int(request.form['umur'])
    tahun = int(request.form['tahun'])
    alergi = request.form['alergi']
    
    # Menyusun menu makan mingguan
    menu_hasil = generate_menu_logic(umur, tahun, alergi)
    
    # Jika aloritma gagal menghasilkan menu mingguan dan menu_hasil adalah None
    if menu_hasil is None:
        # Menampilkan halaman error
        return render_template('error.html')
    
    # Jika algoritma berhasil menghasilkan menumingguan
    return render_template('result.html', menu=menu_hasil)

# Menjalankan website
if __name__ == '__main__':
    app.run(debug=True)
# REKOMENDASI MENU MAKANAN MINGGUAN DENGAN OPTIMASI 17 OBJEKTIF

# Algoritma yang dibuat akan menyusun menu makan mingguan menggunakan optimasi dengan 17 objektif menggunakan solver C-TAEA
# 17 objektif yang digunakan mencakup :
#   1. Makronutrisi : kalori, protein, lemak, karbohidrat, dan serat
#   2. Mineral : kalsium, fosfor, besi, natrium, kalium, tembaga, dan seng
#   3. Vitamin : vitamin A, vitamin B1, vitamin B2, vitamin B3, dan vitamin C.
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
#   1. File excel yang menyimpan hasil optimasi
#   2. File excel yang menyimpan menu makanan mingguan

# Algoritma : 

# Import library yang akan digunakan
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

# Membaca Dataset yang digunakan
# Digunakan dua dataset pada optimasi
#   1. Dataset AKG yang akan menyimpan data AKG berdasarkan usia anak dan tahun standar AKG yang digunakan
#   2. Dataset makanan yang menyimpan data makanan dan informasi-informasi seperti kandungan nutrisi, porsi, dan tipe makanan
data_AKG = pd.read_excel("AKG.xlsx")    # Membaca dataset AKG target
data_makanan = pd.read_excel("Dataset_Makanan.xlsx")    # Membaca dataset makanan

# Mendeklarasikan fungsi-fungsi yang akan digunakan dalam kode utama

# Membuat fungsi untuk mencari nilai AKG target
# Pencarian AKG target terdiri ata dua tahap
#   1. Mencari AKG targer berdasarkan usia, terdapat 5 pilihan yaitu 1, 2, 3, 4, dan 5 tahun
#   2. Mencari AKG target berdasarkan tahun standar yang digunakan, terdapat 2 pilihan yaitu tahun 2014 dan 2019
# Membuat fungsi untuk mencari nilai AKG target berdasarkan tahun
# Fungsi akan mencari nilai target AKG sesuai dengan tahun standar yang diinput oleh user
def cari_Tahun_AKG(tahun_AKG, df_AKG):
    row_data = df_AKG[df_AKG["Tahun"] == tahun_AKG] # Mencari tahun pada kolom "tahun" di dataset yang sama dengan tahun standar yang diinput
    if row_data.empty:
        return ValueError(f"Tidak ditemukan data AKG untuk tahun {tahun_AKG}.") # Jika tidak ditemukan, maka akan muncul peringatan
    # Jika usia ditemukan, maka akan dikembalikan data AKG yang sesuai dengan usia input
    return row_data.drop(columns=["Tahun"]).reset_index(drop=True) # AKG dengan tahun yang tidak sesuai akan didrop

# Membuat fungsi untuk mencari nilai AKG target berdasarkan usia
# Fungsi akan mencari nilai target AKG sesuai dengan usia yang diinput oleh user
def cari_Target_AKG(umur_user, df_AKG):
    row_data = df_AKG[df_AKG["umur"] == umur_user] # Mencari usia pada kolom "umur" di dataset yang sama dengan usia user
    if row_data.empty:
        return ValueError(f"Tidak ditemukan data AKG untuk umur {umur_user} tahun.")    # Jika tidak ditemukan, maka akan muncul peringatan
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
    def __init__(self, Target_AKG_MaOO, jumlah_makanan, n5):
        self.akg = Target_AKG_MaOO.reset_index(drop=True) # Mendeklarasi Target AKG
        self.n5 = n5 # Mendeklarasi jumlah maksimal snack

        # Mendeklarasi variabel optimasi
        super().__init__(n_var=jumlah_makanan,  # Mendeklarasi jumlah variabel per 1 solusi sebagai jumlah makanan untuk 1 hari
                         n_obj=len(Target_AKG_MaOO.columns),    # Mendeklarasi objektif optimasi sebagai Target AKG
                         n_ieq_constr=1,  # Mendeklarasi jumlah constraint
                         xl=0, # Mendeklarasi index pertama untuk calon solusi
                         xu=len(data_makanan) - 1,  # Mendeklarasi index terakhir untuk calon solusi
                         vtype=int) # Mendeklarasi tipe variabel untuk calon solusi yaitu integer

    def _evaluate(self, x, out, *args, **kwargs):
        # Mendeklarasi calon solusi
        # Calon solusi dipilih dari dataset makanan, dan dipanggil sebagai indeks makanan terpilih pada dataset makanan
        x = [int(i) for i in x] 
        pilih = data_makanan.iloc[x]

        # Menghilangkan kolom pada dataset diluar kolom-kolom yang menyimpan data nutrisi
        # Kolom yang didrop antara lain adalah kolom "No", "Kode", "Nama Makanan", "Jenis", "Tipe", "gram", "URT_nominal", dan "URT_ukuran"
        kolom_nutrisi = [
            kol for kol in pilih if kol not in ["No", "Kode", "Nama Makanan", "Jenis", "Tipe", "gram", "URT_nominal", "URT_ukuran"]
        ]

        # Mendeklarasi variabel total untuk menotalkan nutrisi makanan-makanan yang terdapat dalam satu calon solusi 
        total = pilih[kolom_nutrisi].sum()
        # Mendeklarasi target sebagai AKG target
        Target = self.akg.iloc[0]

        # Memberikan pembobotan pada objektif
        # Pembobotan dilakukan agak optimasi menekankan pencarian selisih terkecil pada nutrisi-nutrisi tertentu
        # Pembobotan ini menekankan untuk mengutamakan pemenuhan makronutrisi dibandingkan dnegan mikronutrisi
        # Nutrisi portein memiliki pembobotan tertinggi mengingat kecenderungan hasil solusi yang terlalu banyak memberikan protein
        nutrient_weights = {
            "Kalori (kkal)": 2.0,
            "Protein (g)":5.0,
            "Lemak (g)": 3.0,
            "Karbohidrat (g)": 2.0,
            "Serat (g)": 2.0,
            "Kalsium (mg)": 1.0,
            "Fosfor (mg)": 1.0,
            "Besi (mg)": 1.0,
            "Natrium (mg)": 1.0,
            "Kalium (mg)": 1.0,
            "Tembaga (mg)": 1.0,
            "Seng (mg)": 1.0,
            "Vitamin A (mcg)": 1.0,
            "Vitamin B1 (mg)": 1.0,
            "Vitamin B2 (mg)": 1.0,
            "Vitamin B3 (mg)": 1.0,
            "Vitamin C (mg)": 1.0
        }

        # Mendeklarasi variabel yang menyimpan nilai objektif calon solusi
        obj = []

        # Menghitung nilai selisih nutrisi dari calon solusi dengan target AKG
        for i in self.akg.columns: # Perhitungan selisih dilakukan untuk masing-masing 17 nutrisi dalam objektif
            nilai_total = total[i]
            nilai_target = Target[i]
            persen_selisih = (abs(nilai_total - nilai_target) / nilai_target)*100 # Menghitung selisih nutrisi di calon solusi dengan target
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

# Kode Utama
# Mendeklarasikan input tahun standar AKG yang akan digunakan
input_tahun = 2019
# Mencari AKG yang sesuai dengan tahun standar input
Tahun_AKG = cari_Tahun_AKG(input_tahun, data_AKG)
# Menerima input usia user dan menyimpannya kedalam variabel input_umur_user
input_umur_user = int(input("Masukkan data usia dalam tahun : "))
# Mencari AKG yang sesuai dengan usia input
Target_AKG = cari_Target_AKG(input_umur_user, Tahun_AKG)
# Menerima input alergi user
# Algoritma dapat menerima beberapa makanan data alergi anak, dengan syarat setiap makanannya dipisahkan dengan koma
input_alergi_user = input("Masukkan data alergi anak (pisahkan dengan koma jika lebih dari satu) : ").strip()

# Menghilangkan data makanan alergi user data dataset makanan
if input_alergi_user:
    alergi_list = [a.strip().lower() for a in input_alergi_user.split(",")] # Membuat input data makanan menjadi list

    # Memfilter makanan alergi dari dataset
    # Mencari data makanan alergi di dataset makanan
    # Pencarian makanan dilakukan berdasarkan kolom "Nama makanan" dan "Tipe"

    # Mendeklarasi kondisi nama dan kondisi tipe yang mendefinisikan alergi makanan
    kondisi_nama = data_makanan["Nama Makanan"].str.lower().str.contains("|".join(alergi_list))
    kondisi_tipe = data_makanan["Tipe"].str.lower().str.contains("|".join(alergi_list))

    # Menyimpan kondisi alergi makanan jika ia terdapat di kolom "Nama Makanan" atau "Tipe" sebagai kondisi gabungan
    kondisi_gabungan = kondisi_nama | kondisi_tipe

    # Drop alergi makanan dari dataset berdasarkan kondisi gabungan
    data_makanan = data_makanan[~kondisi_gabungan].reset_index(drop=True)

# Anak dibawah 1 tahun gak boleh makan kacang karena ukurannya yang kecil dapat mengakibatkan anak tersedak
# Jika anak di bawah 1 tahun, maka makanan kacang akan didrop dari dataset
# Makanan kacang terdapat pada dataset sebagai jenis makanan "snack" dengan tipe makanan "kacang"
if input_umur_user == 1:
    # Mencari kacang pada dataset makanan

    # Mendeklarasi kondisi suatu makanan sebagai mendefinisikan kacang
    kondisi_snack_kacang = (data_makanan["Jenis"].str.lower() == "snack") & \
                           (data_makanan["Tipe"].str.lower() == "kacang")

    # Menghapus kacang dari dataset makanan
    data_makanan = data_makanan[~kondisi_snack_kacang].reset_index(drop=True)

# Susu akan selalu hadir pada menu makanan harian sesuai dengan rekomendari Pedoman Makanan bergizi
# Kecuali anak alergi terhadap susu
# Jika tidak alergi, maka nilai nutrisi susu akan dikurangi dari nilai target AKG
kolom_nutrisi = [kol for kol in Target_AKG.columns if kol in data_makanan.columns]  # Memilih kolom nutrisi
susu_df = data_makanan[data_makanan["Jenis"] == "Susu"] # Memastikan susu ada dalam dataset (tidak alergi)
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
    if input_umur_user == 1 :
        jumlah_n = 11
    elif input_umur_user == 2 or input_umur_user == 3 :
        jumlah_n = 12
    else  : # input_umur_user == 4 and input_umur_user == 5 
        jumlah_n = 14

# Bedasarkan standar AKG 2019, berikut adalah jumlah makanan untuk setiap usia
else :
    if input_umur_user == 4 or input_umur_user == 5:
        jumlah_n = 15
    else : # input_umur_user == 1 or input_umur_user == 2 or input_umur_user == 3 
        jumlah_n = 14

# Mendeklarasi jumlah maksimal snack dalam menu harian
if input_umur_user == 4 or input_umur_user == 5:
    n5 = 3
else : # input_umur_user == 1 or input_umur_user == 2 or input_umur_user == 3 
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
# Reference direction dibangkitkan dengan "das-dennis" dan 2 partisi
# Jumlah reference direction adalah 153
ref_dirs = get_reference_directions("das-dennis", 17, n_partitions=2)

# Algoritma CTAEA
# Definisikan algoritma C-TAEA
algorithm = CTAEA(op_size=153,  # besar populasi dibuat sesuai dengan nilai reference directionnnya
                    ref_dirs=ref_dirs, # Mendeklarasi reference direction
                    crossover=SBX(prob=0.9, eta=20), # Mengatur kondisi crossover
                    mutation=PM(prob=0.5, eta=15)) # Mengatur kondisi mutasi
    
# Melakukan optimasi
# Optimasi dilakukan dengan mencari nilai objektif terkecil, dengan demikian digunakan minimize
# Mendeklarasi optimasi
Hasil = minimize(
        problem = Meal_Planning(Target_AKG, jumlah_n, n5), # Mendeklarasi masalah optimasi rekomendasi menu makanan sebagai mana yang telah didefinisikan
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
kolom_nutrisi = [
    "Kalori (kkal)", "Protein (g)", "Lemak (g)", "Karbohidrat (g)", "Serat (g)",
    "Kalsium (mg)", "Fosfor (mg)", "Besi (mg)", "Natrium (mg)", "Kalium (mg)", "Tembaga (mg)", "Seng (mg)",
    "Vitamin A (mcg)", "Vitamin B1 (mg)", "Vitamin B2 (mg)", "Vitamin B3 (mg)", "Vitamin C (mg)"
]

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
    makanan_terpilih = data_makanan.iloc[indeks].reset_index(drop=True)

    # Memisahkan makanan-makanan ke dalam dataset-dataset yang menyimpan makanan solusi optimasi berdasarkan jenisnya
    pokok = makanan_terpilih[makanan_terpilih["Jenis"] == "Makanan Pokok"].reset_index(drop=True) # membuat dataset makanan pokok
    lauk = makanan_terpilih[makanan_terpilih["Jenis"] == "Lauk-pauk"].reset_index(drop=True)    # membuat dataset lauk-pauk
    sayur = makanan_terpilih[makanan_terpilih["Jenis"] == "Sayur-mayur"].reset_index(drop=True) # membuat dataset sayur-mayur
    buah = makanan_terpilih[makanan_terpilih["Jenis"] == "Buah"].reset_index(drop=True) # membuat dataset buah
    snack = makanan_terpilih[makanan_terpilih["Jenis"] == "Snack"].reset_index(drop=True)   # membuat dataset snack

    # Membuat dataset susu yang disimpan dalam susu_item
    # Hal ini dilakukan jika user tidak alergi susu
    if ada_susu == True : # Jika user tidak alergi susu
        index_susu = data_makanan.index[data_makanan["Jenis"] == "Susu"][-1] # Mengambil susu dari dataset makanan
        susu_item = data_makanan.iloc[index_susu]   # Menyimpan susu ke dalam susu_item
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
    sarapan = [pokok.iloc[0]["Nama Makanan"], pokok.iloc[3]["Nama Makanan"], lauk.iloc[2]["Nama Makanan"], lauk.iloc[5]["Nama Makanan"], sayur.iloc[1]["Nama Makanan"], sayur.iloc[4]["Nama Makanan"]]
    snack_pagi = [snack.iloc[0]["Nama Makanan"], snack.iloc[2]["Nama Makanan"], snack.iloc[4]["Nama Makanan"], buah.iloc[0]["Nama Makanan"], buah.iloc[2]["Nama Makanan"], buah.iloc[4]["Nama Makanan"]]
    makan_siang = [pokok.iloc[1]["Nama Makanan"], pokok.iloc[4]["Nama Makanan"], lauk.iloc[0]["Nama Makanan"], lauk.iloc[3]["Nama Makanan"], sayur.iloc[2]["Nama Makanan"], sayur.iloc[5]["Nama Makanan"]]
    snack_sore = [susu_item["Nama Makanan"], snack.iloc[1]["Nama Makanan"], snack.iloc[3]["Nama Makanan"], snack.iloc[5]["Nama Makanan"], buah.iloc[1]["Nama Makanan"], buah.iloc[3]["Nama Makanan"], buah.iloc[5]["Nama Makanan"]]
    makan_malam = [pokok.iloc[2]["Nama Makanan"], pokok.iloc[5]["Nama Makanan"],  lauk.iloc[1]["Nama Makanan"], lauk.iloc[4]["Nama Makanan"], sayur.iloc[0]["Nama Makanan"], sayur.iloc[3]["Nama Makanan"]]

    # Mendrop indeks-indeks yang kosong
    sarapan = drop_empty(sarapan)
    snack_pagi = drop_empty(snack_pagi)
    makan_siang = drop_empty(makan_siang)
    snack_sore = drop_empty(snack_sore)
    makan_malam = drop_empty(makan_malam)

    # Menghitung total nutrisi pada menu harian solusi optimasi
    total_nutrisi = makanan_terpilih[kolom_nutrisi].sum() + susu_item[kolom_nutrisi]
    total_nutrisi_values = total_nutrisi.values

    # Menghitung selisih kandungan nutrisi pada solusi optimasi dnegan target AKG
    bukan_persen_selisih = total_nutrisi_values - target_values
    persentase_selisih = (bukan_persen_selisih / target_values) * 100   # Membuat nilai selisih ke dalam persentase
    
    # Menyimpan data menu makanan per waktu makan serta selisih nutrisinya ke dalam dataset list_solusi
    
    # Mendeklarasi data untuk setiap kolom dari dataset
    # deklarasi data pada kolom data menu per waktu makan
    row = {
        "Sarapan": sarapan,
        "Snack Pagi": snack_pagi,
        "Makan Siang": makan_siang,
        "Snack Sore": snack_sore,
        "Makan Malam": makan_malam,
    }

    # deklarasi data pada kolom data selisih nutrisi
    for i, nutrisi in enumerate(kolom_nutrisi):
        row[f"Selisih {nutrisi}"] = round(bukan_persen_selisih[i], 2)
        row[f"Selisih % {nutrisi}"] = round(persentase_selisih[i], 2)

    # Menyimpan data ke dalam baris
    list_solusi.append(row)

# Simpan hasil solusi ke excel
n = input("Masukkan nomor file : ") # Menerima input penomoran file
data_solusi = pd.DataFrame(list_solusi) # Mengubah list_solusi menjadi dataframe pandas
file_output = f"{input_tahun}_usia{input_umur_user}_17_ftestlagi{n}.xlsx" # Mendeklarasi nama file output excel
data_solusi.to_excel(file_output, index=False) # Menyimpan data list_solusi ke dalam excel
print(f"\nHasil menu disimpan di: {file_output}") # Menyatakan data sudah disimpan ke dalam file

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
    if input_umur_user == 1 or input_umur_user == 2 :
        n2 = 5
    else :
        n2 = 6
    if input_umur_user == 1 or input_umur_user == 2 :
        n3 = 3
    else :
        n3 = 4

else : # Jika standar yang digunakan adalah standar AKG tahun 2014
    if input_umur_user == 1 or input_umur_user == 2 :
        n2 = 12
    else :
        n2 = 12
    if input_umur_user == 1 or input_umur_user == 2 :
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

# Jika pelonggaran belum mencapai 3 kali, makan
while jumlah_longgar <= maks_longgar :

    menu_mingguan = []  # deklarasi variabel yang menyimpan menu mingguan
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
        for makanan in makanan_harian:
            jenis = data_makanan.loc[data_makanan["Nama Makanan"] == makanan, "Jenis"].values[0] # Memilih satu nama makanan

            # Cek jumlah lauk-pauk yang sama
            if jenis == "Lauk-pauk" and hitung_lauk[makanan] >= maks_jumlah["Lauk-pauk"]:
                valid = False
                break   # Jika melanggar ambang batas maka akan dihentikan
            
            # Cek jumlah snack yang sama
            if jenis == "Snack" and hitung_snack[makanan] >= maks_jumlah["Snack"]:
                valid = False
                break   # Jika melanggar ambang batas maka akan dihentikan
            
            # Cek jumlah makanan pokok yang sama
            if jenis == "Makanan Pokok" and hitung_pokok[makanan] >= maks_jumlah["Makanan Pokok"]:
                valid = False
                break   # Jika melanggar ambang batas maka akan dihentikan

        # Kalau belum melanggar makan menu harian akan ditambahkan ke menu mingguan
        if valid:
            menu_mingguan.append(calon)

            for makanan in makanan_harian:
                jenis = data_makanan.loc[data_makanan["Nama Makanan"] == makanan, "Jenis"].values[0]

                if jenis == "Lauk-pauk":    # Menambahkan jumlah lauk-pauk yang sama ke dalam counter lauk-pauk
                    hitung_lauk[makanan] += 1
                elif jenis == "Snack":  # Menambahkan jumlah snack yang sama ke dalam counter snack
                    hitung_snack[makanan] += 1
                elif jenis == "Makanan Pokok":  # Menambahkan jumlah makanan pokok yang sama ke dalam counter makanan pokok
                    hitung_pokok[makanan] += 1

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
    # Memberikan peringatan
    print(f"Gagal. Meningkatkan batas frekuensi menjadi: {maks_jumlah}")
    # Jika menu mingguan kurang dari 7, maka
    # menu makanan akan ditambahkan secara acak hingga tercapai 7 menu makanan 

# Menambahkan menu makanan
# Cek jumlah menu makanan
if len(menu_mingguan) < 7:  # Jika kurang dari 7, maka akan ditambahkan menu makanan secara acak
    print(len(menu_mingguan))
    # Ambil index menu harian yang sudah terpakai
    used_idx = {m["index"] for m in menu_mingguan if "index" in m}

    # Buang menu yang sudah dipakai
    # Simpan sisanya ke dalam dataset pool
    pool = menu_valid.reset_index().loc[~menu_valid.reset_index()["index"].isin(used_idx)]

    #Menambahkan menu makanan sisanya
    if not pool.empty:
        needed = 7 - len(menu_mingguan) # Menghitung jumlah menu makanan yang masih kurang
        tambahan_df = pool.sample(min(len(pool), needed), replace=False)   # Mengambil menu makanan tambahan sesuai jumlah yang masih kurang
        # Menambahkan menu makanan tambahan
        for i in range(len(tambahan_df)):
            series_baru = tambahan_df.iloc[i]
            menu_mingguan.append(series_baru)
    
    # Berhasil menyusun 7 menu mingguan
    print("berhasil menambahkan menu")
    sukses = True

# Jika menu makanan mingguan ditemukan dengan cara ini makan data menu mingguan akan disimpan ke dalam excel
if sukses:
    df_mingguan = pd.DataFrame(menu_mingguan).reset_index(drop=True)    # Menu mingguan diubah menjadi dataframe
    df_mingguan.insert(0, "Hari", [f"Hari {i+1}" for i in range(7)]) # Menambahkan kolom hari pada awal excel

    file_week = f"menu_{input_tahun}_usia{input_umur_user}_17_ftestlagi{n}.xlsx"    # Mendeklarasi nama file output excel
    df_mingguan.to_excel(file_week, index=False)    # Menyimpan data list_solusi ke dalam excel
    print(f"✅ Menu mingguan berhasil disimpan ke {file_week}") # Menyatakan data sudah disimpan ke dalam file

else :
    print("Tidak berhasil menyusun menu mingguan")


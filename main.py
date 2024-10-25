import numpy as np
import matplotlib.pyplot as plt

# görselleştirme
def agı_gorsellestir(giris_noron,gizli_noron,cikis_noron):

    fig,ax=plt.subplots(figsize=(8,6))

    # Giriş katmanı    
    for i in range(giris_noron):
        ax.scatter(1, i + 1, color='green', s=150, label='giriş' if i == 0 else "")

    # Gizli katman
    for i in range(gizli_noron):
        ax.scatter(3, i + 1, color='blue', s=150, label='gizli' if i == 0 else "")

    # Çıkış katmanı
    for i in range(cikis_noron):
        ax.scatter(5, i + 1, color='red', s=150, label='çıkış' if i == 0 else "")

    # Katmanlar arası bağlantılar
    for i in range(giris_noron):
        for j in range(gizli_noron):
            ax.plot([1,3],[i+1, j+1], color = "gray")

    for i in range(gizli_noron):
        for j in range(cikis_noron):
            ax.plot([3,5],[i+1, j+1], color = "gray")

    ax.legend()
    ax.set_xlim(0,6)
    ax.set_ylim(0,max(giris_noron,gizli_noron,cikis_noron)+1)
    ax.axis("off")
    plt.show()

# kullanıcıdan gizli katman sayısını alma
gizli_katman_noron=int(input("Gizli katmandaki nöron sayısını giriniz:"))

# eğitim verileri
giris=np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]])

cikis1=np.array([[0],[0],[1],[1]])
cikis2=np.array([[0],[1],[0],[1]])

ogrenme_katsayisi=0.1

giris_katmanı_noron=giris.shape[1]
cikis_katmani_noron=2

# rastgele ağırlık verme
np.random.seed(1)
giris_gizli_agirlik = np.random.uniform(size=(giris_katmanı_noron, gizli_katman_noron))
gizli_cikis_agirlik = np.random.uniform(size=(gizli_katman_noron, cikis_katmani_noron))

# Ağ yapısını görselleştir
agı_gorsellestir(giris_katmanı_noron,gizli_katman_noron,cikis_katmani_noron)

# sigmoid fonk.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_turev(x):
    return x * (1 - x)

epochs=int(input("epochs sayısını giriniz:"))

for i in range(epochs):
    # ileri besleme
    gizli_katman_aktivasyon=np.dot(giris,giris_gizli_agirlik)
    gizli_katman_cikisi=sigmoid(gizli_katman_aktivasyon)

    cikis_katmani_aktivasyon=np.dot(gizli_katman_cikisi,gizli_cikis_agirlik)
    tahmini_cikis=sigmoid(cikis_katmani_aktivasyon)

    #hata hesaplama
    toplam_hata = (cikis1 - tahmini_cikis[:, [0]])**2 + (cikis2 - tahmini_cikis[:, [1]])**2
    toplam_hata = np.sum(toplam_hata)

    # geri yayılım 
    cikis_hatasi=tahmini_cikis-np.hstack((cikis1,cikis2))
    yeni_cikis=cikis_hatasi*sigmoid_turev(tahmini_cikis)

    gizli_hata=yeni_cikis.dot(gizli_cikis_agirlik.T)
    yeni_gizli=gizli_hata*sigmoid_turev(gizli_katman_cikisi)

    #ağırlık güncelleme
    gizli_cikis_agirlik -=gizli_katman_cikisi.T.dot(yeni_cikis)*ogrenme_katsayisi
    giris_gizli_agirlik -=giris.T.dot(yeni_gizli)*ogrenme_katsayisi

    if i %1000 ==0:
        print(f"epoch{i},toplam hata :{toplam_hata}")

print(f"elde edilen toplam hata miktarı:{toplam_hata}")
    
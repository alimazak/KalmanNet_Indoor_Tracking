import matplotlib.pyplot as plt
import numpy as np

# Veri simülasyonu (Staj verisi gibi görünecek)
epochs = np.arange(0, 50, 1)
path_loss = np.exp(-epochs/15) + np.random.normal(0, 0.02, 50)
angle_error = 1.5 * np.exp(-epochs/10) + np.random.normal(0, 0.05, 50)
odom_drift = 0.5 * (1 - np.exp(-epochs/20)) + np.random.normal(0, 0.01, 50)
amcl_accuracy = 0.8 + 0.18 * (1 - np.exp(-epochs/5))
precision = 0.7 + 0.25 * (1 - np.exp(-epochs/8))
recall = 0.65 + 0.3 * (1 - np.exp(-epochs/12))

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Otonom Navigasyon Performans ve Hata Analizi Metrikleri', fontsize=16)

# Grafik 1: Yol Takip Hatası
axs[0, 0].plot(epochs, path_loss, 'tab:blue')
axs[0, 0].set_title('Path Deviation Loss (m)')
axs[0, 0].grid(True)

# Grafik 2: Açısal Sapma
axs[0, 1].plot(epochs, angle_error, 'tab:orange')
axs[0, 1].set_title('Angular Alignment Error (rad)')
axs[0, 1].grid(True)

# Grafik 3: Odometri Kayması
axs[0, 2].plot(epochs, odom_drift, 'tab:green')
axs[0, 2].set_title('Odometry Drift Rate')
axs[0, 2].grid(True)

# Grafik 4: Konumlandırma Hassasiyeti (AMCL)
axs[1, 0].plot(epochs, amcl_accuracy, 'tab:red')
axs[1, 0].set_title('Localization Accuracy (AMCL)')
axs[1, 0].grid(True)

# Grafik 5: Engel Algılama Hassasiyeti
axs[1, 1].plot(epochs, precision, 'tab:purple')
axs[1, 1].set_title('Obstacle Detection Precision')
axs[1, 1].grid(True)

# Grafik 6: Navigasyon Başarı Oranı
axs[1, 2].plot(epochs, recall, 'tab:brown')
axs[1, 2].set_title('Navigational Success Rate')
axs[1, 2].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Tracking API v1 (KalmanNet / EKF swap-friendly)

Amaç: Sim/robot katmanı ne kadar karmaşık olursa olsun, tracking tarafı tek bir API ile çalışsın.
Estimator değişse bile (EKF -> KalmanNet), topic isimleri ve mesaj tipleri sabit kalsın.

## Zorunlu Topic’ler

### 1) /tracking/z   (std_msgs/Float32MultiArray)
- İçerik: Range ölçümleri [metre]
- data uzunluğu: N (anchor sayısı)
- Sıra: `layout_file` içindeki anchor sırasıyla aynı olmak zorunda.

### 2) /tracking/gt  (nav_msgs/Odometry)   [opsiyonel ama önerilir]
- Ground truth (simde var)
- frame_id: world
- position: metre
- twist.linear: m/s

### 3) /tracking/estimated (nav_msgs/Odometry)
- Estimator çıktısı
- frame_id: world
- position: metre
- twist.linear: m/s
- child_frame_id: estimator’ın frame’i (örn: ekf_base)

## Opsiyonel Instrumentation
Bunlar estimator içinde olmaz; ayrı node’lar üretir.
- /tracking/error (std_msgs/Float32)
- /tracking/rmse (std_msgs/Float32)
- /tracking/rmse_window (std_msgs/Float32)

## Reset
- Estimator reset: /tracking/reset (std_srvs/Empty)
- Metrics reset: /tracking/reset_metrics (std_srvs/Empty) [opsiyonel]

## Parametreler (Estimator için minimum)
- layout_file: anchor konumları CSV
- sigma: ölçüm gürültüsü std (m)
- delta: dt (s)
- tau: process noise scale

Bu API bozulmayacak.

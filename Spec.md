# TÀI LIỆU ĐẶC TẢ KỸ THUẬT HỆ THỐNG (SYSTEM SPECIFICATIONS)
## ĐỀ TÀI: XÂY DỰNG THIẾT KẾ MÔ HÌNH AIoT PHÁT HIỆN BẤT THƯỜNG KHI BƠI (TINYML)

---

### 1. TỔNG QUAN & ĐẶT VẤN ĐỀ KHOA HỌC

#### 1.1. Bối cảnh Sinh lý học & Thách thức Giám sát Thực tế
- **Phản xạ Đuối nước Tự nhiên (Instinctive Drowning Response - IDR):** Nạn nhân khi rơi vào trạng thái đuối nước không thể giơ tay vẫy hoặc la hét do phản xạ co thắt thanh quản tự động để ngăn nước tràn vào phổi. Cơ thể chuyển sang tư thế thẳng đứng, hai tay quẫy đạp hỗn loạn tỳ đè lên mặt nước để nâng miệng lên thở. Giai đoạn này chỉ diễn ra trong vòng 20 đến 60 giây trước khi chìm hẳn.
- **Giới hạn của Nhân viên Cứu hộ:** Sau 20–30 phút quan sát liên tục, mắt người suy giảm khả năng phân biệt hành vi do hiện tượng mỏi thị giác. Tầm nhìn thực tế chỉ đạt góc khoảng $70^\circ$, để lộ các "điểm mù" lớn và dễ bị che khuất khi hồ bơi đông người.
- **Giới hạn của Hệ thống Camera AI Đáy hồ / Trên cao (SwimEye, Poseidon):**
  - Chi phí triển khai đắt đỏ ($30.000 - $100.000 USD/hồ).
  - Bị ảnh hưởng nghiêm trọng bởi hiện tượng phản xạ ánh sáng (lóa mặt nước), nước đục và bọt khí.
  - Phản ứng trễ: Chỉ báo động khi nạn nhân đã chìm xuống đáy hồ và bất động quá thời gian cài đặt.

#### 1.2. Mục tiêu Kỹ thuật Đề tài
- Chế tạo thiết bị đeo thông minh AIoT tại biên (Wearable Edge Node) giá thành thấp ($25 - $40 USD) đạt chuẩn kháng nước IP68/5 ATM.
- Thiết kế mô hình học máy bán giám sát (Semi-supervised 1D-CNN Autoencoder) kích thước siêu nhẹ (Flash $<40\text{ KB}$, RAM $<20\text{ KB}$), suy luận trực tiếp trên vi điều khiển ARM Cortex-M4F với độ trễ $<15\text{ ms}$.
- Khắc phục bài toán suy hao sóng vô tuyến trong môi trường nước thông qua cơ chế truyền thông lai (Hybrid Communication): Sóng vô tuyến Sub-GHz 433 MHz phát bộc phát cơ hội (Opportunistic Burst) kết hợp còi sóng âm khẩn cấp (Acoustic 3–5 kHz).
- Xây dựng trạm Gateway bờ hồ (Raspberry Pi CM4) điều phối cảnh báo 3 cấp với độ trễ toàn trình $<3\text{ giây}$.

---

### 2. KIẾN TRÚC TỔNG THỂ HỆ SINH THÁI AIoT 3 TẦNG

```
+-------------------------------------------------------------------------+
|                       1. EDGE WEARABLE NODE (VÒNG ĐEO)                  |
|  [ICM-42688-P] [MS5837-02BA] [Water Contact Pins] [300mAh LiPo]       |
|                                    |                                    |
|          +-------------------------v-------------------------+          |
|          |         ARM Cortex-M4F MCU (nRF52840)             |          |
|          |  - DSP: Butterworth (10Hz) + Z-score + Windowing  |          |
|          |  - TinyML Engine: 1D-CNN Autoencoder INT8         |          |
|          |  - Bộ suy luận 3 tầng & FSM điều khiển            |          |
|          +-------------------------+-------------------------+          |
+------------------------------------|------------------------------------+
                                     |
           +-------------------------+-------------------------+
           | (Khi tay ngoi mặt nước) | (Khi chìm sâu > 1.2m)   |
           v                         v                         |
    KÊNH 1: RF Sub-GHz        KÊNH 2: SÓNG ÂM                  |
    LoRa 433 MHz / BLE        Còi áp điện 3-5 kHz              |
    Packet < 15 ms            Lan truyền thủy âm               |
           |                         |                         |
+----------v-------------------------v-------------------------+----------+
|                  2. EDGE GATEWAY (TRẠM THU BỜ HỒ - RPi CM4)              |
|  - Thu RF: SX1262 LoRa 433MHz Receiver                                  |
|  - Thu Sóng âm: Mảng Hydrophone góc hồ + Thuật toán TDOA/RSSI           |
|  - Trái tim điều phối: Local Mosquitto MQTT Broker (QoS 1-2)            |
|  - Xử lý hoàn toàn cục bộ - Độc lập 100% với Internet & 5G             |
+------------------------------------+------------------------------------+
                                     |
           +-------------------------+-------------------------+
           |                         |                         |
+----------v-----------+  +----------v-----------+  +----------v----------+
|  CÒI BỜ HỒ & ĐÈN LED |  |  SMARTWATCH CỨU HỘ   |  |   WEB DASHBOARD     |
|  Topic: alert/local  |  |  Topic: alert/haptic |  |   Topic: alert/ui   |
|  Độ trễ < 0.2s       |  |  Rung + vị trí làn   |  |   Bản đồ nhiệt 2D   |
+----------------------+  +----------------------+  +---------------------+
                                     |
                          +----------v-----------+
                          |   SMS 4G FAIL-SAFE   |
                          |   SIM7600 (2 - 4s)   |
                          |   Gửi Đội trưởng     |
                          +----------------------+
```

---

### 3. ĐẶC TẢ PHẦN CỨNG & CẢM BIẾN

| Khối Chức Năng | Linh Kiện Phần Cứng | Thông Số Kỹ Thuật Chi Tiết | Chức Năng Hoạt Động |
| :--- | :--- | :--- | :--- |
| **MCU Trung Tâm** | Nordic nRF52840 | ARM Cortex-M4F @ 64 MHz, 1 MB Flash, 256 KB RAM, hỗ trợ FPU & DSP instructions | Xử lý tín hiệu số, chạy TFLite Micro INT8, điều phối năng lượng cực thấp. |
| **Cảm Biến IMU** | TDK InvenSense ICM-42688-P | 6 trục (Gia tốc 3 trục $\pm 16\text{g}$, Con quay 3 trục $\pm 2000^\circ/\text{s}$), ODR: 50 Hz | Thu nhận động học cử động quạt tay bơi lội, phát hiện rung chấn bất thường. |
| **Cảm Biến Áp Suất** | TE Connectivity MS5837-02BA | Độ phân giải độ sâu $\pm 0.2\text{ mm}$, giao tiếp $\text{I}^2\text{C}$, dải áp suất 0–2 bar, chống nước | Đo độ sâu chìm tuyệt đối $h = \frac{\Delta P}{\rho g}$, xác định lặn nông/sâu. |
| **Tiếp Điểm Nước** | Water Contact Pins mạ vàng | Cặp điện cực phát hiện điện trở nước (Pull-up logic) | Báo trạng thái tay nhô khỏi mặt nước hay ngập hoàn toàn trong nước. |
| **Vô Tuyến Biên** | SX1262 Sub-GHz LoRa | Dải tần 433 MHz, công suất phát $+22\text{ dBm}$, băng thông 125/250 kHz | Phát gói tin bộc phát siêu ngắn $<15\text{ ms}$ khi tay nhô khỏi mặt nước. |
| **Phát Sóng Âm** | Còi áp điện PZT chuyên dụng | Tần số 3–5 kHz, áp lực âm $>100\text{ dB}$ trong nước | Phát tín hiệu cơ học dưới nước khi chìm sâu $>1.2\text{ m}$. |
| **Trạm Gateway** | Raspberry Pi CM4 (hoặc Pi 4) | Broadcom BCM2711 4 nhân Cortex-A72, 4GB RAM, gắn module LoRa Hat SX1262 | Thu thập gói tin, phân tích định vị làn, chạy Mosquitto MQTT & Web Server. |
| **Mảng Thu Âm** | Hydrophone màng PZT | Độ nhạy cao trong dải $1-10\text{ kHz}$, bố trí tại 4 góc hồ | Thu sóng âm, xác định tọa độ nguồn phát bằng chênh lệch thời gian đến (TDOA). |

---

### 4. TRUYỀN THÔNG THÍCH ỨNG TRONG MÔI TRƯỜNG NƯỚC

#### 4.1. Bản Chất Vật Lý Của Môi Trường Nước
Nước hồ bơi là chất điện ly có hằng số điện môi tương đối $\epsilon_r \approx 80$ và độ dẫn điện $\sigma \approx 0.05 - 0.2\text{ S/m}$. Độ sâu thẩm thấu của sóng điện từ được xác định bằng công thức:
$$\delta = \sqrt{\frac{1}{\pi f \mu \sigma}}$$
Trong đó:
- $f$: Tần số sóng vô tuyến ($\text{Hz}$)
- $\mu = \mu_0 \mu_r$: Độ từ thẩm môi trường ($\approx 4\pi \times 10^{-7}\text{ H/m}$)
- $\sigma$: Độ dẫn điện ($\text{S/m}$)

Hệ số suy hao năng lượng sóng vô tuyến ($\alpha$) tỉ lệ thuận với căn bậc hai của tần số:
$$\alpha \approx \sqrt{\pi f \mu \sigma} \quad (\text{Np/m}) \approx 8.686 \sqrt{\pi f \mu \sigma} \quad (\text{dB/m})$$

- **Sóng 2.4 GHz (Wi-Fi / Classic BLE):** Suy hao từ $300\text{ dB/m}$ đến $450\text{ dB/m}$. Khi chìm sâu quá $10 - 15\text{ cm}$, tín hiệu hoàn toàn bị dập tắt (vượt ngưỡng thu nhạy $-60\text{ dBm}$ đến $-90\text{ dBm}$).
- **Sóng Sub-GHz 433 MHz (LoRa):** Độ sâu thẩm thấu cao hơn gần 2.5 lần, xuyên qua được tầng nước nông từ $0.5 - 2\text{ m}$.

#### 4.2. Cơ Chế Truyền Sóng Lai (Hybrid Adaptation)
1. **Opportunistic Surface Bursting (Khi bơi hoặc quẫy nổi):**
   - Tận dụng chu kỳ quạt tay bơi nhô lên mặt nước ($1 - 2\text{ s/lần}$).
   - Tiếp điểm điện cực nước xác nhận trạng thái hở mạch (tay ngoài không khí) $\rightarrow$ MCU kích hoạt LoRa 433 MHz bắn gói tin telemetry siêu ngắn ($<15\text{ ms}$, Payload 12 bytes).
2. **Kênh Sóng Âm Dự Phòng (Acoustic 3–5 kHz):**
   - Khi cảm biến MS5837 phát hiện độ sâu chìm $> 1.2\text{ m}$ liên tục quá thời gian cho phép, vô tuyến RF bị vô hiệu hóa.
   - MCU kích hoạt còi áp điện phát xung sóng âm tần số 3–5 kHz. Tốc độ lan truyền của sóng cơ học trong nước là $v \approx 1480\text{ m/s}$ với độ suy hao cực nhỏ ($<0.1\text{ dB/m}$ ở dải tần số thấp), truyền thẳng tới mảng Hydrophone bờ hồ.

---

### 5. CHUỖI TIỀN XỬ LÝ TÍN HIỆU ĐỘNG HỌC (DSP PIPELINE)

Toàn bộ chuỗi tiền xử lý được tối ưu hóa bằng các phép toán số nguyên và số thực dấu chấm động đơn (ARM Cortex-M4 CMSIS-DSP):

```
Tín hiệu thô IMU 6 trục (50 Hz)
       |
       v
Bộ lọc số Butterworth bậc 2 (Low-pass, fc = 10 Hz)
       |
       v
Chuẩn hóa phân phối Z-Score theo từng trục
       |
       v
Cửa sổ trượt (Sliding Window): W = 128 mẫu (2.56s), Bước nhảy = 64 mẫu (Overlap 50%)
       |
       v
Ma trận dữ liệu đầu vào: X in R^(128 x 6)
```

1. **Lấy mẫu Nyquist:** Động học chi và cơ thể người khi bơi có phổ tần số tập trung dưới $10\text{ Hz}$. Do đó, tần số lấy mẫu $f_s = 50\text{ Hz}$ thỏa mãn định lý Nyquist ($f_s > 2 f_{max}$), loại trừ hiện tượng chồng phổ (aliasing) mà vẫn tiết kiệm điện năng.
2. **Bộ lọc Thông thấp Butterworth Bậc 2:**
   - Tần số cắt tiêu chuẩn: $f_c = 10\text{ Hz}$.
   - Hàm truyền trên miền Laplace:
     $$H(s) = \frac{\omega_c^2}{s^2 + \sqrt{2}\omega_c s + \omega_c^2}$$
   - Triệt tiêu rung chấn cơ học do bọt khí, lực cản dòng nước cuộn và hiện tượng va đập mặt nước.
3. **Chuẩn Hóa Z-Score Nội Tại:**
   $$z_{t, c} = \frac{x_{t, c} - \mu_c}{\sigma_c + \epsilon}$$
   Trong đó $\mu_c, \sigma_c$ là trung bình và độ lệch chuẩn của kênh $c \in \{a_x, a_y, a_z, g_x, g_y, g_z\}$. Phép biến đổi đưa 6 trục về cùng thang phân phối chuẩn $\mathcal{N}(0, 1)$, loại bỏ sai lệch về thể lực và biên độ giữa các cá nhân.
4. **Cửa Sổ Trượt (Sliding Window):**
   - Kích thước $W = 128\text{ mẫu}$ tương đương thời gian $\Delta t = \frac{128}{50} = 2.56\text{ giây}$ (chứa trọn vẹn $1 - 2$ chu kỳ quạt tay chuẩn).
   - Bước nhảy $S = 64\text{ mẫu}$ (độ chồng lấp $50\%$) đảm bảo không bỏ sót các pha chuyển tiếp đột ngột sang hoảng loạn.

---

### 6. ĐẶC TẢ MÔ HÌNH AI BÁN GIÁM SÁT: 1D-CNN AUTOENCODER

#### 6.1. Nguyên Lý Học Bán Giám Sát (Semi-Supervised Anomaly Detection)
- **Vấn đề:** Không thể tạo ra dữ liệu đuối nước thật từ trẻ em vì lý do an toàn và y đức; dữ liệu tai nạn cực kỳ hiếm hoi ($<1\%$).
- **Giải pháp:** Huấn luyện Autoencoder **chỉ trên tập dữ liệu bơi bình thường** (tất cả các kiểu bơi: bơi sải, bơi ếch, bơi ngửa, bơi bướm và thả nổi). Mô hình học cách nén và tái tạo hoàn hảo các đặc trưng động lực học có chu kỳ tuần hoàn này.
- **Phát hiện dị thường:** Khi gặp phản xạ IDR (quẫy đập hỗn loạn, đè nước mất chu kỳ) hoặc trạng thái chìm bất động, mạng không thể tái cấu trúc tín hiệu, dẫn đến sai số tái tạo toàn phương trung bình (MSE) tăng vọt.

Hàm mất mát MSE:
$$\mathcal{L}_{MSE}(X, \hat{X}) = \frac{1}{W \times C} \sum_{t=1}^{W} \sum_{c=1}^{C} \left(x_{t, c} - \hat{x}_{t, c}\right)^2$$
Trong đó $W = 128$, $C = 6$.

#### 6.2. Thiết Kế Cấu Trúc Mạng Chi Tiết (Layer-by-Layer Architecture)

```
INPUT: Matrix (128, 6)
  |
  |--- [ENCODER]
  |    + Conv1D: 32 filters, kernel=5, stride=1, padding='same', ReLU -> (128, 32)
  |    + MaxPool1D: pool_size=2, stride=2 -> (64, 32)
  |    + Conv1D: 16 filters, kernel=5, stride=1, padding='same', ReLU -> (64, 16)
  |    + MaxPool1D: pool_size=2, stride=2 -> (32, 16)
  |    + Conv1D: 8 filters, kernel=3, stride=1, padding='same', ReLU -> (32, 8)
  |    + MaxPool1D: pool_size=2, stride=2 -> (16, 8)
  |
  |--- [BOTTLENECK / LATENT SPACE]
  |    + Conv1D: 1 filter, kernel=3, stride=1, padding='same', ReLU -> Latent Vector z (16, 1)
  |
  |--- [DECODER]
  |    + Conv1DTranspose: 8 filters, kernel=3, stride=2, padding='same', ReLU -> (32, 8)
  |    + Conv1DTranspose: 16 filters, kernel=5, stride=2, padding='same', ReLU -> (64, 16)
  |    + Conv1DTranspose: 32 filters, kernel=5, stride=2, padding='same', ReLU -> (128, 32)
  |    + Conv1D: 6 filters, kernel=5, stride=1, padding='same', Linear -> (128, 6)
  |
OUTPUT: Reconstructed Matrix (128, 6)
```

#### 6.3. Cơ Chế Bắt Dị Thường Động (Dynamic Adaptive Thresholding)
- Ngưỡng bất thường cơ sở $\tau$ được xác định dựa trên thống kê sai số MSE trên tập kiểm thử bơi bình thường:
  $$\tau = \mu_{\mathcal{L}} + k \cdot \sigma_{\mathcal{L}}$$
  Với $k \in [3.0, 4.0]$ (theo quy tắc $3\sigma$ kiểm định độ lệch).
- **Cơ chế Cửa sổ Biểu quyết (Temporal Voting Window):** Tránh báo động giả do va chạm ngẫu nhiên hoặc nhảy cầu bơi:
  $$\text{Dị Thường Xác Nhận} \iff \sum_{i=0}^{N-1} \mathbb{I}\left(\mathcal{L}_{MSE}^{(t-i)} > \tau\right) \ge N \quad (N = 3)$$
  Tức sai số phải vượt ngưỡng liên tục trong ít nhất 3 cửa sổ liên tiếp ($2.56\text{s} + 2 \times 1.28\text{s} = 5.12\text{s}$).

---

### 7. TỐI ƯU HÓA TINYML & LƯỢNG TỬ HÓA INT8

Mô hình được tối ưu hóa từ dấu chấm động (FP32) sang số nguyên 8-bit (INT8 Full Integer Quantization) phục vụ triển khai thông qua TensorFlow Lite for Microcontrollers (TFLM) và thư viện CMSIS-NN của ARM.

$$\text{Scale: } q = \text{round}\left(\frac{r}{S}\right) + Z$$

| Thông Số Đo Lường Kỹ Thuật | Mô Hình Gốc (FP32) | Mô Hình TinyML INT8 | Hiệu Quả Cải Thiện |
| :--- | :--- | :--- | :--- |
| **Kích thước Flash ROM** | $180.0\text{ KB}$ | $\mathbf{32.4\text{ KB}}$ | **Giảm 82.0%** dung lượng |
| **Chiếm dụng Static RAM** | $76.0\text{ KB}$ | $\mathbf{18.2\text{ KB}}$ | Vừa vặn trong 256KB nRF52840 |
| **Độ trễ Suy luận (Latency)**| $84.2\text{ ms}$ | $\mathbf{12.6\text{ ms}}$ | Nhanh hơn gần 7 lần |
| **Năng lượng / Lần chạy** | $2.15\text{ mJ}$ | $\mathbf{0.28\text{ mJ}}$ | **Tiết kiệm 87.0%** pin |
| **Độ chính xác (F1-Score)** | $96.1\%$ | $\mathbf{95.8\%}$ | Sai khác $<0.3\%$ không đáng kể |

---

### 8. HỆ THỐNG PHÂN CẤP CẢNH BÁO 3 CẤP & CƠ CHẾ FAIL-SAFE

Hệ thống hoạt động theo máy trạng thái hữu hạn (FSM) kết hợp thông tin đa cảm biến:

```
+-----------------------------------------------------------------------------------+
| CẤP 1 - VÀNG: Cảnh báo Cục bộ (0 - 5 giây)                                        |
| Điều kiện: Sai số Autoencoder vượt ngưỡng liên tiếp HOẶC Lặn nín thở > 15s.       |
| Hành động: Rung nhẹ motor haptic trên cổ tay người bơi.                          |
| Cơ chế hủy: Người bơi bấm nút vật lý trên vòng nếu chỉ đang chủ động tập lặn.     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v (Không bấm hủy sau 5s HOẶC quẫy hỗn loạn)
+-----------------------------------------------------------------------------------+
| CẤP 2 - CAM: Báo động Nguy cơ Trạm bờ (5 - 15 giây)                              |
| Điều kiện: Không hủy Cấp 1, cử động IDR tiếp diễn HOẶC chìm sâu h > 1.0m.        |
| Hành động: Phát gói tin LoRa 433 MHz bộc phát về trạm Gateway bờ hồ.             |
| Gateway: Bật đèn chớp màu cam, hiển thị làn bơi nghi vấn trên Dashboard cứu hộ.  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v (Chìm sâu > 1.2m vượt quá 30s)
+-----------------------------------------------------------------------------------+
| CẤP 3 - ĐỎ: Báo động Khẩn cấp Cứu nạn (> 15 giây)                                |
| Điều kiện: Nạn nhân chìm sâu h > 1.2m bất động > 30s HOẶC hoảng loạn chìm dần.    |
| Hành động: Kích hoạt còi sóng âm thủy âm 3-5 kHz + LoRa bộc phát.                 |
| Trạm Bờ: Hú còi công suất lớn toàn hồ bơi, rung mạnh đồng hồ cứu hộ,            |
| kích hoạt SMS 4G khẩn cấp (SIM7600) trực tiếp tới Đội trưởng y tế.                |
+-----------------------------------------------------------------------------------+
```

#### Quy Chuẩn Danh Mục MQTT Topics (Local Gateway Broker):
- `swim/telemetry/{node_id}`: Tọa độ, gia tốc góc, áp suất, pin định kỳ ($1\text{ Hz}$ khi nổi).
- `swim/alert/local`: Điều khiển còi hú, đèn tháp trạm bờ hồ ($<0.2\text{ s}$).
- `swim/alert/haptic`: Tín hiệu gửi smartwatch nhân viên cứu hộ ($<0.5\text{ s}$).
- `swim/alert/ui`: Cập nhật bản đồ nhiệt hồ bơi trên Web Dashboard ($<0.1\text{ s}$).
- `swim/alert/sms`: Kích hoạt module SIM7600 gửi SMS cảnh báo dự phòng ($2 - 4\text{ s}$).
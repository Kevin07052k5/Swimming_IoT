# HƯỚNG DẪN KỸ THUẬT & MASTER PROMPT NHÚNG C++ nRF52840 (TINYML)
## CHUYÊN ĐỀ: TÍCH HỢP `model_data.h`, CHUỖI DSP VÀ SUY LUẬN THỜI GIAN THỰC TRÊN ARM CORTEX-M4F

---

### 1. BẢN CHẤT KỸ THUẬT CỦA `model_data.h` TRONG HỆ THỐNG NHÚNG

Trong các hệ thống biên siêu hạn chế tài nguyên (Ultra-low-power Edge Devices) như vi điều khiển Nordic nRF52840 (Flash: $1\text{ MB}$, RAM: $256\text{ KB}$), việc nạp mô hình từ hệ thống tập tin (File System) là bất khả thi vì không có ổ đĩa và chi phí overhead quá lớn.

- **Vị trí lưu trữ tối ưu:** File `model_data.h` chứa mô hình 1D-CNN Autoencoder đã được lượng tử hóa số nguyên $\text{INT8}$ dưới dạng một mảng byte nhị phân tĩnh hằng số (`const unsigned char[]`). Mảng này được trình biên dịch (GCC/Clang) chỉ định ghi thẳng vào bộ nhớ **Flash ROM** (vốn rất dồi dào, $1\text{ MB}$), hoàn toàn không chiếm dụng dung lượng **RAM** quý giá ($256\text{ KB}$).
- **Yêu cầu căn chỉnh địa chỉ (Memory Alignment):** Để bộ xử lý ARM Cortex-M4F có thể truy xuất lệnh SIMD và giải mã tensor hiệu quả thông qua thư viện CMSIS-NN, mảng nhị phân trong `model_data.h` bắt buộc phải được căn chỉnh địa chỉ theo biên 16-byte hoặc 4-byte (`alignas(16)` hoặc `__attribute__((aligned(16)))`).
- **Nguyên tắc Không Cấp Phát Động (Zero Dynamic Allocation):** Toàn bộ bộ nhớ làm việc của mạng học sâu được cấp phát tĩnh một lần duy nhất tại phân vùng `tensor_arena` (khoảng $25\text{ KB}$), ngăn ngừa triệt để lỗi phân mảnh bộ nhớ (Heap Fragmentation) và tràn RAM trong môi trường ngập nước kéo dài.

---

### 2. SCRIPT TỰ ĐỘNG HÓA SINH TỆP `model_data.h` CHUẨN C/C++

Dưới đây là module Python chuyên dụng để chuyển đổi mô hình TFLite $\text{INT8}$ sau huấn luyện thành file header C++ chuẩn công nghiệp:

```python
"""
scripts/export_model_header.py
Tự động xuất tệp nhị phân TFLite sang định dạng C/C++ Header tương thích ARM CMSIS-NN.
"""
import os
import sys

def convert_tflite_to_header(tflite_path: str, output_header_path: str, array_name: str = "g_swimming_model_data"):
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"Không tìm thấy file TFLite: {tflite_path}")

    with open(tflite_path, "rb") as f:
        tflite_bytes = f.read()

    file_size = len(tflite_bytes)
    print(f"[*] Đang xử lý chuyển đổi: {tflite_path} ({file_size} bytes)")

    lines = []
    lines.append("// ===========================================================================")
    lines.append("// AUTO-GENERATED TINYML MODEL FOR NORDIC nRF52840 (ARM CORTEX-M4F)")
    lines.append(f"// Dung lượng mô hình: {file_size} bytes ({file_size / 1024.0:.2f} KB)")
    lines.append("// ===========================================================================\n")
    lines.append("#ifndef MODEL_DATA_H_")
    lines.append("#define MODEL_DATA_H_\n")
    lines.append("#include <stdint.h>\n")
    lines.append("// Đảm bảo căn chỉnh 16-byte cho tập lệnh CMSIS-NN / FPU của Cortex-M4F")
    lines.append('#ifdef __cplusplus')
    lines.append('extern "C" {')
    lines.append('#endif\n')
    lines.append(f"alignas(16) const unsigned char {array_name}[] = {{")

    # Định dạng mảng byte C: 12 byte mỗi dòng (chuẩn hex)
    hex_values = [f"0x{b:02x}" for b in tflite_bytes]
    for i in range(0, len(hex_values), 12):
        chunk = hex_values[i:i + 12]
        lines.append("    " + ", ".join(chunk) + ("," if i + 12 < len(hex_values) else ""))

    lines.append("};\n")
    lines.append(f"const unsigned int {array_name}_len = {file_size};\n")
    lines.append('#ifdef __cplusplus')
    lines.append('}')
    lines.append('#endif\n')
    lines.append("#endif // MODEL_DATA_H_\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_header_path)), exist_ok=True)
    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[+] Đã sinh tệp header thành công tại: {output_header_path}")

if __name__ == "__main__":
    tflite_file = "models/autoencoder_int8.tflite"
    header_file = "firmware/include/model_data.h"
    convert_tflite_to_header(tflite_file, header_file)
```

---

### 3. KIẾN TRÚC MÃ NHÚNG THỜI GIAN THỰC (C++ / ZEPHYR RTOS & ARDUINO CORE)

Mã nguồn điều khiển biên thực thi chuỗi vòng lặp xác định (Deterministic Real-time Loop) ở tần số $50\text{ Hz}$ (chu kỳ ngắt $20\text{ ms}$).

#### 3.1. Header Điều Khiển & Bộ Đệm Tĩnh (`firmware/src/swim_tinyml_core.h`)

```cpp
#ifndef SWIM_TINYML_CORE_H_
#define SWIM_TINYML_CORE_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hằng số kích thước cảm biến và cửa sổ trượt
#define IMU_CHANNELS          6      // [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
#define SLIDING_WINDOW_SIZE   128    // 128 mẫu @ 50Hz = 2.56 giây
#define SLIDING_WINDOW_STEP   64     // 50% Overlap = 64 mẫu (1.28 giây)
#define TOTAL_INPUT_ELEMENTS  (SLIDING_WINDOW_SIZE * IMU_CHANNELS)

// Cấu hình máy trạng thái an toàn
typedef enum {
    ALERT_NORMAL = 0,
    ALERT_LEVEL_1_YELLOW = 1,   // Cảnh báo rung nhẹ cổ tay (0 - 5s)
    ALERT_LEVEL_2_ORANGE = 2,   // Nguy cơ: Gửi LoRa bộc phát lên bờ (5 - 15s)
    ALERT_LEVEL_3_RED = 3       // Báo động đỏ: Hú còi, còi sóng âm thủy âm (> 15s)
} AlertState_t;

typedef struct {
    float mse_loss;
    float threshold_tau;
    float current_depth_m;
    bool is_surface_wrist;
    AlertState_t current_state;
    uint32_t inference_time_us;
} SwimmingTelemetry_t;

// Khai báo các API điều phối lõi
bool TinyML_System_Init(void);
void TinyML_Process_IMU_Sample(const float raw_imu[IMU_CHANNELS], float depth_m, bool wrist_out_water);
bool TinyML_Is_Window_Ready(void);
AlertState_t TinyML_Execute_Inference(SwimmingTelemetry_t* out_telemetry);

#ifdef __cplusplus
}
#endif

#endif // SWIM_TINYML_CORE_H_
```

#### 3.2. Hiện Thực Lõi TFLM & DSP (`firmware/src/swim_tinyml_core.cpp`)

```cpp
#include "swim_tinyml_core.h"
#include "model_data.h"

// TensorFlow Lite for Microcontrollers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <math.h>
#include <string.h>

// =========================================================================
// CẤP PHÁT BỘ NHỚ TĨNH (ZERO HEAP)
// =========================================================================
constexpr int kTensorArenaSize = 26 * 1024; // 26 KB Static RAM
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input = nullptr;
static TfLiteTensor* g_output = nullptr;

// Bộ đệm tròn quản lý 128 mẫu dữ liệu chuẩn hóa
static float g_window_buffer[SLIDING_WINDOW_SIZE][IMU_CHANNELS];
static uint16_t g_sample_count = 0;
static uint8_t g_consecutive_anomaly_count = 0;

// Hệ số bộ lọc số Butterworth bậc 2 (fc=10Hz, fs=50Hz)
static const float BUTTER_B[3] = {0.09763107f, 0.19526215f, 0.09763107f};
static const float BUTTER_A[2] = {-0.94280904f, 0.33333333f};

typedef struct {
    float x1, x2, y1, y2;
} ButterworthState_t;

static ButterworthState_t g_dsp_states[IMU_CHANNELS];

// Thống kê chuẩn hóa Z-Score
static const float AXIS_MEANS[IMU_CHANNELS] = {0.12f, -0.45f, 0.88f, 0.01f, -0.02f, 0.05f};
static const float AXIS_STDS[IMU_CHANNELS]  = {1.34f,  1.52f, 1.21f, 1.15f,  1.08f, 0.95f};

// Ngưỡng phát hiện bất thường thực nghiệm (tính từ Python script)
static const float DYNAMIC_TAU = 0.0585f;

// =========================================================================
// HÀM KHỞI TẠO MÔ HÌNH VÀ BỘ SUY LUẬN
// =========================================================================
bool TinyML_System_Init(void) {
    memset(g_dsp_states, 0, sizeof(g_dsp_states));
    g_sample_count = 0;
    g_consecutive_anomaly_count = 0;

    // 1. Ánh xạ mô hình từ Flash ROM
    g_model = tflite::GetModel(g_swimming_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        return false;
    }

    // 2. Khởi tạo Resolver chứa các toán tử Conv1D và TransposeConv
    static tflite::AllOpsResolver resolver;

    // 3. Khởi tạo Interpreter trên vùng nhớ tĩnh
    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, tensor_arena, kTensorArenaSize
    );
    g_interpreter = &static_interpreter;

    // 4. Phân bổ tensor arena
    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        return false;
    }

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);
    return true;
}

// =========================================================================
// XỬ LÝ TÍN HIỆU DSP TRÊN TỪNG MẪU (50 HZ)
// =========================================================================
void TinyML_Process_IMU_Sample(const float raw_imu[IMU_CHANNELS], float depth_m, bool wrist_out_water) {
    if (g_sample_count >= SLIDING_WINDOW_SIZE) {
        return;
    }

    for (int c = 0; c < IMU_CHANNELS; c++) {
        // Bộ lọc số thông thấp Butterworth bậc 2
        float x = raw_imu[c];
        float y = BUTTER_B[0] * x + BUTTER_B[1] * g_dsp_states[c].x1 + BUTTER_B[2] * g_dsp_states[c].x2
                - BUTTER_A[0] * g_dsp_states[c].y1 - BUTTER_A[1] * g_dsp_states[c].y2;

        g_dsp_states[c].x2 = g_dsp_states[c].x1;
        g_dsp_states[c].x1 = x;
        g_dsp_states[c].y2 = g_dsp_states[c].y1;
        g_dsp_states[c].y1 = y;

        // Chuẩn hóa Z-Score
        float normalized = (y - AXIS_MEANS[c]) / AXIS_STDS[c];
        g_window_buffer[g_sample_count][c] = normalized;
    }
    g_sample_count++;
}

bool TinyML_Is_Window_Ready(void) {
    return (g_sample_count >= SLIDING_WINDOW_SIZE);
}

// =========================================================================
// SUY LUẬN MÔ HÌNH VÀ PHÂN CẤP AN TOÀN
// =========================================================================
AlertState_t TinyML_Execute_Inference(SwimmingTelemetry_t* out_telemetry) {
    // 1. Lượng tử hóa Affine Float32 -> INT8: q = round(r / S) + Z
    float in_scale = g_input->params.scale;
    int32_t in_zero_point = g_input->params.zero_point;
    int8_t* in_ptr = g_input->data.int8;

    int element_idx = 0;
    for (int t = 0; t < SLIDING_WINDOW_SIZE; t++) {
        for (int c = 0; c < IMU_CHANNELS; c++) {
            float val = g_window_buffer[t][c];
            int32_t q = (int32_t)roundf(val / in_scale) + in_zero_point;
            q = (q < -128) ? -128 : (q > 127) ? 127 : q;
            in_ptr[element_idx++] = (int8_t)q;
        }
    }

    // 2. Chạy suy luận TFLM
    TfLiteStatus status = g_interpreter->Invoke();
    if (status != kTfLiteOk) {
        return ALERT_NORMAL;
    }

    // 3. Tính toán Sai số Tái tạo toàn phương trung bình (MSE)
    float out_scale = g_output->params.scale;
    int32_t out_zero_point = g_output->params.zero_point;
    int8_t* out_ptr = g_output->data.int8;

    float sum_squared_diff = 0.0f;
    for (int i = 0; i < TOTAL_INPUT_ELEMENTS; i++) {
        float original = (in_ptr[i] - in_zero_point) * in_scale;
        float reconstructed = (out_ptr[i] - out_zero_point) * out_scale;
        float diff = original - reconstructed;
        sum_squared_diff += (diff * diff);
    }
    float mse = sum_squared_diff / (float)TOTAL_INPUT_ELEMENTS;

    // 4. Thuật toán Biểu quyết Đa Cửa sổ (Temporal Voting)
    if (mse > DYNAMIC_TAU) {
        g_consecutive_anomaly_count++;
    } else {
        if (g_consecutive_anomaly_count > 0) g_consecutive_anomaly_count--;
    }

    // 5. Xác định trạng thái cảnh báo 3 cấp
    AlertState_t state = ALERT_NORMAL;
    if (g_consecutive_anomaly_count >= 3) {
        if (g_consecutive_anomaly_count >= 8) {
            state = ALERT_LEVEL_3_RED;
        } else if (g_consecutive_anomaly_count >= 5) {
            state = ALERT_LEVEL_2_ORANGE;
        } else {
            state = ALERT_LEVEL_1_YELLOW;
        }
    }

    // 6. Dịch cửa sổ trượt (50% Overlap)
    for (int t = 0; t < (SLIDING_WINDOW_SIZE - SLIDING_WINDOW_STEP); t++) {
        for (int c = 0; c < IMU_CHANNELS; c++) {
            g_window_buffer[t][c] = g_window_buffer[t + SLIDING_WINDOW_STEP][c];
        }
    }
    g_sample_count = SLIDING_WINDOW_SIZE - SLIDING_WINDOW_STEP;

    if (out_telemetry) {
        out_telemetry->mse_loss = mse;
        out_telemetry->threshold_tau = DYNAMIC_TAU;
        out_telemetry->current_state = state;
    }
    return state;
}
```

---

### 4. THIẾT LẬP MÔ PHỎNG RENODE (HARDWARE-IN-THE-LOOP SIMULATION)

Để chạy kiểm thử mô hình và mã nguồn mà không cần phần cứng thật, sử dụng công cụ mô phỏng chuẩn nhúng **Renode**:

#### 4.1. File Kịch Bản Cấu Hình Nền Tảng (`scripts/nrf52840_swim.resc`)

```resc
# Renode Script: Mô phỏng Thiết bị đeo AIoT Phát hiện Đuối nước
using sysbus

# 1. Khởi tạo máy ảo Cortex-M4F
mach create "swimming_wearable_node"
machine LoadPlatformDescription @platforms/cpus/nrf52840.repl

# 2. Định tuyến cổng giao tiếp nối tiếp Serial / UART
showAnalyzer uart0

# 3. Gán file nhị phân firmware đã biên dịch
sysbus LoadELF @build/zephyr/zephyr.elf

# 4. Thiết lập điểm ngắt kiểm tra độ trễ suy luận
sysbus.cpu AddHook `sysbus.cpu GetSymbolAddress "TinyML_Execute_Inference"` "print '>> [BENCHMARK] Bắt đầu thực thi suy luận 1D-CNN INT8...'"

# 5. Khởi động môi trường thời gian thực
start
```

---

### 5. MASTER PROMPT CHO ANTIGRAVITY IDE (AGENT CODE GENERATION)

Sao chép toàn bộ khối lệnh dưới đây vào khung chat của **Antigravity IDE** hoặc Coding Agent để tự động tạo toàn bộ mã nhúng:

```markdown
### MASTER PROMPT: TRIỂN KHAI MÃ NHÚNG TINYML C++ TRÊN NRF52840

Học vị: Tiến Sĩ, Kỹ Sư Trí Tuệ Nhân Tạo & Hệ Thống Nhúng Cấp Cao.
Nhiệm vụ: Hiện thực hóa firmware biên C++ cho thiết bị đeo theo đề tài "XÂY DỰNG THIẾT KẾ MÔ HÌNH AIoT PHÁT HIỆN BẤT THƯỜNG KHI BƠI (TINYML)".

YÊU CẦU THỰC HIỆN:
1. Tiếp nhận file mô hình đã lượng tử hóa INT8 từ `model_data.h` (mảng nhị phân `g_swimming_model_data`, căn chỉnh địa chỉ alignas(16)).
2. Triển khai kiến trúc TFLite Micro trên vi điều khiển Nordic nRF52840 (ARM Cortex-M4F, CMSIS-NN).
3. Thiết lập vùng nhớ tĩnh `tensor_arena` đúng 26 KB (Zero dynamic memory allocation).
4. Viết chuỗi tiền xử lý DSP trực tiếp bằng C++:
   - Bộ lọc số Butterworth thông thấp bậc 2 ($f_c = 10\text{ Hz}$, $f_s = 50\text{ Hz}$).
   - Chuẩn hóa phân phối Z-Score tĩnh trên 6 trục IMU.
   - Quản lý cửa sổ trượt (Sliding Window) 128 mẫu với độ chống lấp 50% (bước trượt 64 mẫu).
5. Hiện thực thuật toán tính sai số tái tạo toàn phương trung bình $\mathcal{L}_{MSE}$ từ đầu ra giải mã INT8:
   - Dequantize đầu ra và so sánh trực tiếp với tensor đầu vào.
   - Bộ biểu quyết 3 cửa sổ liên tiếp ($N=3$) vượt ngưỡng động $\tau = 0.0585$.
6. Điều khiển máy trạng thái an toàn 3 cấp:
   - Cấp 1 (0-5s): Rung haptic cổ tay.
   - Cấp 2 (5-15s): Bắn gói tin LoRa 433MHz bộc phát (<15ms) khi tay nhô khỏi mặt nước (Opportunistic Surface Burst).
   - Cấp 3 (>15s): Kích hoạt còi sóng âm thủy âm 3-5 kHz kết hợp hú còi trạm bờ.
7. Tạo tập lệnh kiểm thử mô phỏng trên nền tảng Renode (.resc).

Mã nguồn phải hoàn chỉnh, có type-hinting/kiểu dữ liệu chuẩn C++11, tối ưu hóa triệt để chu kỳ xung nhịp và sẵn sàng nạp thẳng vào nRF Connect SDK / Arduino IDE.
```
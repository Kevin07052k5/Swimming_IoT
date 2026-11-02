#include "swim_tinyml_core.h"
#include "model_data.h"

// TensorFlow Lite for Microcontrollers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <math.h>
#include <string.h>

// =========================================================================
// CẤP PHÁT BỘ NHỚ TĨNH (ZERO HEAP DYNAMIC ALLOCATION)
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

// Hệ số bộ lọc số Butterworth thông thấp bậc 2 (fc=10Hz, fs=50Hz)
static const float BUTTER_B[3] = {0.09763107f, 0.19526215f, 0.09763107f};
static const float BUTTER_A[2] = {-0.94280904f, 0.33333333f};

typedef struct {
    float x1, x2, y1, y2;
} ButterworthState_t;

static ButterworthState_t g_dsp_states[IMU_CHANNELS];

// Thống kê chuẩn hóa Z-Score mặc định (được cập nhật theo tập huấn luyện)
static const float AXIS_MEANS[IMU_CHANNELS] = {0.12f, -0.45f, 0.88f, 0.01f, -0.02f, 0.05f};
static const float AXIS_STDS[IMU_CHANNELS]  = {1.34f,  1.52f, 1.21f, 1.15f,  1.08f, 0.95f};

// Ngưỡng phát hiện bất thường thực nghiệm (từ DYNAMIC_ANOMALY_THRESHOLD hoặc mặc định)
#ifndef DYNAMIC_ANOMALY_THRESHOLD
#define DYNAMIC_ANOMALY_THRESHOLD (0.0585f)
#endif
static const float DYNAMIC_TAU = DYNAMIC_ANOMALY_THRESHOLD;

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

    // 3. Khởi tạo Interpreter trên vùng nhớ tĩnh tensor_arena
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
        float normalized = (y - AXIS_MEANS[c]) / (AXIS_STDS[c] + 1e-8f);
        g_window_buffer[g_sample_count][c] = normalized;
    }
    g_sample_count++;
}

bool TinyML_Is_Window_Ready(void) {
    return (g_sample_count >= SLIDING_WINDOW_SIZE);
}

// =========================================================================
// SUY LUẬN MÔ HÌNH VÀ PHÂN CẤP AN TOÀN FSM
// =========================================================================
AlertState_t TinyML_Execute_Inference(SwimmingTelemetry_t* out_telemetry) {
    if (!g_interpreter || !g_input || !g_output) {
        return ALERT_NORMAL;
    }

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


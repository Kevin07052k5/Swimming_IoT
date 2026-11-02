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

// Cấu hình máy trạng thái an toàn 3 cấp
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

// Khai báo các API điều phối lõi TinyML
bool TinyML_System_Init(void);
void TinyML_Process_IMU_Sample(const float raw_imu[IMU_CHANNELS], float depth_m, bool wrist_out_water);
bool TinyML_Is_Window_Ready(void);
AlertState_t TinyML_Execute_Inference(SwimmingTelemetry_t* out_telemetry);

#ifdef __cplusplus
}
#endif

#endif // SWIM_TINYML_CORE_H_


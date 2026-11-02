"""
src/gateway_service.py
Dịch vụ trạm bờ hồ Raspberry Pi CM4 (Edge Gateway):
1. Điều phối bản tin MQTT Broker cục bộ (Mosquitto)
2. Phát cảnh báo đa kênh 3 cấp:
   - Còi hú công suất lớn và đèn chớp bờ hồ (swim/alert/local)
   - Xung rung haptic Smartwatch cứu hộ (swim/alert/haptic)
   - Cập nhật bản đồ nhiệt 2D thời gian thực Web Dashboard (swim/alert/ui)
   - Bắn SMS 4G khẩn cấp SIM7600 dự phòng (swim/alert/sms)
3. Hỗ trợ chế độ Standalone Mock Test khi không có broker vật lý.
"""
import os
import sys
import json
import time
from typing import Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from src.config import CONFIG

try:
    import paho.mqtt.client as mqtt
    HAS_PAHO_MQTT = True
except ImportError:
    HAS_PAHO_MQTT = False

class PoolsideGatewayService:
    def __init__(self, broker_host: str = CONFIG.MQTT_BROKER_HOST, broker_port: int = CONFIG.MQTT_BROKER_PORT):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client: Optional[Any] = None
        self.is_connected = False

        if HAS_PAHO_MQTT:
            try:
                # Tương thích cả paho-mqtt v1 và v2
                self.client = mqtt.Client(client_id="EdgeGateway_CM4", clean_session=True)
                self.client.on_connect = self.on_connect
                self.client.on_message = self.on_message
            except Exception:
                try:
                    self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="EdgeGateway_CM4")
                    self.client.on_connect = self.on_connect
                    self.client.on_message = self.on_message
                except Exception as e:
                    print(f"[!] Không thể khởi tạo MQTT client: {e}")
                    self.client = None

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            print(f"[+] Gateway kết nối MQTT Broker thành công: {self.broker_host}:{self.broker_port}")
            self.client.subscribe(f"{CONFIG.TOPIC_TELEMETRY}/#", qos=1)
            print(f"    - Đã đăng ký lắng nghe kênh: {CONFIG.TOPIC_TELEMETRY}/#")
        else:
            print(f"[!] Lỗi kết nối MQTT Broker, mã lỗi: {rc}")

    def trigger_poolside_alarm(self, node_id: str, lane_id: int):
        """Kích hoạt còi hú & đèn chớp bờ hồ (Độ trễ < 0.2s)"""
        payload = json.dumps({
            "action": "SIREN_ON",
            "alert_level": "LEVEL_3_RED",
            "node_id": node_id,
            "lane_id": lane_id,
            "timestamp": time.time()
        })
        if self.client and self.is_connected:
            self.client.publish(CONFIG.TOPIC_ALERT_LOCAL, payload, qos=2)
        print(f"[!] [LOCAL SIREN] BẬT CÒI HÚ CÔNG SUẤT LỚN & ĐÈN CHỚP TẠI BỜ HỒ! (Làn {lane_id} - Node {node_id})")

    def notify_lifeguard_smartwatch(self, node_id: str, lane_id: int, level: str):
        """Gửi xung rung tới Smartwatch cứu hộ (Độ trễ < 0.5s)"""
        pattern = "HIGH_PULSE_CONTINUOUS" if level == "ĐỎ" else "DOUBLE_PULSE"
        payload = json.dumps({
            "vibration": "HIGH",
            "pattern": pattern,
            "level": level,
            "lane_id": lane_id,
            "node_id": node_id,
            "timestamp": time.time()
        })
        if self.client and self.is_connected:
            self.client.publish(CONFIG.TOPIC_ALERT_HAPTIC, payload, qos=2)
        print(f"[!] [SMARTWATCH CỨU HỘ] Gửi xung rung khẩn cấp: Mức {level} | Làn số {lane_id} | Thiết bị: {node_id}")

    def send_fail_safe_sms(self, node_id: str, lane_id: int, depth_m: float):
        """Điều khiển module SIM7600 bắn tin nhắn SMS khẩn cấp tới Đội trưởng Y tế (2 - 4s)"""
        sms_body = (
            f"[SOS KHẨN CẤP] PHÁT HIỆN ĐUỐI NƯỚC CẤP 3!\n"
            f"- Làn bơi: Số {lane_id}\n"
            f"- Vòng đeo: {node_id}\n"
            f"- Độ sâu chìm: {depth_m:.2f} m\n"
            f"- Còi bờ & cứu hộ đã kích hoạt!"
        )
        payload = json.dumps({
            "action": "SEND_SMS",
            "recipient": "HEAD_LIFEGUARD",
            "message": sms_body,
            "timestamp": time.time()
        })
        if self.client and self.is_connected:
            self.client.publish(CONFIG.TOPIC_ALERT_SMS, payload, qos=1)
        print(f"[!] [SMS 4G FAIL-SAFE] Gửi SMS SIM7600 tới Đội trưởng Cứu hộ:\n    \"{sms_body.replace(chr(10), ' ')}\"")

    def process_telemetry_event(self, payload: Dict[str, Any]):
        """
        Xử lý sự kiện telemetry nhận được (từ MQTT hoặc từ hàm gọi giả lập).
        """
        node_id = payload.get("node_id", "SWIM_NODE_01")
        lane_id = payload.get("lane_id", 1)
        state = payload.get("state", "NORMAL")
        depth_m = payload.get("depth_m", 0.0)
        timer = payload.get("state_timer_sec", 0.0)

        # Điều phối phân cấp cảnh báo 3 cấp
        if state == "LEVEL_1_YELLOW":
            print(f"[i] [CẤP 1 - VÀNG] Vòng {node_id} (Làn {lane_id}): Dị thường mức nhẹ ({timer:.1f}s), rung haptic tại vòng.")
        elif state == "LEVEL_2_ORANGE":
            print(f"[!] [CẤP 2 - CAM] BẬT ĐÈN CAM TRẠM BỜ LÀN {lane_id}! Rung Smartwatch cứu hộ.")
            self.notify_lifeguard_smartwatch(node_id, lane_id, "CAM")
        elif state == "LEVEL_3_RED":
            print(f"[ALERT RED] BÁO ĐỘNG CẤP 3 - ĐUỐI NƯỚC XÁC NHẬN TẠI LÀN {lane_id}! (Độ sâu {depth_m}m)")
            self.trigger_poolside_alarm(node_id, lane_id)
            self.notify_lifeguard_smartwatch(node_id, lane_id, "ĐỎ")
            self.send_fail_safe_sms(node_id, lane_id, depth_m)

        # Cập nhật bản đồ nhiệt lên Web Dashboard
        if self.client and self.is_connected:
            self.client.publish(CONFIG.TOPIC_ALERT_UI, json.dumps(payload), qos=0)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            self.process_telemetry_event(payload)
        except Exception as e:
            print(f"[!] Lỗi phân tích gói tin MQTT: {e}")

    def run(self):
        """Khởi chạy lắng nghe MQTT liên tục"""
        if not HAS_PAHO_MQTT or not self.client:
            print("[!] Thư viện paho-mqtt chưa khả dụng, chạy chế độ Standalone Mock.")
            return

        try:
            print(f"[*] Gateway đang kết nối tới Mosquitto Broker {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_forever()
        except ConnectionRefusedError:
            print(f"[!] Không thể kết nối tới Broker {self.broker_host}:{self.broker_port}. Hãy đảm bảo dịch vụ Mosquitto đang chạy.")
        except KeyboardInterrupt:
            print("[*] Đang dừng Gateway Service...")
            self.client.disconnect()

if __name__ == "__main__":
    gateway = PoolsideGatewayService()
    # Chạy thử nghiệm phân giải sự kiện mẫu
    print("--- CHẠY THỬ NGHIỆM ĐIỀU PHỐI CẢNH BÁO GATEWAY RPI CM4 ---")
    mock_event_yellow = {"node_id": "NODE_LANE_3", "lane_id": 3, "state": "LEVEL_1_YELLOW", "depth_m": 0.45, "state_timer_sec": 3.2}
    mock_event_orange = {"node_id": "NODE_LANE_3", "lane_id": 3, "state": "LEVEL_2_ORANGE", "depth_m": 0.95, "state_timer_sec": 8.5}
    mock_event_red    = {"node_id": "NODE_LANE_3", "lane_id": 3, "state": "LEVEL_3_RED",    "depth_m": 1.40, "state_timer_sec": 16.8}

    gateway.process_telemetry_event(mock_event_yellow)
    print()
    gateway.process_telemetry_event(mock_event_orange)
    print()
    gateway.process_telemetry_event(mock_event_red)

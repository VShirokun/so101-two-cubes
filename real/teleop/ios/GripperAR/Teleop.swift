import Foundation
import Combine

struct Block: Decodable { let ang: Double; let w: Double }
struct Limits: Decodable { let x: Bool; let y: Bool; let z: Bool; let r: Bool; let ik: Bool }
struct Status: Decodable {
    let tcp: [Double]; let yaw_deg: Double; let pitch_deg: Double; let grip_pct: Double
    let limits: Limits; let blocks: [Block]; let up: Double; let down: Double
    let msg: String; let alive: Bool; let stop: Bool
}

/// WebSocket-клиент телеоперации: шлёт позу телефона относительно якоря 20 раз в секунду,
/// получает статус (координаты, стены зоны, сообщения охраны).
final class Teleop: ObservableObject {
    @Published var status: Status?
    @Published var connected = false
    @Published var following = false
    var grip: Double? = nil                     // 0..1, отправляется один раз при изменении
    private var task: URLSessionWebSocketTask?
    private var timer: Timer?
    private var anchor: (pos: SIMD3<Float>, yaw: Float, down: Float, fwd: SIMD3<Float>)?
    private var pose: (pos: SIMD3<Float>, yaw: Float, down: Float) = (.zero, 0, 0)
    @Published var scale: Double = 1.0          // 1:1 или 1:2
    var baseURL: String = ""
    var key: String = ""

    func connect(base: String, key: String) {
        baseURL = base; self.key = key
        let ws = base.replacingOccurrences(of: "https://", with: "wss://").replacingOccurrences(of: "http://", with: "ws://")
        guard let url = URL(string: "\(ws)/ws?k=\(key)") else { return }
        task?.cancel()
        task = URLSession.shared.webSocketTask(with: url)
        task?.resume()
        connected = true
        receive()
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in self?.send() }
    }

    func update(pos: SIMD3<Float>, yaw: Float, down: Float) { pose = (pos, yaw, down) }

    func setFollow(_ on: Bool) {
        following = on
        anchor = nil
    }

    private func send() {
        var msg: [String: Any] = ["mode": "xr", "engaged": following]
        if following {
            if anchor == nil {
                let f = SIMD3<Float>(-sinf(pose.yaw), 0, -cosf(pose.yaw))
                anchor = (pose.pos, pose.yaw, pose.down, f)
            }
            if let a = anchor {
                let d = pose.pos - a.pos
                let left = SIMD3<Float>(-a.fwd.z, 0, a.fwd.x)
                var dyaw = pose.yaw - a.yaw
                while dyaw > .pi { dyaw -= 2 * .pi }
                while dyaw < -.pi { dyaw += 2 * .pi }
                msg["dx"] = Double(d.x * a.fwd.x + d.z * a.fwd.z)   // вперёд телефона
                msg["dy"] = Double(d.x * left.x + d.z * left.z)     // влево телефона
                msg["dz"] = Double(d.y)                             // вверх
                msg["dyaw"] = Double(dyaw)
                msg["dpitch"] = Double(pose.down - a.down)          // наклонили ниже -> подход вертикальнее
                msg["scale"] = scale
            }
        }
        if let g = grip { msg["grip"] = g; grip = nil }
        if let data = try? JSONSerialization.data(withJSONObject: msg), let s = String(data: data, encoding: .utf8) {
            task?.send(.string(s)) { [weak self] err in if err != nil { DispatchQueue.main.async { self?.connected = false } } }
        }
    }

    func command(_ cmd: String) {
        task?.send(.string("{\"cmd\":\"\(cmd)\"}")) { _ in }
    }

    private func receive() {
        task?.receive { [weak self] result in
            guard let self = self else { return }
            if case .success(let m) = result, case .string(let s) = m, let data = s.data(using: .utf8),
               let st = try? JSONDecoder().decode(Status.self, from: data) {
                DispatchQueue.main.async { self.status = st; self.connected = true }
            } else if case .failure = result {
                DispatchQueue.main.async { self.connected = false }
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.connect(base: self.baseURL, key: self.key) }
                return
            }
            self.receive()
        }
    }
}

import ARKit
import Combine
import simd

/// Поза iPhone из ARKit (world tracking): положение и «вперёд» задней камеры.
/// ARKit: y вверх, камера смотрит вдоль -Z своей системы. Те же формулы, что в xr.html.
final class ARPose: NSObject, ObservableObject, ARSessionDelegate {
    @Published var position = SIMD3<Float>(0, 0, 0)
    @Published var yaw: Float = 0            // рад, вокруг вертикали; влево = +
    @Published var down: Float = 0           // рад, насколько задняя камера смотрит вниз
    @Published var tracking = "нет трекинга"
    let session = ARSession()

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        let cfg = ARWorldTrackingConfiguration()
        cfg.worldAlignment = .gravity
        session.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() { session.pause() }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let m = frame.camera.transform
        position = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        let f = -SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z)   // взгляд камеры
        yaw = atan2f(-f.x, -f.z)
        down = asinf(max(-1, min(1, -f.y)))
        switch frame.camera.trackingState {
        case .normal: tracking = "трекинг ок"
        case .limited(let r): tracking = "трекинг ограничен: \(r)"
        case .notAvailable: tracking = "нет трекинга"
        }
    }
}

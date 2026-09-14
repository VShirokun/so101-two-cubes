import SwiftUI

/// Подсветка границы зоны: красное свечение у края кадра в направлении стены (0° = вправо, 90° = вверх),
/// сила w; полоска справа — потолок (сверху) и пол (снизу). Та же логика, что в web_teleop.html.
struct GlowOverlay: View {
    let blocks: [Block]; let allBlocked: Bool; let up: Double; let down: Double
    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height, R = hypot(w, h) / 2
            Canvas { ctx, _ in
                if allBlocked {
                    let g = Gradient(colors: [.red.opacity(0), .red.opacity(0.55)])
                    ctx.fill(Path(CGRect(origin: .zero, size: geo.size)), with: .radialGradient(g, center: CGPoint(x: w/2, y: h/2), startRadius: R*0.55, endRadius: R))
                }
                for b in blocks {
                    let a = b.ang * .pi / 180
                    let p = CGPoint(x: w/2 + cos(a) * R * 1.05, y: h/2 - sin(a) * R * 1.05)
                    let g = Gradient(stops: [.init(color: .red.opacity(0.85*b.w), location: 0), .init(color: .red.opacity(0.35*b.w), location: 0.55), .init(color: .red.opacity(0), location: 1)])
                    ctx.fill(Path(CGRect(origin: .zero, size: geo.size)), with: .radialGradient(g, center: p, startRadius: 0, endRadius: R*0.95))
                }
            }
            .overlay(alignment: .trailing) {
                LinearGradient(stops: [.init(color: .red.opacity(up), location: 0), .init(color: .clear, location: 0.3), .init(color: .clear, location: 0.7), .init(color: .red.opacity(down), location: 1)], startPoint: .top, endPoint: .bottom)
                    .frame(width: 10).cornerRadius(5).padding(8)
            }
        }
        .allowsHitTesting(false)
    }
}

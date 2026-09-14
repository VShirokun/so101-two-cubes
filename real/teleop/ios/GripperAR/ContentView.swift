import SwiftUI

struct ContentView: View {
    @AppStorage("base") private var base = "https://marker-temporarily-beta-easy.trycloudflare.com"
    @AppStorage("key") private var key = ""
    @StateObject private var ar = ARPose()
    @StateObject private var tele = Teleop()
    @State private var grip: Double = 0
    @State private var started = false

    var body: some View {
        if !started {
            VStack(spacing: 14) {
                Text("Телефон как схват").font(.title2.bold())
                Text("Держите iPhone экраном к себе. После «Следовать» смещение и поворот телефона повторяет схват 1:1.")
                    .foregroundStyle(.secondary).multilineTextAlignment(.center)
                TextField("адрес сервера", text: $base).textFieldStyle(.roundedBorder).autocapitalization(.none).disableAutocorrection(true)
                TextField("ключ (k=…)", text: $key).textFieldStyle(.roundedBorder).autocapitalization(.none).disableAutocorrection(true)
                Button("Старт") { ar.start(); tele.connect(base: base, key: key); started = true }
                    .buttonStyle(.borderedProminent).controlSize(.large)
            }.padding(24)
        } else {
            VStack(spacing: 8) {
                ZStack(alignment: .topLeading) {
                    MJPEGView(url: "\(base)/cam/wrist.mjpg?k=\(key)")
                    if let s = tele.status { GlowOverlay(blocks: s.blocks, allBlocked: s.limits.ik && s.blocks.isEmpty, up: s.up, down: s.down) }
                    Text(hud).font(.system(size: 11, design: .monospaced)).padding(4).background(.black.opacity(0.45)).foregroundStyle(.white).cornerRadius(4).padding(6)
                }.frame(maxWidth: .infinity, maxHeight: .infinity).background(.black).cornerRadius(8)
                HStack { Text("схват").font(.caption).foregroundStyle(.secondary)
                    Slider(value: $grip, in: 0...1) { _ in tele.grip = grip }.onChange(of: grip) { v in tele.grip = v }
                    Text("\(Int(grip*100)) %").font(.caption).foregroundStyle(.secondary).frame(width: 44, alignment: .trailing) }
                HStack(spacing: 10) {
                    Button(tele.following ? "Следую" : "Следовать") { tele.setFollow(!tele.following) }
                        .buttonStyle(.borderedProminent).tint(tele.following ? .green : .gray).frame(maxWidth: .infinity)
                    Button(tele.status?.stop == true ? "Продолжить" : "СТОП") { tele.command(tele.status?.stop == true ? "resume" : "stop") }
                        .buttonStyle(.borderedProminent).tint(.red).frame(maxWidth: .infinity)
                    Button(tele.scale == 1 ? "1:1" : "1:2") { tele.scale = tele.scale == 1 ? 0.5 : 1 }.buttonStyle(.bordered)
                    Button("Домой") { tele.setFollow(false); tele.command("home") }.buttonStyle(.bordered).frame(maxWidth: .infinity)
                }
            }
            .padding(8)
            .onReceive(ar.$position) { p in tele.update(pos: p, yaw: ar.yaw, down: ar.down) }
        }
    }

    private var hud: String {
        guard let s = tele.status else { return tele.connected ? "ждём статус…" : "нет связи" }
        let t = s.tcp.map { String(format: "%.3f", $0) }.joined(separator: " ")
        return "\(t) м · yaw \(Int(s.yaw_deg))° наклон \(Int(s.pitch_deg))° · \(s.alive ? "связь" : "НЕТ СВЯЗИ")\(s.stop ? " · СТОП" : "")\(s.msg.isEmpty ? "" : " · " + s.msg) · \(ar.tracking)"
    }
}

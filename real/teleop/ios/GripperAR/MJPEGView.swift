import SwiftUI
import WebKit

/// Поток камеры кисти: WKWebView с одной картинкой multipart/x-mixed-replace.
struct MJPEGView: UIViewRepresentable {
    let url: String
    func makeUIView(context: Context) -> WKWebView {
        let v = WKWebView()
        v.isOpaque = false; v.backgroundColor = .black; v.scrollView.isScrollEnabled = false
        v.loadHTMLString("<html><body style='margin:0;background:#000'><img src='\(url)' style='width:100%;height:100vh;object-fit:contain'></body></html>", baseURL: nil)
        return v
    }
    func updateUIView(_ uiView: WKWebView, context: Context) {}
}

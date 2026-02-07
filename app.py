import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
from datetime import datetime
from fpdf import FPDF

class MyCNN(nn.Module):
    def __init__(self, in_channel, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*64, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN(in_channel=3, dropout=0.4)
try:
    model.load_state_dict(torch.load('model_epoch_10.pth', map_location=device))
    model.to(device).eval()
except:
    pass 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_technical_report(verdict, real_conf, ai_conf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", 'B', 16)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(200, 10, txt="FORENSIC AUTHENTICITY ANALYSIS DATASHEET", ln=True, align='L')
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, 22, 200, 22)
    pdf.ln(10)
    pdf.set_font("Courier", size=10)
    pdf.cell(200, 5, txt=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)
    pdf.cell(200, 5, txt=f"Neural Engine: MyCNN-v1.0", ln=True)
    pdf.ln(10)
    pdf.set_font("Courier", 'B', 12)
    pdf.cell(200, 10, txt=f"ANALYSIS VERDICT: {verdict}", ln=True)
    pdf.set_font("Courier", size=10)
    pdf.cell(200, 7, txt=f"Probability Real: {real_conf:.4f}", ln=True)
    pdf.cell(200, 7, txt=f"Probability AI:   {ai_conf:.4f}", ln=True)
    filename = f"Forensic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

def analyze_image(image):
    if image is None: return None, "", gr.update(visible=False)
    img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).item()
    real_conf, ai_conf = prob, 1 - prob
    verdict_text = "REAL" if real_conf > 0.5 else "AI GENERATED"
    verdict_color = "#ffffff" if real_conf > 0.5 else "#f4812a"
    verdict_html = f"""
    <div style="text-align:center;">
        <p style="color:#888; font-size:10px; letter-spacing:2px; margin:0;">VERDICT</p>
        <h2 style="color:{verdict_color}; font-size:28px; font-weight:800; margin:0; font-family:'Rajdhani';">{verdict_text}</h2>
    </div>
    """
    report_path = generate_technical_report(verdict_text, real_conf, ai_conf)
    return {"Real Photo": real_conf, "AI Generated": ai_conf}, verdict_html, gr.update(value=report_path, visible=True)

def clear_all():
    return None, None, "", gr.update(visible=False)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;700&display=swap');
body::-webkit-scrollbar { display: none; }
body { 
    -ms-overflow-style: none; 
    scrollbar-width: none; 
    background-color: #080a0c !important; 
    font-family: 'Rajdhani', sans-serif !important; 
}
.gradio-container { 
    background-color: #080a0c !important; 
    font-family: 'Rajdhani', sans-serif !important; 
}
#main-wrap { max-width: 1000px !important; margin: 40px auto !important; }
.outer-card {
    background: #101214 !important;
    border: 1px solid #1f2226 !important;
    border-radius: 4px !important;
    padding: 20px !important;
}
.inner-dotted {
    border: 1px dashed #333 !important;
    background: #0c0e10 !important;
    border-radius: 2px !important;
    padding: 15px !important;
    margin-top: 10px;
}
.header-area { text-align: center; margin-bottom: 30px; border-bottom: 1px solid #1f2226; padding-bottom: 30px; }
.header-area h1 { 
    color: #ffffff; 
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700; 
    letter-spacing: 3px; 
    font-size: 42px; 
    margin-bottom: 0px; 
    text-transform: uppercase; 
}
.tech-sub { 
    color: #aaaaaa; 
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 16px; 
    margin-top: 12px; 
    max-width: 800px; 
    margin-left: auto; 
    margin-right: auto; 
    line-height: 1.5; 
}
#submit-btn { background: #f4812a !important; color: #000000 !important; font-weight: 700 !important; border-radius: 2px !important; border: none !important; cursor: pointer; transition: 0.2s; font-family: 'Rajdhani' !important; }
#submit-btn:hover { background: #cccccc !important; }
#clear-btn { background: #1a1c1e !important; color: #ffffff !important; font-weight: 700 !important; border-radius: 2px !important; border: 1px solid #2d3135 !important; cursor: pointer; font-family: 'Rajdhani' !important; }
.guide-box { margin-top: 20px; padding: 15px; background: #101214; border-radius: 4px; border: 1px solid #1f2226; }
.guide-text { color: #555; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; font-family: 'Rajdhani' !important; }
.label-box span { display: none !important; }
.gradio-container .prose h2 { 
    margin: 0 !important; 
    color: #ffffff !important; 
    font-size: 16px !important; 
    letter-spacing: 1px; 
    text-transform: uppercase; 
    font-family: 'Rajdhani' !important;
}
.source-link { text-align: center; margin-top: 50px; }
.source-link a { 
    color: #ffffff; 
    text-decoration: none; 
    font-size: 20px; 
    letter-spacing: 1px; 
    transition: 0.3s; 
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 500;
}
.source-link a:hover { color: #f4812a; }
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_id="main-wrap"):
        gr.HTML("""
        <div class="header-area">
            <h1>AI Image Authentication</h1>
            <div class="tech-sub">
                Forensic Neural Analysis Interface
            </div>
        </div>
        """)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="outer-card"):
                gr.Markdown("## 01. SOURCE MATERIAL")
                with gr.Column(elem_classes="inner-dotted"):
                    input_img = gr.Image(label="Source", type="pil", show_label=False, container=False)
                with gr.Row():
                    clear_btn = gr.Button("CLEAR", elem_id="clear-btn")
                    submit_btn = gr.Button("SUBMIT ANALYSIS", elem_id="submit-btn")
            with gr.Column(scale=1, elem_classes="outer-card"):
                gr.Markdown("## 02. FORENSIC OUTPUT")
                with gr.Column(elem_classes="inner-dotted"):
                    out_verdict = gr.HTML("<div style='text-align:center; padding:15px; color:#444;'>Awaiting input analysis...</div>")
                    out_label = gr.Label(num_top_classes=2, label="Confidence", elem_classes="label-box")
                gr.HTML("<div style='height:15px;'></div>")
                report_file = gr.File(label="DATASHEET", visible=False)
        with gr.Row(elem_classes="guide-box"):
            gr.HTML("""
            <div style="display:flex; justify-content:space-around; width:100%; text-align:center;">
                <div class="guide-text">STEP 1: UPLOAD TARGET IMAGE</div>
                <div class="guide-text" style="color:#222;">|</div>
                <div class="guide-text">STEP 2: INITIATE CNN SCAN</div>
                <div class="guide-text" style="color:#222;">|</div>
                <div class="guide-text">STEP 3: EXPORT TECHNICAL PDF</div>
            </div>
            """)
        gr.HTML("""
        <div class="source-link">
            <a href="https://github.com/rkcode2025" target="_blank">SOURCE CODE</a>
        </div>
        """)
        gr.HTML("""
        <div style="text-align:center; margin-top:10px; border-top:1px solid #1f2226; padding-top:20px; color:#333; font-size:10px; letter-spacing:1px;">
            SYSTEM STATUS: OPERATIONAL • BUILD: 5.0.4 • HARDWARE: ACCELERATED
        </div>
        """)
    submit_btn.click(
        fn=analyze_image,
        inputs=input_img,
        outputs=[out_label, out_verdict, report_file],
        show_progress="full"
    )
    clear_btn.click(
        fn=clear_all,
        outputs=[input_img, out_label, out_verdict, report_file]
    )

if __name__ == "__main__":
    demo.launch()

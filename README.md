AI Image Authentication System
A forensic tool for detecting AI-generated imagery using a custom Convolutional Neural Network (CNN).

🚀 Features
AI Detection: Accurately classifies images as "REAL" or "AI GENERATED."

Forensic Reports: Generates a downloadable PDF datasheet for every analysis.

Modern UI: Sleek, high-tech interface with orange accents and Rajdhani typography.

Optimized: Supports CUDA acceleration and Mixed Precision training.

🛠 Setup
1. Requirements
Install the necessary Python packages:

Bash
pip install torch torchvision gradio fpdf datasets tqdm Pillow
2. Model Weights
Ensure your trained model file model_epoch_10.pth is located in the project root directory.

3. Running the App
Launch the web interface:

Bash
python main.py
🧠 Architecture
The system utilizes a custom 5-Layer CNN designed specifically to identify the frequency artifacts found in synthetic media.

Input: 224x224 RGB

Features: 5 blocks (Conv2D -> ReLU -> BatchNorm -> MaxPool)

Output: Sigmoid (0 = AI, 1 = Real)

📂 Project Structure
main.py - Gradio UI and inference logic.

train.py - Training script and dataset preprocessing.

Forensic_Report_*.pdf - Generated analysis outputs.

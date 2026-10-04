\# 🛡️ Deepfake Detection for Information Security



A multi-modal deepfake detection platform to identify synthetic media, altered audio, and deepfake videos designed to spread misinformation or bypass biometric identity checks.



\## 🎯 Problem Statement



Deepfakes are AI-generated media that convincingly fake real people. Used for:

\- Local misinformation campaigns

\- Biometric identity bypass

\- Non-consensual content creation

\- Financial fraud



\## 🏗️ Architecture



Input Media → Preprocessing → Multi-modal Detection → Fusion → Output



\## 📊 Results



| Metric | Value |

|--------|-------|

| Accuracy | 99.91% |

| Precision | 99.91% |

| Recall | 99.91% |

| F1 Score | 99.91% |

| AUC-ROC | 99.98% |



\## 🛠️ Tech Stack



\- PyTorch + EfficientNet-B0 (visual)

\- wav2vec2 (audio)

\- Grad-CAM (explainability)

\- FastAPI (backend)

\- HTML + TailwindCSS (frontend)



\## 🚀 Quick Start



```bash

pip install -r requirements.txt

python -m uvicorn portal.app:app --reload --port 8000


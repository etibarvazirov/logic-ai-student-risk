# 🎓 Educational Outcome Prediction (LTN vs Standard NN)

This project compares a **Standard Neural Network** and a **Logic Tensor Network (LTN)-based model**  
to predict student performance and identify at-risk learners early.

🔍 **Key Features**
- Dataset: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- Models: Standard NN vs LTN (NN + Logic Rules)
- Metrics: Accuracy, F1(FAIL), Confusion Matrix, Rule Penalty
- Visualized and deployed using **Streamlit**

🚀 **Live Demo:** [Click to open app](https://ltn-student-prediction.streamlit.app/) *(link aktivləşəcək uploaddan sonra)*

---

## 📦 How to Run Locally
```bash
git clone https://github.com/etibarvazirov/LTN-Student-Prediction.git
cd LTN-Student-Prediction
pip install -r requirements.txt
streamlit run app.py


---

## 🌐 4. GitHub-a yükləmək

1. GitHub-da yeni repo yarat → **LTN-Student-Prediction**
2. Kompüterdə bu qovluğu yarat və yuxarıdakı faylları ora at.
3. Terminalda bu əmrləri yaz:

```bash
git init
git add .
git commit -m "Initial commit: Streamlit LTN app"
git branch -M main
git remote add origin https://github.com/etibarvazirov/LTN-Student-Prediction.git
git push -u origin main

# 🔒 ClickArmor: Phishing URL Detection  

ClickArmor is a **machine learning-powered tool** that detects whether a given URL is **benign** ✅ or **phishing** 🚨.  
It leverages **LightGBM** and **custom feature engineering** to classify URLs in real-time.  

---

### 💻 End-to-End Pipeline  

ClickArmor implements a **complete pipeline**:

**1️⃣ Data Ingestion:** Raw URL datasets → train/test split  
**2️⃣ Data Transformation:** Feature engineering + sensitive words extraction  
**3️⃣ Model Training:** LightGBM classifier → optimized threshold  
**4️⃣ Prediction Pipeline:** Single URL → feature extraction → prediction  
**5️⃣ Deployment:** Streamlit app → future browser extension

---

## ✨ Features  
- 🧠 **Machine Learning Model** (LightGBM) for phishing detection  
- 🔍 **Feature Extraction** from raw URLs  
- ⚡ **Streamlit Web App** for interactive predictions  
- 📊 **High Accuracy** with probability-based confidence  
- 🚀 Easy to deploy and extend  

---

## 📂 Project Structure  
```
ClickArmor/
│── src/
│ ├── components/ # Data ingestion, transformation, model trainer
│ ├── pipeline/ # Prediction pipeline 
│ ├── utils.py # Utility functions
│ ├── exception.py # Custom exception handling
│ ├── logger.py # Logging Utility
│── artifacts/ # Saved models & transformers
│── app.py # Streamlit app
│── requirements.txt # Project dependencies
│── README.md # Project documentation
```

---

## 📊 Model Performance  

```
| Metric            | Value    |
|-------------------|----------|
| Accuracy          | 98.78%   |
| Recall (Phishing) | <0.7     |
 -------------------------------
 ```
Notes: Model optimized to **minimize false positives**, ensuring benign URLs are rarely flagged as phishing. High confidence predictions for phishing URLs only.


---

## ⚡ Installation  

Clone the repository and set up the environment:  

```bash
git clone https://github.com/yourusername/ClickArmor.git
cd ClickArmor

# Create virtual environment
python -m venv clickarmor_env

# Activate environment
# Linux/Mac
source clickarmor_env/bin/activate  
# Windows
clickarmor_env\Scripts\activate  

# Install dependencies
pip install -r requirements.txt
```
## ▶️ **Run the App**

Run the Streamlit app locally:

```bash
streamlit run app.py
```
Open your browser and navigate to http://localhost:8501
 🎉

## 🔮 **Future Improvements**
- 🌍 **Develop a browser extension** for real-time protection  
- 📈 **Add continuous model retraining** with fresh phishing datasets  
- 🔐 **Integrate with security dashboards** for enterprise use  

---

## 📜 **License**
This project is licensed under the **MIT License**.  

---

## 🤝 **Contributing**
Contributions are welcome! 🚀  

1. **Fork** the repo  
2. **Create a new branch** (`feature-xyz`)  
3. **Commit** your changes  
4. **Submit a Pull Request**  

---

## 🙌 **Acknowledgements**
- ⚡ **LightGBM**  
- 🎨 **Streamlit**  
- 🌐 **Open-source phishing datasets** used for  training  


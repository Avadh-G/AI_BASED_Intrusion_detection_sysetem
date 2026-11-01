# 🛡️ AI-Based Intrusion Detection System (CN-IDS)

### A Hybrid AI-Powered Network Intrusion Detection System

**CN-IDS** is an AI-powered Intrusion Detection System developed to identify suspicious and malicious network activities using Machine Learning, Deep Learning, anomaly detection, and custom network traffic simulation.

The project combines a Python-based network simulator, a hybrid AI detection engine, and an interactive Streamlit dashboard to demonstrate network monitoring, attack detection, security alerts, and simulated threat mitigation.

---

## 🚀 Key Features

- **Custom Network Simulation:** Simulates network devices, connections, and traffic flows.
- **Automated Traffic Generation:** Generates normal and malicious network traffic.
- **55 Network Flow Features:** Extracts network-level features for AI-based traffic analysis.
- **Hybrid AI Detection:** Integrates four specialized ML and Deep Learning models.
- **Anomaly Detection:** Identifies unusual traffic patterns and potential unknown threats.
- **Security Alert Management:** Generates alerts with severity levels and supporting evidence.
- **Simulated Firewall:** Supports IP blocking, ACL rules, and host isolation.
- **Interactive Dashboard:** Streamlit-based interface for monitoring and analysing network activity.
- **Model Evaluation:** Provides classification metrics, confusion matrices, and performance comparisons.

---

## 🧠 AI & Machine Learning Architecture

CN-IDS uses four specialized detection models:

| Model | Purpose |
|---|---|
| Random Forest | Supervised attack classification |
| XGBoost | Gradient-boosted attack classification |
| Isolation Forest | Unsupervised anomaly detection |
| Autoencoder | Learns normal traffic patterns through reconstruction |

The outputs are integrated through a hybrid detection engine to generate a unified security decision.

---

## 🌐 Network Simulation

The custom Python network simulator models network devices, connections, and traffic behaviour.

Supported simulated scenarios include:

- Normal network traffic
- Port Scanning
- Denial of Service (DoS)
- Distributed Denial of Service (DDoS)
- FTP and SSH brute-force-like attempts
- Botnet-like communication
- Unusual network behaviour

All attack scenarios are generated within a controlled simulation environment.

---

## 📊 Dataset

The primary dataset used for model development is **CIC-IDS2017**.

**Official Dataset:**  
https://www.unb.ca/cic/datasets/ids-2017.html

The dataset contains labelled network flow records used for supervised classification and anomaly detection experiments.

Raw dataset files are not included in this repository.

---

## 🛠️ Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- NetworkX
- Streamlit
- Plotly
- Joblib

---

## 📁 Project Structure

```text
AI-Based-Intrusion-Detection-System/
│
├── CN_IDS.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── ids_src/
│   ├── core.py
│   ├── sim.py
│   ├── data.py
│   ├── models.py
│   ├── ids.py
│   ├── evalx.py
│   ├── backend.py
│   ├── tests.py
│   └── app.py
│
├── data/
│   └── README.md
│
├── reports/
│   └── Evaluation reports and metrics
│
├── screenshots/
│   └── Dashboard screenshots
│
└── docs/
    └── Project documentation
```

---

## ⚙️ Installation & Execution

### Option 1: Google Colab (Recommended)

1. Clone or download this repository.
2. Open `CN_IDS.ipynb` in Google Colab.
3. Configure the CIC-IDS2017 dataset location.
4. Install the required dependencies.
5. Execute the notebook cells in sequence.
6. Launch the Streamlit dashboard using the notebook instructions.

### Option 2: Local Environment

**Step 1: Clone the repository**

```bash
git clone https://github.com/Avadh-G/AI-Based-Intrusion-Detection-System.git
```

**Step 2: Navigate to the project directory**

```bash
cd AI-Based-Intrusion-Detection-System
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Open the notebook**

Open `CN_IDS.ipynb` using Jupyter Notebook or upload it to Google Colab.

Configure the dataset path and execute the notebook cells in the correct order.

---

## 📈 Model Evaluation

The project evaluates its detection capabilities using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Per-class attack detection performance
- Hybrid model comparison
- Ablation studies

Evaluation reports and metrics are stored in the `reports/` directory when generated.

---

## 🔮 Future Improvements

- Improved detection of low-intensity and previously unseen attack patterns.
- Reduction of false-positive and false-negative rates.
- Better alignment between simulated and real-world network traffic.
- Real-time packet capture integration.
- Explainable AI for intrusion detection.
- Deployment on cloud-based network infrastructure.

---

## ⚠️ Disclaimer

This project is developed for academic, educational, and research purposes.

Network traffic generation, attack scenarios, and firewall responses are simulated. This project is not a replacement for a production-grade Network Intrusion Detection System and has not been validated as a live enterprise security solution.

---

## 👨‍💻 Author

**Avadh Goswami**

Computer Science & Engineering  
Artificial Intelligence & Machine Learning

**GitHub:** [Avadh-G](https://github.com/Avadh-G)

**Project Repository:** [AI-Based-Intrusion-Detection-System](https://github.com/Avadh-G/AI-Based-Intrusion-Detection-System)

---

## ⭐ Acknowledgements

- Canadian Institute for Cybersecurity – CIC-IDS2017 Dataset
- Scikit-learn
- XGBoost
- TensorFlow
- Streamlit
- NetworkX
- Python Open-Source Community

---

<p align="center">
  <b>Built with Python, Machine Learning, Deep Learning, and Network Security.</b>
  <br>
  <i>Exploring intelligent approaches to network intrusion detection.</i>
</p>

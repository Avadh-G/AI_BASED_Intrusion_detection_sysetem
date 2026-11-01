# AI-Based Intrusion Detection System (CN-IDS)

An AI-powered hybrid intrusion detection system built using
a custom Python network simulator, machine learning,
deep learning and a Streamlit dashboard.

## Features

- Custom simulated network topology
- Automated normal and malicious traffic generation
- 55 network flow features
- Random Forest classification
- XGBoost classification
- Isolation Forest anomaly detection
- Deep Learning Autoencoder
- Hybrid intrusion detection engine
- Temporal anomaly detection
- Security alerts and false-positive management
- Simulated firewall and host isolation
- Streamlit SOC dashboard
- Model evaluation and performance analysis

## Technology Stack

- Python
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- NumPy and Pandas
- NetworkX
- Streamlit
- Plotly

## Project Structure

- `ids_src/` - Python source modules
- `data/` - Dataset instructions
- `reports/` - Evaluation reports and metrics
- `screenshots/` - Dashboard screenshots
- `docs/` - Project documentation
- `requirements.txt` - Python dependencies
- `CN_IDS.ipynb` - Main Google Colab notebook

## Dataset

CIC-IDS2017:
https://www.unb.ca/cic/datasets/ids-2017.html

The raw dataset is not included.

## Running

The primary implementation is designed for Google Colab.

Open `CN_IDS.ipynb` in Google Colab and execute its cells
in the correct order.

## Disclaimer

All network traffic and security responses in this project
are simulated. They do not represent real production network
traffic or real firewall enforcement.

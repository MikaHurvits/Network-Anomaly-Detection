# Network Traffic Anomaly Detection

A comprehensive tool for detecting anomalous network behavior in PCAP (Wireshark capture) files using machine learning techniques.

![Network Security](https://img.shields.io/badge/Network-Security-blue)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)

## Overview

This project provides a robust framework for analyzing network traffic and identifying potentially suspicious behavior using advanced machine learning techniques. The tool processes PCAP files (network packet captures), extracts meaningful features, and applies machine learning models to distinguish between normal and anomalous network patterns.

## Features

- **Comprehensive Packet Analysis**: Extract rich feature sets from network traffic using pyshark
- **Multiple Model Options**:
  - **Isolation Forest**: Traditional machine learning approach for anomaly detection
  - **Hugging Face Transformers**: Advanced NLP-based approach using pre-trained models
- **Detailed Reporting**: Get comprehensive reports of detected anomalies with explanations
- **Flexible Architecture**: Easily extensible to add new features or models
- **Production-Ready Logging**: Full logging support for troubleshooting and auditing

## Installation

### Prerequisites

- Python 3.8 or higher
- Wireshark (required by pyshark)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/network-anomaly-detection.git
   cd network-anomaly-detection
   
2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
``` 
pip install -r requirements.txt
```


### Requirements
```
Copynumpy>=1.20.0
pandas>=1.3.0
pyshark>=0.4.5
scikit-learn>=1.0.2
joblib>=1.1.0
tqdm>=4.62.3
transformers>=4.18.0
torch>=1.10.0
```


## Usage
### Training a Model
Before you can detect anomalies, you need to train a model on normal network traffic:
```
python network_anomaly_detector.py normal_traffic.pcap --train 
```
Additional training options:
```
# Specify model type (isolation_forest or huggingface)
python network_anomaly_detector.py normal_traffic.pcap --train --model-type huggingface

# Specify custom model path
python network_anomaly_detector.py normal_traffic.pcap --train --model models/custom_model
```

## Detecting Anomalies
Once your model is trained, you can analyze PCAP files to detect anomalies:
```
python network_anomaly_detector.py suspicious_traffic.pcap
```
Additional detection options:
```
# Specify model type (use the same type used for training)
python network_anomaly_detector.py suspicious_traffic.pcap --model-type huggingface
```
# Specify custom model path
```
python network_anomaly_detector.py suspicious_traffic.pcap --model models/custom_model
```

## How It Works
### Feature Extraction
The system extracts a wide range of features from network packets, including:

- IP-based features (source/destination IPs, TTL values)
- Protocol-specific features (TCP/UDP ports, window sizes, flags)
- Traffic pattern features (packet lengths, protocol distribution)
- Behavioral features (connection patterns, packets per destination)

These features are then aggregated by source IP to create comprehensive behavioral profiles.

### Anomaly Detection Models
#### Isolation Forest
The Isolation Forest algorithm works by isolating observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. This process is repeated recursively until all observations are isolated. Anomalies require fewer splits to be isolated, making them easier to identify.

#### Hugging Face Transformers
This approach converts network behavior profiles into text descriptions and leverages pre-trained language models (like DistilBERT) to understand the semantic context of network patterns. The model is fine-tuned on normal traffic patterns and can detect deviations that might indicate suspicious behavior.

## Example Output
When an anomaly is detected, the system provides detailed information:
```
==================================================
Suspicious source IP ending with: 192
  - Anomaly score: 0.8735
  - Unique destinations: 45
  - Unique source ports: 12
  - Unique destination ports: 80
  - Avg packet length: 562.24
  - Max packet length: 1500
  - Total packets: 1240
  - Port ratio: 0.15
  - Packets per destination: 27.6
==================================================
```

## Use Cases

1. Network Security Monitoring: Identify potential intrusions or suspicious activities
2. Baseline Deviation Detection: Detect when network behavior deviates from normal patterns
3. Data Exfiltration Detection: Identify unusual data transfer patterns
4. Malware Communication Detection: Detect command and control communications


### Limitations

Detection quality depends on the quality of training data (normal traffic)
May require fine-tuning for specific network environments
Resource-intensive for very large PCAP files
Not a replacement for comprehensive security solutions

 
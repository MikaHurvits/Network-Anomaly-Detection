import numpy as np
import pandas as pd
import pyshark
import os
import argparse
import warnings
import joblib
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('NetworkAnomalyDetector')


class PacketFeatureExtractor:
    """
    Extract features from network packets in a PCAP file using pyshark
    """

    def __init__(self):
        self.protocol_map = {
            1: 'ICMP',
            6: 'TCP',
            17: 'UDP'
        }

    def process_pcap(self, pcap_file_path):
        """
        Process a PCAP file and extract relevant features

        Args:
            pcap_file_path (str): Path to the PCAP file

        Returns:
            pandas.DataFrame: Extracted features aggregated by source IP
        """
        logger.info(f"Processing PCAP file: {pcap_file_path}")
        packet_data = []

        # Use pyshark to read the PCAP file
        capture = pyshark.FileCapture(pcap_file_path)

        for packet_index, packet in enumerate(tqdm(capture, desc="Extracting packet features")):
            try:
                if hasattr(packet, 'ip'):
                    packet_features = self._extract_packet_features(packet)
                    if packet_features:
                        packet_data.append(packet_features)
            except (AttributeError, IndexError, ValueError) as error:
                # Skip problematic packets
                continue

        # Close the capture
        capture.close()

        if not packet_data:
            logger.warning("No suitable packets found in the PCAP file")
            return pd.DataFrame()

        # Convert to DataFrame
        packets_df = pd.DataFrame(packet_data)

        # Handle missing values
        packets_df.fillna(0, inplace=True)

        # Encode the protocol layer as a number for machine learning
        protocol_layer_map = {layer: index for index, layer in enumerate(packets_df["protocol_layer"].unique())}
        packets_df["protocol_layer_encoded"] = packets_df["protocol_layer"].map(protocol_layer_map)

        # Create aggregate features by source IP
        aggregated_features = self._aggregate_features_by_source(packets_df)

        return aggregated_features

    def _extract_packet_features(self, packet):
        """
        Extract features from a single packet

        Args:
            packet: pyshark packet object

        Returns:
            dict: Features extracted from the packet
        """
        features = {}

        # IP features
        features["source_ip"] = int(packet.ip.src.split('.')[-1])  # Last octet of source IP
        features["destination_ip"] = int(packet.ip.dst.split('.')[-1])  # Last octet of destination IP
        features["ttl"] = int(packet.ip.ttl) if hasattr(packet.ip, 'ttl') else 0
        features["packet_length"] = int(packet.length)

        # Protocol features
        if hasattr(packet, 'tcp'):
            features["protocol"] = 6  # TCP
            features["source_port"] = int(packet.tcp.srcport)
            features["destination_port"] = int(packet.tcp.dstport)
            features["window_size"] = int(packet.tcp.window_size) if hasattr(packet.tcp, 'window_size') else 0
            features["flags"] = int(packet.tcp.flags, 16) if hasattr(packet.tcp, 'flags') else 0
        elif hasattr(packet, 'udp'):
            features["protocol"] = 17  # UDP
            features["source_port"] = int(packet.udp.srcport)
            features["destination_port"] = int(packet.udp.dstport)
            features["window_size"] = 0  # No window for UDP
            features["flags"] = 0  # No flags for UDP
        else:
            features["protocol"] = int(packet.ip.proto) if hasattr(packet.ip, 'proto') else 0
            features["source_port"] = 0
            features["destination_port"] = 0
            features["window_size"] = 0
            features["flags"] = 0

        # Additional features
        if hasattr(packet, 'highest_layer'):
            features["protocol_layer"] = packet.highest_layer
        else:
            features["protocol_layer"] = "Unknown"

        return features

    def _aggregate_features_by_source(self, packets_df):
        """
        Aggregate packet features by source IP

        Args:
            packets_df (pandas.DataFrame): DataFrame containing packet features

        Returns:
            pandas.DataFrame: Aggregated features by source IP
        """
        aggregated = packets_df.groupby('source_ip').agg({
            'destination_ip': 'nunique',
            'source_port': 'nunique',
            'destination_port': 'nunique',
            'packet_length': ['mean', 'std', 'max', 'count'],
            'ttl': 'mean',
            'flags': 'sum',
            'protocol_layer_encoded': lambda x: x.nunique(),
            'protocol': lambda x: x.nunique()
        }).reset_index()

        # Flatten hierarchical column names
        aggregated.columns = [
            'source_ip',
            'unique_destinations',
            'unique_source_ports',
            'unique_destination_ports',
            'avg_packet_length',
            'std_packet_length',
            'max_packet_length',
            'packet_count',
            'avg_ttl',
            'sum_flags',
            'unique_protocol_layers',
            'unique_protocols'
        ]

        # Create derived features
        aggregated['port_ratio'] = aggregated['unique_source_ports'] / (aggregated['unique_destination_ports'] + 1)
        aggregated['dest_source_ratio'] = aggregated['unique_destinations'] / (aggregated['source_ip'] + 1)
        aggregated['packets_per_dest'] = aggregated['packet_count'] / (aggregated['unique_destinations'] + 1)

        return aggregated


class IsolationForestModel:
    """
    Anomaly detection model using Isolation Forest algorithm
    """

    def __init__(self, model_path="models/isolation_forest_model.pkl"):
        """
        Initialize the model

        Args:
            model_path (str): Path where the model will be saved/loaded
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None

    def train(self, training_data):
        """
        Train the Isolation Forest model

        Args:
            training_data (pandas.DataFrame): Data containing network features

        Returns:
            bool: True if training was successful, False otherwise
        """
        logger.info("Training Isolation Forest model on normal network traffic...")

        try:
            # Remove the source IP column for training (identifier, not a feature)
            X = training_data.drop(['source_ip'], axis=1)

            # Standardize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Create and train the model
            self.model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=0.05,  # Expected proportion of anomalies
                random_state=42,
                n_jobs=-1  # Use all available cores
            )

            self.model.fit(X_scaled)

            # Save the model and scaler
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump((self.model, self.scaler), self.model_path)

            logger.info(f"Model saved to {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False

    def load(self):
        """
        Load the trained model and scaler

        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            self.model, self.scaler = joblib.load(self.model_path)
            return True
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(self, data):
        """
        Detect anomalies in the given data

        Args:
            data (pandas.DataFrame): Data to analyze

        Returns:
            pandas.DataFrame: Data with anomaly scores and predictions
        """
        if self.model is None or self.scaler is None:
            logger.error("Model not loaded. Please train or load the model first.")
            return None

        try:
            # Make a copy to avoid modifying the original data
            results = data.copy()

            # Extract features (excluding the source IP)
            X = results.drop(['source_ip'], axis=1)

            # Scale the features
            X_scaled = self.scaler.transform(X)

            # Get anomaly scores (-1 for anomalies, 1 for normal)
            predictions = self.model.predict(X_scaled)

            # Convert to anomaly flag (1 for anomalies, 0 for normal)
            results['anomaly_flag'] = np.where(predictions == -1, 1, 0)

            # Get decision function scores (negative = more anomalous)
            scores = self.model.decision_function(X_scaled)

            # Convert to anomaly score (higher = more anomalous)
            results['anomaly_score'] = -scores

            return results

        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            return None


class HuggingFaceModel:
    """
    Anomaly detection model using HuggingFace transformers
    """

    def __init__(self, model_dir="models/huggingface_model"):
        """
        Initialize the model

        Args:
            model_dir (str): Directory where the model will be saved/loaded
        """
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None

    def _convert_to_text(self, dataframe):
        """
        Convert network features to text sequences

        Args:
            dataframe (pandas.DataFrame): Network traffic features

        Returns:
            list: Text sequences describing network behavior
        """
        text_sequences = []

        for _, row in dataframe.iterrows():
            # Create a descriptive text sequence
            sequence = (
                f"Source IP ending with {row['source_ip']} connected to {row['unique_destinations']} unique destinations "
                f"using {row['unique_source_ports']} source ports and {row['unique_destination_ports']} destination ports. "
                f"Average packet size was {row['avg_packet_length']:.1f} bytes with maximum of {row['max_packet_length']} bytes. "
                f"Total packets: {row['packet_count']}. Port ratio: {row['port_ratio']:.2f}. "
                f"Packets per destination: {row['packets_per_dest']:.1f}."
            )
            text_sequences.append(sequence)

        return text_sequences

    class NetworkTextDataset(Dataset):
        """
        PyTorch Dataset for network text data
        """

        def __init__(self, texts, labels, tokenizer, max_length=256):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].squeeze()
            attention_mask = inputs["attention_mask"].squeeze()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)
            }

    def train(self, training_data):
        """
        Fine-tune a HuggingFace model for anomaly detection

        Args:
            training_data (pandas.DataFrame): Data containing network features

        Returns:
            bool: True if training was successful, False otherwise
        """
        logger.info("Fine-tuning HuggingFace model on normal network traffic...")

        try:
            # Convert network data to text sequences
            text_sequences = self._convert_to_text(training_data)

            # For training, we assume all examples are normal (label 0)
            labels = [0] * len(text_sequences)

            # Load pre-trained model and tokenizer
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased",
                    num_labels=2
                )
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            except Exception as e:
                logger.warning(f"Error loading distilbert model: {e}")
                logger.info("Falling back to a smaller model...")

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "prajjwal1/bert-tiny",
                    num_labels=2
                )
                self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

            # Create dataset
            train_dataset = self.NetworkTextDataset(text_sequences, labels, self.tokenizer)

            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=self.model_dir,
                num_train_epochs=3,
                per_device_train_batch_size=8,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=os.path.join(self.model_dir, 'logs'),
                logging_steps=10,
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
            )

            # Fine-tune the model
            trainer.train()

            # Save the model and tokenizer
            os.makedirs(self.model_dir, exist_ok=True)
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)

            logger.info(f"Model saved to {self.model_dir}")
            return True

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False

    def load(self):
        """
        Load the trained model and tokenizer

        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(self, data):
        """
        Detect anomalies in the given data

        Args:
            data (pandas.DataFrame): Data to analyze

        Returns:
            pandas.DataFrame: Data with anomaly scores and predictions
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded. Please train or load the model first.")
            return None

        try:
            # Make a copy to avoid modifying the original data
            results = data.copy()

            # Convert to text sequences
            text_sequences = self._convert_to_text(results)

            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()

            # Get predictions
            anomaly_scores = []

            for text in tqdm(text_sequences, desc="Detecting anomalies"):
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    # Higher score for class 1 (anomaly) means more suspicious
                    score = probs[0][1].item()
                    anomaly_scores.append(score)

            # Add anomaly scores to dataframe
            results['anomaly_score'] = anomaly_scores

            # Consider anomalies where score > 0.5
            results['anomaly_flag'] = results['anomaly_score'] > 0.5

            return results

        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            return None


class NetworkAnomalyDetector:
    """
    Main class for network anomaly detection
    """

    def __init__(self, model_type="isolation_forest", model_path="models/network_model"):
        """
        Initialize the anomaly detector

        Args:
            model_type (str): Type of model to use ('isolation_forest' or 'huggingface')
            model_path (str): Path for saving/loading the model
        """
        self.model_type = model_type
        self.model_path = model_path
        self.feature_extractor = PacketFeatureExtractor()

        # Initialize the model based on type
        if model_type == "isolation_forest":
            self.model = IsolationForestModel(model_path)
        elif model_type == "huggingface":
            self.model = HuggingFaceModel(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, pcap_file):
        """
        Train the model on normal traffic

        Args:
            pcap_file (str): Path to the PCAP file containing normal traffic

        Returns:
            bool: True if training was successful, False otherwise
        """
        # Extract features from the PCAP file
        features = self.feature_extractor.process_pcap(pcap_file)

        if features.empty:
            logger.error("Failed to extract features from the PCAP file")
            return False

        # Train the model
        return self.model.train(features)

    def analyze(self, pcap_file):
        """
        Analyze a PCAP file for anomalies

        Args:
            pcap_file (str): Path to the PCAP file to analyze

        Returns:
            pandas.DataFrame: Anomalies detected in the PCAP file
        """
        # Load the model
        if not self.model.load():
            logger.error("Failed to load the model")
            return None

        # Extract features from the PCAP file
        features = self.feature_extractor.process_pcap(pcap_file)

        if features.empty:
            logger.error("Failed to extract features from the PCAP file")
            return None

        # Detect anomalies
        results = self.model.predict(features)

        if results is None:
            logger.error("Failed to detect anomalies")
            return None

        # Return only the anomalies
        anomalies = results[results['anomaly_flag'] == 1] if self.model_type == "isolation_forest" else results[
            results['anomaly_flag']]

        return anomalies


def main():
    """
    Main entry point for the script
    """
    parser = argparse.ArgumentParser(description="Network Traffic Anomaly Detection")
    parser.add_argument("pcap", help="Path to PCAP file")
    parser.add_argument("--model-type", choices=["isolation_forest", "huggingface"],
                        default="isolation_forest", help="Model type to use")
    parser.add_argument("--train", action="store_true", help="Train mode")
    parser.add_argument("--model", default="models/network_model", help="Model path/directory")

    args = parser.parse_args()

    # Create the detector
    detector = NetworkAnomalyDetector(args.model_type, args.model)

    if args.train:
        # Train the model
        logger.info(f"Training {args.model_type} model on {args.pcap}")
        success = detector.train(args.pcap)

        if success:
            logger.info("Training completed successfully")
        else:
            logger.error("Training failed")
    else:
        # Analyze the PCAP file
        logger.info(f"Analyzing {args.pcap} for anomalies")
        anomalies = detector.analyze(args.pcap)

        if anomalies is None:
            logger.error("Analysis failed")
        elif len(anomalies) == 0:
            logger.info("No anomalies detected")
        else:
            logger.info(f"Found {len(anomalies)} anomalous IPs:")

            for idx, row in anomalies.iterrows():
                print("\n" + "=" * 50)
                print(f"Suspicious source IP ending with: {int(row['source_ip'])}")
                print(f"  - Anomaly score: {row['anomaly_score']:.4f}")
                print(f"  - Unique destinations: {int(row['unique_destinations'])}")
                print(f"  - Unique source ports: {int(row['unique_source_ports'])}")
                print(f"  - Unique destination ports: {int(row['unique_destination_ports'])}")
                print(f"  - Avg packet length: {row['avg_packet_length']:.2f}")
                print(f"  - Max packet length: {row['max_packet_length']}")
                print(f"  - Total packets: {int(row['packet_count'])}")
                print(f"  - Port ratio: {row['port_ratio']:.2f}")
                print(f"  - Packets per destination: {row['packets_per_dest']:.2f}")
                print("=" * 50)


if __name__ == "__main__":
    main()
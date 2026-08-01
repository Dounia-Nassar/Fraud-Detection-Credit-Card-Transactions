# 🔍 Fraud Detection — Credit Card Transactions

> Detecting fraudulent credit card transactions using unsupervised anomaly detection with KMeans and IsolationForest on 10,000 unlabeled transactions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dounia-Nassar/Fraud-Detection-Credit-Card-Transactions/blob/main/Anomaly_Detection(Core).ipynb)

---

## 📋 Project Overview

This project detects anomalous (potentially fraudulent) credit card transactions using **unsupervised machine learning** — without any labeled fraud data. Two independent approaches are used:

1. **KMeans + Distance Thresholding** — transactions far from all cluster centers are flagged as anomalies
2. **IsolationForest** — anomalies are isolated faster in random trees because they are rare and extreme

Both models agree on the same **40 anomalous transactions (0.4%)** — matching the stakeholder-reported fraud rate — giving high confidence in the results.

**Stakeholder context:** A credit card company reported that approximately 0.4% of transactions are fraudulent (99.6% are valid). The task was to locate these anomalous transactions using the given threshold.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Records | 10,000 transactions |
| Features | 29 (V1–V28 + Amount) |
| Labels | None — fully unlabeled |
| Missing Values | 0 |
| Duplicates | 0 |
| Reported Fraud Rate | 0.4% (40 transactions) |

> **Note:** Features V1–V28 are PCA-transformed components from the original transaction data (anonymized for privacy). Only `Amount` retains its original meaning.

---

## 🔧 Pipeline

### 1. Load & Validate Data
- Loaded 10,000 transaction records with 29 features
- Confirmed **0 null values** and **0 duplicate rows**

### 2. Scale the Data
- Applied `StandardScaler` to all 29 features
- Required before KMeans — ensures no feature dominates distance calculations due to scale

### 3. KMeans Anomaly Detection
- Fitted `KMeans(n_clusters=3, random_state=42)` on scaled data
- Used `scipy.spatial.distance.cdist` to compute a **distance matrix** (10,000 × 3) — distance from every transaction to every cluster center
- Took the **minimum distance** per transaction (distance to the nearest cluster center)
- Applied threshold = **99.6th percentile** of distances → flagged top 0.4% as fraudulent

```python
threshold = np.percentile(min_distances, 99.6)  # = 20.818
anomaly_indices = X_df[min_distances > threshold].index
```

- **Result:** 40 anomalies detected

### 4. IsolationForest Anomaly Detection
- Fitted `IsolationForest(contamination=0.004, random_state=42)`
- `contamination=0.004` tells the model to expect 0.4% anomalies — matching stakeholder's report
- IsolationForest isolates anomalies by randomly partitioning features; anomalies require fewer splits (shorter paths) because they are rare and extreme

- **Result:** 40 anomalies detected

### 5. PCA Visualization
- Applied `PCA(n_components=2)` to reduce 29 dimensions to 2 for visualization
- Plotted cluster separation and anomaly locations in 2D PCA space
- KMeans anomalies, IsolationForest anomalies, and their overlap visualized separately

---

## 📈 Results

| Model | Anomalies Detected | Fraud Rate |
|---|---|---|
| KMeans (distance threshold) | 40 | 0.40% |
| IsolationForest | 40 | 0.40% |
| **Detected by BOTH** | **40** | **0.40%** |

**Both models independently identified the exact same 40 transactions** — providing high-confidence fraud flagging without any labeled training data.

### Anomaly Indices (KMeans)
```
[159, 1376, 1619, 2156, 2212, 2439, 2594, 2654, 2756, 2911,
 2914, 2917, 2923, 3443, 5303, 5412, 5413, 5529, 5674, 5704,
 5764, 5977, 6489, 6643, 6672, 7322, 7338, 7470, 7596, 7597,
 8124, 8163, 8437, 8442, 8856, 8939, 8999, 9071, 9304, 9326]
```

**KMeans Distance Threshold:** 20.818 (99.6th percentile)

---

## 🔍 How Each Model Works

### KMeans Approach
KMeans groups transactions into 3 clusters based on behavioral similarity. Normal transactions cluster tightly around their group center. Fraudulent transactions don't belong to any normal behavioral pattern — they sit far from all 3 cluster centers. By measuring each transaction's minimum distance to its nearest cluster center and flagging the most distant 0.4%, we identify the outliers.

### IsolationForest Approach
IsolationForest builds an ensemble of random decision trees. For each tree, it randomly selects a feature and a random split value. Anomalies are isolated in **fewer splits** (shorter paths) because they are rare and sit in sparse regions of the feature space. Setting `contamination=0.004` tells the model to expect exactly 0.4% fraud — matching the business context.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | KMeans, IsolationForest, StandardScaler, PCA |
| SciPy (`cdist`) | Distance matrix computation |
| Matplotlib & Seaborn | PCA visualizations |
| Google Colab | Development environment |

---

## 📁 Project Structure

```
Fraud-Detection-Credit-Card-Transactions/
│
├── Anomaly_Detection(Core).ipynb     # Main notebook
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Dounia-Nassar/Fraud-Detection-Credit-Card-Transactions
cd Fraud-Detection-Credit-Card-Transactions

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Open the notebook
jupyter notebook "Anomaly_Detection(Core).ipynb"
```

> Or click the **Open in Colab** badge at the top of this README.

---

## 👩‍💻 Author

**Dounia Nassar**
- 📧 dounia.nassar@outlook.com

# EFGTNet: Edge Feature Guided Transformer Network for Point Cloud Salient Object Detection
![](architecture.png)


---

## Dataset
The training and testing data is provided by [PCSOD](https://git.openi.org.cn/OpenDatasets/PCSOD-Dataset/datasets).

---

## Train
```bash
python train.py
```

## Evaluate
```bash
python test.py
```
You can directly obtain our visualization results from [here](), or download our pretrained model. The checkpoints can be found [here]().

---

## 🚀 Installation
### 1️⃣ Requirements
This code has been tested with the following environment:
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.3
- Other dependencies are listed in `requirements.txt`.

### 2️⃣ Installation Steps
```bash
# Clone the repository
git clone https://github.com/your-repo.git
cd your-repo

# Create and activate a Conda virtual environment
conda create -n 3Dstructure python=3.8
conda activate 3Dstructure

# Install dependencies
pip install -r requirements.txt


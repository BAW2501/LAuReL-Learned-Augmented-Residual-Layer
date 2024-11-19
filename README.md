# LAuReL Implementations

This repository contains my independent implementations of the three LAuReL variants described in the article titled **"LAuReL: Learned Augmented Residual Layer"**. These implementations aim to explore the concepts presented in the paper and evaluate their effectiveness on image datasets. 

Original article  [LAUREL: Learned Augmented Residual Layer]( https://arxiv.org/pdf/2411.07501)
## ⚠️ Disclaimer

- **Unofficial Code**: This project is my interpretation of the LAuReL blocks based on the descriptions in the paper. Until the authors release the official implementation, there is no guarantee that this code exactly replicates their methodology. 
- **Independent Validation**: I have conducted some independent tests on small datasets (e.g., CIFAR-10), and the implementations appear to perform well. However, further rigorous testing and validation are recommended before using these in production.

## Project Structure

```
.
├── LAuReL.py             # Basic implementation of LAuReL with learnable scalar weights (α and β)
├── LAuReL-LR.py          # Implementation of the Low-Rank version of LAuReL
├── LAuReL-PA.py          # Implementation of the Previous Activations version of LAuReL
├── TrainUtil.py          # Utility functions for training and evaluation
├── Test.py               # Script for running and comparing models with different LAuReL blocks
├── pyproject.toml        # Project configuration and dependencies
├── uv.lock               # Locked dependency versions
├── .python-version       # Python version specification
├── README.md             # This README file
├── requirements.txt      # List of dependencies
```

## Implemented Variants

1. **LAuReL (Basic)**:
   - Adds learnable scalar weights (\( \alpha, \beta \)) to combine the residual and transformed paths.
   - Normalizes weights using softmax for stability.

2. **LAuReL-LR (Low-Rank)**:
   - Introduces a low-rank approximation \( W = AB + I \) for the residual connection.
   - Significantly reduces parameters compared to a full-rank matrix \( W \).

3. **LAuReL-PA (Previous Activations)**:
   - Leverages activations from all previous layers using learnable weights (\( \gamma_j \)).
   - Uses a low-rank transformation for each previous activation.

## Running the Project

### Prerequisites
- Python 3.10+
- PyTorch 2.5 or higher
- torchvision

Install dependencies via `pip`:
```bash
pip install -r requirements.txt
```

or

Install dependencies via `uv`:
```bash
uv sync
```



### Training and Testing

To train a model using one of the LAuReL variants, run the `Test.py` script:
```bash
python Test.py
```

The script will:
1. Train small CNN models with each LAuReL variant on the CIFAR-10 dataset.
2. Compare test accuracy and training time for the variants.

## Example Results

To ADD LATER RUNS TAKE TOO LONG (ran for a few hours with notSOSmallCNN)



## Next Steps

- Conduct more comprehensive testing across larger datasets like ImageNet.
- Investigate combinations of LAuReL variants for potential performance gains.
- Optimize for deployment by profiling model latency and memory usage.

## Contributing

Feel free to open issues or submit pull requests if you find improvements, bugs, or have suggestions for extensions.

## Acknowledgments

This implementation is inspired by the LAuReL concepts introduced in the paper. Full credit goes to the original authors for their innovative approach.

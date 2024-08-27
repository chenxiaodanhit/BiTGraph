# BiTGraph (ICLR 2024)
The code for paper: [Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values](https://openreview.net/pdf?id=O9nZCwdGcG). 

# Getting Start

1. Install requirements. `pip install -r requirements.txt`
2. Download data.

  Download Metr-LA, ETTh1, Electricity, PEMS datasets from [here](https://drive.google.com/file/d/1uOCHzx-xEAIfrPAiyQIy7pgwPvE6itj3/view?usp=sharing). Obtain BeijingAir dataset from [Brits](https://papers.nips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf). Put all the files in the ./data.
  
3. Training.
   
` python main.py --epochs 200 --mask_ratio 0.2 --dataset-name Metr`
 
4. Testing.

`python test_forecasting.py --epochs 200 --mask_ratio 0.2 --dataset-name Metr`


# Citation

```
@inproceedings{BiTGraph, 
  title={Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values},
  author={Chen, Xiaodan and Li, Xiucheng and Liu, Bo and Li, Zhijun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```




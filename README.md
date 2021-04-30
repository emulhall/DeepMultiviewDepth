# Final Model for CSCI 5563 Project
Built off of the original architecture of Deep Multi-view Depth Estimation with Predicted Uncertainty.
Citation:
```@article{ke2020multiview,
  title={Deep Multi-view Depth Estimation with Predicted Uncertainty},
  author={Ke, Tong and Do, Tien and Vuong, Khiem and Sartipi, Kourosh and Roumeliotis, Stergios I},
  journal={arXiv preprint arXiv:2011.09594},
  year={2020}
} 
```

## Training

We trained our network with train.py, using the TUM RGB-D dataset, specifically the Dynamic Objects subset.  COLMAP must first be run on these files as the initial depth estimate is an input to the network.

## Testing

To test our network on our dynamic objects dataset, we ran test.py.

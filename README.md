# Learning Fluid-Structure Interaction Dynamics with Physics-Informed Neural Networks and Immersed Boundary Methods

[![arXiv](https://img.shields.io/badge/arXiv-2505.18565-b31b1b.svg)](https://arxiv.org/abs/2505.18565)



## About

Source code of Learning Fluid-Structure Interaction Dynamics with Physics-Informed Neural Networks and Immersed Boundary Methods described in the paper: [Learning Fluid-Structure Interaction Dynamics with Physics-Informed Neural Networks and Immersed Boundary Methods](https://www.arxiv.org/abs/2505.18565).


### Prerequisites

[Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or any other Python environment.


## Project structure

```bash
.
├── README.md
├── result
│   └── fsi
├── src
│   ├── data
│   │   ├── IBM_data_loader.py
│   ├── nn
│   │   ├── bspline.py
│   │   ├── kan2.py
│   │   ├── nn_functions.py
│   │   ├── pde.py
│   │   ├── tanh.py
│   │   ├── tanh2.py
│   ├── notebook
│   │   ├── M1_fsi_spectral_with_adaptive_methods.ipynb
│   │   ├── M2_fsi_spectral_adaptive_methods.ipynb
│   │   └── contour_plot_all_models_.ipynb  
│   ├── trainer
│   │   ├── m1_trainer.py
│   │   └── m2_trainer.py
│   └── utils
│       ├── colors.py
│       ├── fsi_visualization.py    
│       ├── line_plot2.py
│       ├── logger.py
│       ├── plot_losses.py
│       ├── combine_fluid_csv_files.py
│       ├── combine_solid_csv_files.py
│       ├── draw_contour_plts.py
│       ├── line_plot2.py
│       ├── logger.py
│       ├── plot_losses.py
│       ├── plotting_irregular_2D_interface.py
│       ├── plotting_regular_2D.py
│       ├── plotting_regular_2D_time_seqeunce.py
│       ├── printing.py
│       └── utils.py
```

### Setup environment

The code is tested in Ubuntu 20.04 LTS, using Nvidia A100 GPU.

```bash
conda env create -f environment.yml
conda activate pinn_fsi_ibm

# Check if PyTorch and CUDA available
python -m src.utils.check_torch
    Version 2.4.0
    CUDA: True
    CUDA Version: 12.4
    NCCL Version: (2, 20, 5)
```

### Training

To train models run the following commands.

```bash
# fsi
python -m src.trainer.m1_trainer

```

### Notebooks for Plots

We provided all pre-trained models and training loss log history. The notebooks can be run independently of training models.

Test models

- fsi: `fsi_test_model.ipynb`

Plot loss history and test results

- fsi contour plot of test and error: `contour_plot_all_models_.ipynb`

## License

MIT [LICENSE](LICENSE)

## Citation

If you find this work useful, we would appreciate it if you could consider citing it:

```bibtex
@article{farea2025learning,
  title={Learning Fluid-Structure Interaction Dynamics with Physics-Informed Neural Networks and Immersed Boundary Methods},
  author={Farea, Afrah and Khan, Saiful and Daryani, Reza and Ersan, Emre Cenk and Celebi, Mustafa Serdar},
  journal={arXiv preprint arXiv:2505.18565},
  year={2025}
}
```

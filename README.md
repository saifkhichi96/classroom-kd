# Classroom-Inspired Multi-Mentor Distillation with Adaptive Learning Strategies

This repository contains the code for the paper "Classroom-Inspired Multi-Mentor Distillation with Adaptive Learning Strategies" accepted at the 11th Intelligent Systems Conference 2025 (IntelliSys 2025).

> [!NOTE]
> **Acknowledgement** The source code is based on the [hunto/image_classification_sota](https://github.com/hunto/image_classification_sota) repository. We thank the authors for their work and for making their code publicly available.

## Getting Started

Clone the repository into your local machine and install the required packages:

```bash
git clone https://github.com/saifkhichi96/classroom-kd.git
cd classroom-kd

pip install -r requirements.txt
```

### Preparing Datasets

Create a directory named `data` in the root of the repository and download the CIFAR-100 (and ImageNet, if needed) dataset into it The directory structure should look like this:

```
classroom-kd/
├── configs/
├── data/
│   ├── imagenet/
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   ├── cifar/
│   │   ├── cifar-10-batches-py
│   │   ├── cifar-100-python
├── lib/
├── tools/
```

We provide the configuration files for our experiments in the [`configs/classroom/`](configs/classroom/) directory. This includes multiple classroom setups with CIFAR-100 and ImageNet datasets.

### Training

To train the model, you can use the following command:

```bash
python tools/train.py \
  -c configs/classroom/mobv2_student/1t5p.yaml \
  --experiment experiments/mobv2_student/1t5p \
  --model cifar_MobileNetV2 \
  --ask
```

Other options include:
- `--ask` to activate asking module
- `--ask_b` for ranking of type b
- `--no_ask` to disable asking 
- `--no_mentor` to disable mentoring module
- `--dtkd_mentor` to replace mentoring module with dtkd

For **peer pretraining**, use the following command:

```bash
python tools/train.py -c configs/classroom/trained_peers/mobv2/0t0p.yaml --model cifar_MobileNetV2 --experiment experiments/mobv2_peer
```

To run the ablation study for the number of peers, use the following command:

```bash
python tools/train.py -c configs/classroom/efficientnet_ablation/1t5p.yaml --model cifar_MobileNetV2 --ask --experiment experiments/efficient_ablation/1t5p
```

## Citation
If this project or dataset proves helpful in your work, please cite:

```bibtex
@article{sarode2024classroom, 
  title={Classroom-Inspired Multi-Mentor Distillation with Adaptive Learning Strategies}, 
  author={Sarode, Shalini and Khan, Muhammad Saif Ullah and Shehzadi, Tahira and Stricker, Didier and Afzal, Muhammad Zeshan}, 
  journal={arXiv preprint arXiv:2409.20237}, 
  year={2024} 
}
```

## License
This project is released under the [CC-BY-NC-4.0 License](./LICENSE). Commercial use is prohibited, and appropriate attribution is required for research or educational applications.

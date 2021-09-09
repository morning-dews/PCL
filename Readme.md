#### Instructions

This is the code for paper  **Continual Learning by Using Information of Each Class Holistically**

Some pretrained features can be found here: https://drive.google.com/drive/folders/1S9hHDUiWg5W9co2W9_vZOSUxtnpuDk3i?usp=sharing

*Python Environment*:

python                            3.6.8
numpy                            1.14.5
torch                               1.2.0
torchvision                     0.4.0

Running command:
usage: main.py [-h] [--max_epochs MAX_EPOCHS] [--batch_size BATCH_SIZE]
               [--n_cpu N_CPU] [--gpu GPU] [--dropoutrate DROPOUTRATE]
               [--dataset 	  {mnist,cifar10,cifar100,emnist,twentynews,dbpedia}]

Example: 

python main.py --dataset mnist --gpu 0



 If this code helps you, please cite our paper:

```bibex
@article{hu2021continual,
  title={Continual Learning by Using Information of Each Class Holistically},
  author={Hu, Wenpeng and Qin, Qi and Wang, Mengyu and Ma, Jinwen and Liu, Bing},
  journal={AAAI},
  year={2021}
}
```


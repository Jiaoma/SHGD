# Self-supervised Social Relation Representation for Human Group Detection 
[[paper]](https://arxiv.org/abs/2203.03843)

If you find our work or the codebase inspiring and useful to your research, please cite
```bibtex
@article{li2022self,
  title={Self-supervised Social Relation Representation for Human Group Detection},
  author={Li, Jiacheng and Han, Ruize and Yan, Haomin and Qian, Zekun and Feng, Wei and Wang, Song},
  journal={arXiv preprint arXiv:2203.03843},
  year={2022}
}
```

## Preparation
### Dependence
0. Python env: Pytorch 1.10, Cuda 11.1.
1. Install [cdp](https://github.com/XiaohangZhan/cdp) into reference folder.
2. Install [Shift_GCN](https://github.com/kchengiva/Shift-GCN) into reference folder. In `reference/Shift_GCN/model/shift_gcn.py`, add `import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Temporal_shift/"))` to line 11.
### Datasets
Datasets used in this paper can be downloaded from the dataset websites below:
[PANDA](http://www.panda-dataset.com/)

for PANDA dataset, the structure of folders should be placed as below:
```
.(YOURPATH/PANDA)
├── PANDA_IMAGE
│   ├── image_annos
│   ├── image_test
│   ├── image_train_full
├── PANDA_Video_groups_and_interactions
│   ├── readme.md
│   └── train
├── video_annos
│   ├── 01_University_Canteen
│   ├── 02_OCT_Habour
    ...
├── video_test
│   ├── 11_Train_Station_Square
│   ├── 12_Nanshan_i_Park
    ...
└── video_train
    ├── 01_University_Canteen
    ├── 02_OCT_Habour
    ...

```
### Extract skeletons
1. Clone Unipose into `feature_extract` folder: `git clone git@github.com:bmartacho/UniPose.git`.
2. Download `UniPose_MPII.pth` from [here](https://drive.google.com/drive/folders/1dPc7AayY2Pi3gjUURgozkuvlab5Vr-9n)
3. Edit `YOURPATH` and `SAVEPATH` in `feature_extract/config.py`.
4. Change the working directory into feature_extract by `cd feature_extract` and run `python extractor.py` to extract skeletons and generate `train_all_features.pth.tar`, `train_group_interaction_features.pth.tar` and `train_interaction_features.pth.tar` into `SAVEPATH`.

Then replace the paths in `config.json` with yours.

## Training&Testing
The whole training consists of 2 stages.

### Stage 1
`python selftrain_stage1.py`

### Stage 2
`python selftrain_stage2.py`

### Evaluate
1. Comment `train_net(cfg)` and uncomment `# test_net(cfg)` in `selftrain_stage2.py`
2. Set `stage2_model_path` in `selftrain_stage2.py` to the path of the evaluating model checkpoints.
3. Run `python selftrain_stage2.py`

Or you can edit the pt file path at the bottom of `evaluate.py` and run `python evaluate.py`.
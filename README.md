# 音频分离任务

基于pytorch搭建的神经网络，以实现音频分离任务。

### Prerequisites

```
python == 3.9.0
```

以下为所使用的第三方库：

```
librosa==0.10.1
mir_eval==0.7
numpy==1.25.0
scipy==1.11.3
soundfile==0.12.1
torch==2.1.0
tqdm==4.66.1i
```

### Installing

 可以通过如下命令来配置环境：

```
pip install -r requirements.txt
```

## 

## Getting Started

对于两个说话人集合，我们先将其放在`data_prepare/data_raw`文件下,`A`,`B`表示两个说话人集合：

```
data_raw
│
└───A
│   │
│   └───speaker_1
│  	│   │   1.flac
│   │   │   2.flac
│  	│   │   ...
│   └───speaker_2
│ 	│ ....
│ 
└───B
		│...
		│...
```

可以直接执行`bash data_prepare/bash.sh`来进行音频合成和数据集分割，生成的结果在`result/audio`中。tr,cv,tt分别表示训练、验证、测试集。

### Prerequisites

```
python == 3.9.0
```

以下为所使用的第三方库：

```
librosa==0.10.1
mir_eval==0.7
numpy==1.25.0
scipy==1.11.3
soundfile==0.12.1
torch==2.1.0
tqdm==4.66.1i
```

### Installing

 可以通过如下命令来配置环境：

```
pip install -r requirements.txt
```

## Running the tests

运行`bash run.sh`即可。模型与训练相关的参数可以在`run.sh`中调整。

训练好的模型会保存在`./checkpoints`中。

## Built With

* [zreach](https://github.com/zreach) - Myself.

## Reference

Yi Luo,Nima Mesgarani, [TasNet: time-domain audio separation network for real-time, single-channel speech separation](https://arxiv.org/abs/1711.00541)

```
@misc{luo2018tasnet,
      title={TasNet: time-domain audio separation network for real-time, single-channel speech separation}, 
      author={Yi Luo and Nima Mesgarani},
      year={2018},
      eprint={1711.00541},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## Authors

* **Yizhi Zhou** - *Initial work* - [zreach](
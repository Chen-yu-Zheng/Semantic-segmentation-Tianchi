# Semantic-segmentation-Tianchi

## Intro

最近的项目推动需要学习语义分割的知识，借这个机会把传统的语义分割模型都实现一遍，当作实验下半年论文和比赛的baseline（ref: https://tianchi.aliyun.com/competition/entrance/531872/introduction）



## Results

H：HorizontalFlip

V：VerticalFlip

Ro：RandomRotate90

Re：Resize（推理时将原图片Resize成256然后输入，最后插值获得结果，不然最后结果超过20M（似乎更精细））

|   Method   | Score  |
| :--------: | :----: |
|    FCN     | 0.7582 |
|   FCN+H    | 0.7625 |
| FCN+H+V+Ro | 0.7572 |
|  U-net+R   | 0.8641 |




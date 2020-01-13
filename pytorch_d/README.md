###  Pytorch使用Tensorboard可视化网络结构
1. pip install tensorboardX

2. 在代码中加入
```python
from tensorboardX import SummaryWriter
...
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input, ))
```

3. 上面的代码运行结束后，会在当前目录生成一个叫run的文件夹，里面存储了可视化所需要的日志信息。用cmd进入到runs文件夹所在的目录中（路劲中不能有中文），然后cmd中输入：
```shell
tensorboard --logdir runs
```

### pytorchviz
1. pip install graphviz
2. 代码中加入如下代码
```python
from torchviz import make_dot
...
vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
vis_graph.view()
```

### torchsummary
```shell
pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com torchsummaryX
```
 https://github.com/nmhkahn/torchsummaryX
 ```python
from torchsummary import summary
summary(model, (3, 224, 224))
```
 
 


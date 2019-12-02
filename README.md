# KK_AI
kaike

## 使用指定的GPU
1. 直接在终端中设定：
```shell
CUDA_VISIBLE_DEVICES=1 python my_script.py
```
2. **python代码中设定**
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```
3. 使用函数set_device
```python
import torch
torch.cuda.set_device(id)
```




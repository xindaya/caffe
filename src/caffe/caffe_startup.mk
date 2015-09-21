caffe_startup
===

# tools/caffe-main

## 解析参数

1. caffe_engine::initPS()
2. caffe_engine::start()
---

## caffe_engine::InitPS工作

1. init()->InitPS()
2. InitPS() -> InitPSForTrainNet()
3. initPS() -> InitPSForTestNet()
4. start()-> 调用caffe本身的solver的solve方法

---

# caffe_engine 几个重要变量

1. layer_blobs_global_idx_ 变量初始化
2. InitPSForTrainNet
3. InitPSForTestNet

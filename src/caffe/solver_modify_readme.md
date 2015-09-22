# solver代码移植说明

**注意：**

1）为了表述方便：
A版本caffe：bosen下的（单卡）caffe
B版本caffe：多卡caffe
四变量：`num, channels, height, width`

2）所有修改的部分都添加了注释`modification part`

3）任务就是要将A版本caffe中**关于bosen的东西**移植到B中

总结：
solver的修改核心重点是如何移植bosen下的`net_->SyncWithPS`和`ThreadSyncWithPS`。

目前的解决方案：

`net_SyncWithPS`：放在Solver函数中，在里面的Step函数之前，同时移过去的还有PreSolve
`ThreadSyncWithPS`：bosen下这个函数还调用了`ComputeUpdateValue`进行更新，这样会与ApplyUpdate冲突，因而只需要移植`ThreadSyncWithPS`中的两个函数：**param->UpdatePSTable()**和**param->SyncWithPSTable(clock+1)**。这里选择在ApplyUpdate中加入这两个函数，并保持param输入变量含义一致

另外需要注意的是：

bosen下的ForwardBackward是在Solver层定义和调用的，而Multi-GPU则是在net层定义和在solver调用的。因而会有些许不同。`ThreadSyncWithPS`就在ForwardBackward里被调用的。解决思路见文档最末尾。


## 函数
### 1. 添加函数
- `InitSVB`，直接添加，虽然目前没看出来有什么用
### 2. 增加函数
- B版本增加的内部类`Callback`，包括`on_start & on_gradients_ready`
- A版本是把`PreSolve`放在基类Solver类里作为一个虚函数，而B版本则是对每一个继承类如`SGDSolver`单独定义，并在构造函数中调用，**目前的解决方案：**先把`PreSolve`加入基类中，并实现。然后在`Solve`函数中在`Step`之前调用`PreSolve`
- A版本把`ForwardBackward`提到`Solver`层来实现，而在B版本中是在`net`中实现的。

### 3. 融合函数
- 构造函数：增加变量`layer_blobs_global_idx_ptr`和`thread_id`
- `SGDSolver`等具体Solver的构造函数都需要改变
- `GetSolver`添加`layer_blobs_global_idx_ptr & thread_id`输入变量
- `InitTrainNet`中添加对layer的blob global table setup部分
- **重要核心修改**

### 4. 核心函数
从solver起涉及到了核心函数的修改。

#### solver->Solve()
从A版本中主要添加了两步在B版本中的`Step`函数中。
 - PreSolve()
 - net_->SyncWithPS(clock)

而A版本在`SyncWithPS`后的每次迭代内容则可和`Step`函数中比对。

#### ForwardBackward
A版本中是提到solver->Solve()中实现，而B版本是在net中实现。

两种思路：

1）以B版本为准
需要做的就是**把A版本中ForwardBackward涉及到的blob ps table更新和同步操作移植到B版本中。**
因为A版本直接就在ForwardBackward的中Backward部分的`ThreadSyncWithPS`调用了ComputeUpdateValue，因而修改方法为除了保留`ThreadSyncWithPS`的`UpdatePSTable, SyncWithPSTable`这两个函数（Table的更新同步必须要保留），其余全部丢弃。这两个函数放在`ApplyUpdate`之后。

2）以A版本为准
其实就是在B版本的`Step`函数中调用的是net_->ForwardBackward来计算loss，那么只要改成ForwardBackward，也即调用Solver自身的ForwardBackward来计算即可。

这个思路的问题就是`ThreadSyncWithPS`的ComputeUpdateValue与ApplyUpdate冲突的问题，要以B版本为准，因而选择第一种思路。

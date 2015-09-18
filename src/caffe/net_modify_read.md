# Net源码修改记录

**注意：**

为了表述方便：
A版本caffe：bosen下的（单卡）caffe
B版本caffe：多卡caffe
四变量：`num, channels, height, width`

所有修改的部分都添加了注释`modification part`

任务就是要将A版本caffe中**关于bosen的东西**移植到B中

`net.hpp`不太确定的问题：
- 构造函数如何修改（目前方案是直接融合变量）
- `ForwardDebugInfo`等三个函数是放在`public`还是放在`protected`中？（目前是放在public中，将protected的注释）

`net.cpp`不太确定的问题：
 - 构造函数如何修改（与hpp一致）
 - A版本有一个`GetLayer`，在B版本中是`CreateLayer`，目前留下`remain puzzle`标记，并没做任何修改

其他需要注意的问题：
- 其余的出现不同，都是B中函数最后添加了一个`const`，因而没有修改。
- 从A版本移植过来，要重点注意的：`client_id`（顺便带上`thread_id`）
- 在对`bottom_vecs_, top_vecs_`进行resize的时候，调用了`param.layer_size()`，但是A版本则是`param.layers_size()`，多了一个`s`，要特别注意一下
- 很多if判断打印语句，A版本的判断条件是`if (client_id_ == 0 && thread_id_ == 0)`，B版本则是`if (Caffe::root_solver())`，处理方法为三个判断一并加入。

## 函数方面

###1、添加函数
- `CopyTrainedLayersFrom`输入参数增加了`init_ps_tables`
- `FilterNet`输入参数增加了`thread_id`


###2、增加函数
 - 涉及到PS Table之类的都直接添加
 - `set_net_id, set_debug_info`也是直接添加
 - `InitPS`和`SyncPS`以及`RegisterNetOutputPSTable`等涉及到`table`的操作函数直接copy


###3、融合函数
 - **构造函数**，两版本的不同之处在于：A版本有`thread_id, net_id`，B版本有`root_net, phase`，有两个构造函数相关联，目前选择将这两个构造函数**变量融合**
 - `ForwardDebugInfo`等DebugInfo信息在B版本中是在`protected`中，而A版本则是在public中，**目前是直接把A版本中的拷过来放在public中，但这样就有两个一样的函数了，有待进一步解决**
 - `Init`函数中对`layer`循环做`set_param_propagate_down`处理时有一些逻辑没有添加
 - `FilterNet`函数输入参数有变化
 - `AppendParam`函数中最后`strict dimension checking`这块差距较大，目前没有改变
 - `ForwardFromTo`中涉及到layer操作，A版本输入变量加了指针，而B版本则是引用，因而全部以引用为准（layer未改）
 - `CopyTrainedLayersFrom`修改了`target_blobs`的`FromProto`设置，加入了`client_id`和`thread_id`的判断逻辑。
 - `ToProto`增加了几行循环，包括`add_bottom`和`add_top`操作
 - **`Update`**A版本似乎增加了不少，包含`owner_diff`等，首先B版本中的`learnable_params_`和A版本的`params_`是一样的含义，B版本是直接调用blob的update进行更新，做的是$w+\Delta w$更新，而A版本则是在net的update中直接进行`diff`的add操作，操作内容似乎不同。B版本中是在solver的ApplyUpdate中调用net的update，A版本的solver中并没有用到该update，似乎都是通过UpdatePSTable和SyncWithPSTable来进行的。因此**先保持现状暂时不做任何修改**


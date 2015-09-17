# Blob代码移植说明

**注意：**
1）为了表述方便：
A版本caffe：bosen下的（单卡）caffe
B版本caffe：多卡caffe
四变量：`num, channels, height, width`

2）所有修改的部分都添加了注释`modification part`

3）任务就是要将A版本caffe中**关于bosen的东西**移植到B中

4）多卡caffe把之前blob中的四变量num, channels, height, width改为用vector类的shape代替，不限维数。
所以把A中的Blob移植到B中这些不能移过来。

## 函数方面

###无需移植函数
比如，`inline int num() const { return num_; }`
已经用
`inline int num() const { return LegacyShape(0); }`所代替，所以无需移植

### 融合函数
所谓融合函数即是函数名字相同，但或者代码内容不同，或者输入变量不同，不一而足。
#### 构造函数
A中构造函数里添加了`blob_mode`和`global_id`进去，但是是四变量版本的构造函数，不是shape，目前只是修改了B中四变量版本的blob构造函数，**shape变量版本的blog构造函数是否要添加此两变量，看具体实现怎么样**
#### Reshape函数
B版本做了一个很好的融合函数：
`Reshape(const int num, const int channels, const int height, const int width)`
里面用shape重新写，并调用`Reshape(shape)`进行`Reshape`。
这样的好处在于，A版本涉及到调用Reshape的，比如`ReshapeWithoutAllocation, ReshapeWithoutAllocationLike`之类的函数，其输入变量为四变量`num, channels, height, width`便可以直接移植过来而不用做修改
#### FromProto函数:
A:
`void FromProto(const BlobProto& proto, const bool init_ps_table = false);`
B:
`void FromProto(const BlobProto& proto, bool reshape = true`
将A中变量融入B：
`void FromProto(const BlobProto& proto, bool reshape = true, const bool init_ps_table = false);`

并将代码中涉及到`init_ps_table`的代码添加进函数逻辑中。

#### ToProto函数
A中ToProto直接采用Dtype，而B中很奇怪的写了`Double`和`Float`两个版本，**目前选择全部保留，而不修改为`Dtype`**。只把涉及到对`blob_mode`和`global_id`的`set`代码添加进去
```
  proto->set_blob_mode(blob_mode_);
  proto->set_global_id(global_id_);
```

### 添加函数
所谓添加函数，包括两种情况：
1）是A中有的函数，而并没有在B中出现。这方面显然主要都是基于bosen的函数。
2）是B中有的函数，但并没有在A中出现。这方面显然主要是支持多卡所涉及的函数

对于第一种情况，在blob上主要是关于table的操作，比如`SyncWithPSTable`，`UpdatePSTable`等，这些函数中的变量并没有涉及到四变量等需要做修改的，因而可以直接copy过来。

对于第二种情况，在blob中涉及到的函数有如`sumsq_data`，`sumsq_diff`等，都是基于`data`和`diff`的操作。
还有如`ShareEquals`，目前没看到有什么作用的函数。



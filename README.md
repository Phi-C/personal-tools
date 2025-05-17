# Perftools
`Perftools` is a repo which providies some useful tools for performance analysis

## Toolkit
| keywords | function|
|---|---|
|`multi_proc_dump.py`| dump each process's log/output into one log file|
|`TimerContext`|record code block's time with the help of context management|
|`shape_summurizer.py`| 给定`torch.nn.Module`, 展示模型forward过程的shape信息. 有两种方式实现:1. `TorchInfoSummurizer`: 借助torchinfo实现; 2. 使用pytorch的forward hook实现|
|`tracers.py`| 定义了一些有用的tracer(见`docs/TraceHook.md`).tensor_tracer: 获取函数执行中每一个局部变量的shape, 相比reason_shape.py, 可捕获粒度更小, 即使不是`torch.nn.Module`的计算, 也可以捕获, 比如`tensor_C = tensor_A + tensor_B`|



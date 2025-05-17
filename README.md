# Perftools
`Perftools` is a repo which providies some useful tools for performance analysis

## Toolkit
| 功能 | 脚本|
|---|---|
|获取模型中间结果的shape信息|1. `shape_summurizer.py`: `TorchInforSummurizer`、`register_shape_infer_hook`; <br> 2. `tracer.py`: `tensor_tracer`|
|将每个进程的输出保存到各自的log里| `multi_proc_dump.py`: `multi_proc_dump`|
|记录每个code block话费的事件|`timer_context.py`: `TimerContext`|
|精度对比工具|`precision_checker.py`:`PrecisionChecker`|



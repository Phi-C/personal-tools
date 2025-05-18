# Scalopus简介
Scalopus 是一个追踪框架，能够从C++和Python应用程序中收集并可视化执行轨迹。该框架提供检测功能，帮助开发者理解程序的时间消耗分布，其核心特性在于捕获作用域/栈帧信息。


# 工具使用
```shell
git clone https://github.com/iwanders/scalopus.git
cd scalopus
git submodule update --init --recursive

rm -rf build
mkdir build
cd build

# cmake版本要满足一定要求, 具体看CMakeLists.txt文件
${CMAKE_PATH}cmake -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so ..
make -j

cd build/scalopus_python
pip install .
```
将`scalopus_tracer.py`放至执行目录, 在模型运行处添加如下几行代码:
```python
from scalopus_tracer import ModelTracer
tracer = ModelTracer().init_scalopus("my_model")
tracer.register_hooks(model)
dist.all_reduce = tracer.trace_comm_ops(dist.all_reduce)
tracer.trace_aten_ops()
tracer.start_tracing()
# ...运行模型...
tracer.stop_tracing()
```

# Chrome Trace Viewer
参考资料: 
[强大的可视化利器Chrome Trace Viewer使用详解](https://limboy.me/posts/chrome-trace-viewer/)
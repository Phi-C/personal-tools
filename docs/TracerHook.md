https://openobserve.ai/articles/python-m-trace-code-tracing/
# Trace Hook
python中的`sys.settrace()`函数通过设置一个跟踪函数, 该函数会在程序执行过程的多种事件(event)中被调用。

```python
import sys

def trace_calls(frame, event, arg):
    if event == 'call':
        print(f'Calling function: {frame.f_code.co_name}')
    elif event == 'line':
        print(f'Executing line {frame.f_lineno} in {frame.f_code.co_name}:{frame.f_code.co_filename}')
    elif event == 'return':
        print(f'{frame.f_code.co_name} returned {arg}')
    return trace_calls

sys.settrace(trace_calls)

def add(a, b):
    result = a + b
    return result

def main():
    x = add(1, 2)
    print(f'Result: {x}')

main()
```

# Frame, Event and Arg
## Frame
Cpyhton中的`PyFrameObject`实现
```c
typedef struct _frame {
    PyObject_VAR_HEAD
    struct _frame *f_back;      /* previous frame, or NULL */
    PyCodeObject *f_code;       /* code segment */
    PyObject *f_builtins;       /* builtin symbol table (PyDictObject) */
    PyObject *f_globals;        /* global symbol table (PyDictObject) */
    PyObject *f_locals;         /* local symbol table (any mapping) */
    PyObject **f_valuestack;    /* points after the last local */
    /* Next free slot in f_valuestack.  Frame creation sets to f_valuestack.
       Frame evaluation usually NULLs it, but a frame that yields sets it
       to the current stack top. */
    PyObject **f_stacktop;
    PyObject *f_trace;          /* Trace function */
    char f_trace_lines;         /* Emit per-line trace events? */
    char f_trace_opcodes;       /* Emit per-opcode trace events? */

    /* Borrowed reference to a generator, or NULL */
    PyObject *f_gen;

    int f_lasti;                /* Last instruction if called */
    /* Call PyFrame_GetLineNumber() instead of reading this field
       directly.  As of 2.3 f_lineno is only valid when tracing is
       active (i.e. when f_trace is set).  At other times we use
       PyCode_Addr2Line to calculate the line from the current
       bytecode index. */
    int f_lineno;               /* Current line number */
    int f_iblock;               /* index in f_blockstack */
    char f_executing;           /* whether the frame is still executing */
    PyTryBlock f_blockstack[CO_MAXBLOCKS]; /* for try and loop blocks */
    PyObject *f_localsplus[1];  /* locals+stack, dynamically sized */
} PyFrameObject;
```
## Event
常见事件类型:
1. `call`: 调用函数时触发
2. `line`: 执行一行新的代码时触发
3. `return`: 当函数返回值时触发
4. `exception`: 但函数引起异常时触发
5. `c_call`: 函数调用C function时触发
6. `c_return`: C function返回式触发
7. `c_exception`: C函数引起异常时触发
## Arg
`arg` 提供有关事件的附加信息。例如，对于“return”事件，它包含函数的返回值。

# Examples
## Trace Calls
```python
import sys

def trace_calls(frame, event, arg):
    if event == 'call':
        print(f"Call to {frame.f_code.co_name} at line {frame.f_lineno}")
    return trace_calls

def foo():
    print("Inside foo()")

def bar():
    print("Inside bar()")
    foo()

sys.settrace(trace_calls)
bar()
sys.settrace(None)
```

## Trace Line Execution
```python
import sys

def trace_lines(frame, event, arg):
    if event == 'line':
        lineno = frame.f_lineno
        filename = frame.f_code.co_filename
        print(f"Executing line {lineno} in {filename}")
    return trace_lines

def sample_function():
    x = 1
    y = 2
    z = x + y
    print(z)

sys.settrace(trace_lines)
sample_function()
sys.settrace(None)
```
## Trace Return Values
```pyhton
import sys

def trace_return(frame, event, arg):
    if event == 'return':
        print(f"Returning from {frame.f_code.co_name} with value {arg}")
    return trace_return

def add(a, b):
    return a + b

sys.settrace(trace_return)
result = add(3, 4)
sys.settrace(None)
print("Result:", result)
```

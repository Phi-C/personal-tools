from perftools.multi_proc_dump import multi_proc_dump

fd = multi_proc_dump(0, "logs")

print("Hello, world!")
import subprocess
subprocess.run(["ls", "-lh"])
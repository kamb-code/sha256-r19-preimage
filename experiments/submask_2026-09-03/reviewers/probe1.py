import numpy as np, warnings
print("numpy", np.__version__)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("uint32 1-2 =", hex(int(np.uint32(1)-np.uint32(2))))
    print("uint32 0xffffffff+2 =", hex(int(np.uint32(0xffffffff)+np.uint32(2))))
    a=np.uint32(0xdeadbeef); b=np.uint32(0xcafebabe)
    print("scalar sub matches python:", int(a-b)==((0xdeadbeef-0xcafebabe)&0xffffffff))
    print("scalar add matches python:", int(a+b)==((0xdeadbeef+0xcafebabe)&0xffffffff))

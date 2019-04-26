import ctypes
import os

# clib = ctypes.cdll.LoadLibrary("libc.dylib")
clib = ctypes.cdll.LoadLibrary("libc.so.6")

_posix_spawn = clib.posix_spawn
_posix_spawn.restype = ctypes.c_int
_posix_spawn.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p)
        )


def spawn_command(cmd):
    shell = "/bin/sh"

    args = [s.encode('utf-8') for s in ["sh", "-c", cmd]] + [0]
    argv = (ctypes.c_char_p * len(args))(*args)

    osenv = [(e + "=" + os.environ[e]).encode('utf-8')
             for e in os.environ] + [0]
    env = (ctypes.c_char_p * len(osenv))(*osenv)

    pid = ctypes.c_int()
    pret = _posix_spawn(
            ctypes.byref(pid), shell.encode('utf-8'),
            ctypes.c_void_p(0), ctypes.c_void_p(0),
            ctypes.cast(argv, ctypes.POINTER(ctypes.c_char_p)),
            ctypes.cast(env, ctypes.POINTER(ctypes.c_char_p)))
    status = ctypes.c_int()
    ret = -1
    if (pret == 0
            and clib.waitpid(pid.value, ctypes.byref(status), 0) != -1):
        ret = status

    return ret

"""Linux x86_64 user-mode thread counters, confined to diagnostic tooling."""
import array
import ctypes
import fcntl
import os
import platform
import struct


def decode_group(raw):
    if len(raw)!=56:
        raise ValueError("unexpected perf group read size")
    nr,enabled,running,v0,id0,v1,id1=struct.unpack("=7Q",raw)
    if nr!=2 or id0==id1 or running>enabled:
        raise ValueError("invalid perf group")
    return enabled,running,{id0:v0,id1:v1}


class ThreadPerf:
    """Cycles/instructions for this thread; excludes kernel and hypervisor."""
    def __init__(self):
        if platform.system()!="Linux" or platform.machine()!="x86_64":
            raise RuntimeError("ThreadPerf requires Linux x86_64")
        self.fds=[]
        self.ids={}
        libc=ctypes.CDLL(None,use_errno=True)
        libc.syscall.restype=ctypes.c_long
        try:
            for config,name in ((0,"cycles"),(1,"instructions")):
                # perf_event_attr v0: group, enabled/running times, event IDs.
                attr=ctypes.create_string_buffer(struct.pack("=IIQQQQQIIQ",
                    0,64,config,0,0,15,1|(1<<5)|(1<<6),0,0,0))
                fd=libc.syscall(ctypes.c_long(298),ctypes.byref(attr),
                    ctypes.c_long(0),ctypes.c_long(-1),
                    ctypes.c_long(self.fds[0] if self.fds else -1),ctypes.c_long(8))
                if fd<0:
                    err=ctypes.get_errno()
                    raise OSError(err,os.strerror(err))
                self.fds.append(fd)
                event_id=array.array("Q",[0])
                fcntl.ioctl(fd,0x80082407,event_id,True)  # PERF_EVENT_IOC_ID
                self.ids[event_id[0]]=name
        except Exception:
            self.close()
            raise

    def read(self):
        return decode_group(os.read(self.fds[0],56))

    def start(self):
        fcntl.ioctl(self.fds[0],0x2403,1)  # RESET group
        self.before=self.read()
        fcntl.ioctl(self.fds[0],0x2400,1)  # ENABLE group

    def stop(self):
        fcntl.ioctl(self.fds[0],0x2401,1)  # DISABLE group
        enabled,running,values=self.read()
        enabled-=self.before[0]
        running-=self.before[1]
        if enabled<=0 or running<=0 or running>enabled or running/enabled<.98:
            raise RuntimeError("perf group did not run for >=98% of enabled window")
        if set(values)!=set(self.ids):
            raise RuntimeError("perf event identity mismatch")
        result={self.ids[key]:value for key,value in values.items()}
        if min(result.values())<=0:
            raise RuntimeError("empty perf counts")
        result.update(enabled_ns=enabled,running_ns=running,
                      running_fraction=running/enabled,
                      cycles_per_instruction=result["cycles"]/result["instructions"],
                      scope="calling thread,user-mode only;raw counts,not scaled")
        return result

    def close(self):
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds=[]

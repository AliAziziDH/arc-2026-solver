import gc
import torch

def force_memory_cleanup():
    print("[Sandbox Governance] gc.collect() executed in memory manager.")
    gc.collect()
    if torch.cuda.is_available():
        print("[Sandbox Governance] torch.cuda.empty_cache() executed.")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

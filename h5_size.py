import h5py
import sys
import numpy as np

def format_size(bytes_val):
    if bytes_val == 0: return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"

def get_real_size(dataset):
    # 1. Handle Scalar Datasets (ndim == 0)
    if dataset.ndim == 0:
        val = dataset[()]
        # If it's bytes (compressed blob), return length
        if isinstance(val, (bytes, np.bytes_)):
            return len(val)
        # If it's a numpy scalar, return nbytes
        if hasattr(val, 'nbytes'):
            return val.nbytes
        return sys.getsizeof(val)

    # 2. Handle Variable-Length Data (Strings OR Compressed Blobs)
    vlen_type = h5py.check_vlen_dtype(dataset.dtype)
    
    if vlen_type is not None or h5py.check_string_dtype(dataset.dtype):
        if dataset.size == 0: return 0
        
        # Iterate to sum actual payload size
        total_bytes = 0
        for x in dataset:
            try:
                total_bytes += len(x)
            except TypeError:
                total_bytes += x.nbytes if hasattr(x, 'nbytes') else sys.getsizeof(x)
        return total_bytes

    # 3. Handle Regular Fixed-Size Arrays (Floats, Ints)
    return dataset.nbytes

def inspect_h5(path):
    # Header
    print(f"{'DATASET':<40} | {'ALLOCATED (Meta)':<18} | {'PAYLOAD (Compressed)':<20}")
    print("-" * 85)
    
    total_payload = 0
    total_meta = 0

    with h5py.File(path, 'r') as f:
        def visitor(name, node):
            nonlocal total_payload, total_meta
            if isinstance(node, h5py.Dataset):
                try:
                    # Calculate sizes
                    payload_bytes = get_real_size(node)
                    meta_bytes = node.id.get_storage_size()
                    
                    # Add to totals
                    total_payload += payload_bytes
                    total_meta += meta_bytes

                    # Print every single entry
                    print(f"/{name:<39} | {format_size(meta_bytes):>16} | {format_size(payload_bytes):>18}")
                        
                except Exception as e:
                    print(f"/{name:<39} | {'ERROR':>16} | {str(e)}")

        f.visititems(visitor)
        
        print("-" * 85)
        print(f"{'TOTAL':<40} | {format_size(total_meta):>16} | {format_size(total_payload):>18}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run inspect_h5.py <file.h5>")
    else:
        inspect_h5(sys.argv[1])
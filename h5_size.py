import h5py
import sys

def get_real_size(dataset):
    # 1. Handle Scalar Datasets (ndim == 0)
    if dataset.ndim == 0:
        # Read the single value using empty tuple index
        val = dataset[()] 
        if isinstance(val, bytes):
            return len(val)
        return dataset.nbytes

    # 2. Handle Variable-Length Strings (Object dtype or vlen string)
    if h5py.check_string_dtype(dataset.dtype):
        # Decode bytes to measure actual text length
        # We iterate only if it's not empty
        if dataset.size == 0:
            return 0
        return sum(len(x) for x in dataset[:].flatten())

    # 3. Handle Regular Numerical Arrays
    return dataset.nbytes

def inspect_h5(path):
    print(f"{'DATASET':<30} | {'REPORTED (Meta)':<15} | {'ACTUAL (Payload)':<15}")
    print("-" * 65)
    
    with h5py.File(path, 'r') as f:
        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                try:
                    real_bytes = get_real_size(node)
                    meta_bytes = node.id.get_storage_size()
                    
                    # Convert to meaningful units
                    real_mb = real_bytes / (1024 * 1024)
                    meta_mb = meta_bytes / (1024 * 1024)
                    
                    # Only print if it uses > 0.1 MB of space (to reduce noise)
                    if real_mb > 0.1 or meta_mb > 0.1: 
                        print(f"/{name:<29} | {meta_mb:>10.2f} MB   | {real_mb:>10.2f} MB")
                except Exception as e:
                    print(f"Error reading {name}: {e}")

        f.visititems(visitor)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run h5_size.py <file.h5>")
    else:
        inspect_h5(sys.argv[1])


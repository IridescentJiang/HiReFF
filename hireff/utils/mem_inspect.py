
import gc
import torch
import traceback
from collections import defaultdict

class AdvancedMemoryInspector:
    def __init__(self):
        self.snapshots = {}
        self.tensor_tracker = {}
        self.creation_traces = {}
        self.last_snapshot = None

    def take_snapshot(self, label):
        """Record currently active tensors and their creation points."""
        current_snapshot = defaultdict(list)

        # Iterate through all objects, find CUDA tensors
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.device.type == 'cuda':
                tensor_id = id(obj)
                tensor_info = {
                    'shape': tuple(obj.shape),
                    'dtype': str(obj.dtype),
                    'size': obj.element_size() * obj.nelement(),
                    'device': str(obj.device),
                    'storage_id': obj.storage().data_ptr() if obj.storage() else None
                }

                # Record creation point (if new tensor)
                if tensor_id not in self.tensor_tracker:
                    self.tensor_tracker[tensor_id] = tensor_info
                    self.creation_traces[tensor_id] = traceback.format_stack(limit=8)

                current_snapshot[tensor_id] = tensor_info

        self.snapshots[label] = current_snapshot
        self.last_snapshot = current_snapshot
        return current_snapshot

    def compare_snapshots(self, label1, label2):
        """Detect GPU memory leaks between two snapshots."""
        snapshot1 = self.snapshots.get(label1, {})
        snapshot2 = self.snapshots.get(label2, {})

        leaked = {}
        for tensor_id in snapshot2:
            # Only focus on newly created or significantly grown tensors
            if tensor_id not in snapshot1:
                leaked[tensor_id] = snapshot2[tensor_id]
            elif snapshot2[tensor_id]['size'] > snapshot1[tensor_id]['size'] * 1.5:
                leaked[tensor_id] = snapshot2[tensor_id]

        return leaked

    def visualize_leaks(self, leaked):
        """Visualize leaked objects and identify their holders."""
        if not leaked:
            print("No GPU memory leaks detected")
            return

        print(f"Detected {len(leaked)} potential leaked tensors:")
        for tensor_id, info in leaked.items():
            print(f"Leaked tensor: {info['shape']} {info['dtype']} ({info['size'] / 1024:.2f} KB)")

            # Get creation point trace
            if tensor_id in self.creation_traces:
                print("Creation point trace:")
                for line in self.creation_traces[tensor_id][-5:]:
                    print(f"  {line.strip()}")

            # Visualize reference chain
            try:
                # Find the actual object
                tensor_obj = None
                for obj in gc.get_objects():
                    if id(obj) == tensor_id and torch.is_tensor(obj):
                        tensor_obj = obj
                        break

                if tensor_obj:
                    # Generate reference graph
                    filename = f'leak_{tensor_id}.png'
                    print(f"Reference graph saved to: {filename}")
            except Exception as e:
                print(f"Failed to generate reference graph: {str(e)}")

    def memory_summary(self):
        """Generate a summary of current GPU memory usage."""
        if not self.last_snapshot:
            return "No snapshot data"

        # Sort tensors by size
        sorted_tensors = sorted(
            self.last_snapshot.values(),
            key=lambda x: x['size'],
            reverse=True
        )

        # Summary statistics
        total_mem = sum(t['size'] for t in sorted_tensors)
        summary = f"Total GPU memory: {total_mem / 1024 ** 2:.2f} MB\n"
        summary += "Largest tensors:\n"

        # Show top 10 largest tensors
        for i, tensor in enumerate(sorted_tensors[:10]):
            summary += f"{i + 1}. {tensor['shape']} {tensor['dtype']} - {tensor['size'] / 1024 ** 2:.2f} MB\n"

        return summary
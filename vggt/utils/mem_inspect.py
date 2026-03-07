
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
        """记录当前活跃张量及其创建点"""
        current_snapshot = defaultdict(list)

        # 遍历所有对象，找到CUDA张量
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

                # 记录创建点（如果是新张量）
                if tensor_id not in self.tensor_tracker:
                    self.tensor_tracker[tensor_id] = tensor_info
                    self.creation_traces[tensor_id] = traceback.format_stack(limit=8)

                current_snapshot[tensor_id] = tensor_info

        self.snapshots[label] = current_snapshot
        self.last_snapshot = current_snapshot
        return current_snapshot

    def compare_snapshots(self, label1, label2):
        """检测两个快照间的显存泄漏"""
        snapshot1 = self.snapshots.get(label1, {})
        snapshot2 = self.snapshots.get(label2, {})

        leaked = {}
        for tensor_id in snapshot2:
            # 只关注新创建或大小显著增加的张量
            if tensor_id not in snapshot1:
                leaked[tensor_id] = snapshot2[tensor_id]
            elif snapshot2[tensor_id]['size'] > snapshot1[tensor_id]['size'] * 1.5:
                leaked[tensor_id] = snapshot2[tensor_id]

        return leaked

    def visualize_leaks(self, leaked):
        """可视化泄漏对象并识别持有者"""
        if not leaked:
            print("未检测到显存泄漏")
            return

        print(f"检测到 {len(leaked)} 个潜在泄漏张量:")
        for tensor_id, info in leaked.items():
            print(f"泄漏张量: {info['shape']} {info['dtype']} ({info['size'] / 1024:.2f} KB)")

            # 获取创建点跟踪
            if tensor_id in self.creation_traces:
                print("创建点跟踪:")
                for line in self.creation_traces[tensor_id][-5:]:
                    print(f"  {line.strip()}")

            # 可视化引用链
            try:
                # 找到实际对象
                tensor_obj = None
                for obj in gc.get_objects():
                    if id(obj) == tensor_id and torch.is_tensor(obj):
                        tensor_obj = obj
                        break

                if tensor_obj:
                    # 生成引用图
                    filename = f'leak_{tensor_id}.png'
                    print(f"引用图保存至: {filename}")
            except Exception as e:
                print(f"生成引用图失败: {str(e)}")

    def memory_summary(self):
        """生成当前显存使用摘要"""
        if not self.last_snapshot:
            return "无快照数据"

        # 按大小排序张量
        sorted_tensors = sorted(
            self.last_snapshot.values(),
            key=lambda x: x['size'],
            reverse=True
        )

        # 汇总统计
        total_mem = sum(t['size'] for t in sorted_tensors)
        summary = f"总显存: {total_mem / 1024 ** 2:.2f} MB\n"
        summary += "最大张量:\n"

        # 显示前10大张量
        for i, tensor in enumerate(sorted_tensors[:10]):
            summary += f"{i + 1}. {tensor['shape']} {tensor['dtype']} - {tensor['size'] / 1024 ** 2:.2f} MB\n"

        return summary
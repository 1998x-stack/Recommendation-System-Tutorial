# 00_4.6.1 “快速”Embedding近邻搜索

"""
Lecture: 第4章 Embedding技术在推荐系统中的应用/4.6 局部敏感哈希——让Embedding插上翅膀的快速搜索方法
Content: 00_4.6.1 “快速”Embedding近邻搜索
"""

import numpy as np
from typing import List, Tuple, Optional

class KDTreeNode:
    """
    kd树节点类。
    
    Attributes:
        point: 数据点（坐标）。
        left: 左子节点。
        right: 右子节点。
        axis: 划分维度。
    """
    def __init__(self, point: np.ndarray, left: Optional['KDTreeNode'] = None, right: Optional['KDTreeNode'] = None, axis: int = 0):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis

class KDTree:
    """
    kd树类，用于构建和搜索kd树。
    
    Attributes:
        root: kd树的根节点。
    """
    def __init__(self, data: np.ndarray):
        """
        初始化kd树。
        
        Args:
            data: 输入数据点集（二维numpy数组）。
        """
        self.root = self._build_tree(data)
    
    def _build_tree(self, data: np.ndarray, depth: int = 0) -> Optional[KDTreeNode]:
        """
        递归构建kd树。
        
        Args:
            data: 输入数据点集（二维numpy数组）。
            depth: 当前树深度。
        
        Returns:
            KDTreeNode: 构建的kd树节点。
        """
        if len(data) == 0:
            return None
        
        k = data.shape[1]
        axis = depth % k
        data = data[data[:, axis].argsort()]
        median_idx = len(data) // 2
        
        return KDTreeNode(
            point=data[median_idx],
            left=self._build_tree(data[:median_idx], depth + 1),
            right=self._build_tree(data[median_idx + 1:], depth + 1),
            axis=axis
        )
    
    def _distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        计算两个点之间的欧氏距离。
        
        Args:
            point1: 第一个点（坐标）。
            point2: 第二个点（坐标）。
        
        Returns:
            float: 欧氏距离。
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _nearest(self, root: KDTreeNode, point: np.ndarray, depth: int, best: Tuple[float, Optional[np.ndarray]]) -> Tuple[float, Optional[np.ndarray]]:
        """
        递归搜索kd树，寻找最近邻点。
        
        Args:
            root: 当前kd树节点。
            point: 目标点（坐标）。
            depth: 当前树深度。
            best: 当前最近邻点和距离。
        
        Returns:
            Tuple[float, Optional[np.ndarray]]: 最短距离和最近邻点坐标。
        """
        if root is None:
            return best
        
        k = point.shape[0]
        axis = depth % k
        current_distance = self._distance(root.point, point)
        
        if current_distance < best[0]:
            best = (current_distance, root.point)
        
        diff = point[axis] - root.point[axis]
        if diff <= 0:
            best = self._nearest(root.left, point, depth + 1, best)
            if diff ** 2 < best[0]:
                best = self._nearest(root.right, point, depth + 1, best)
        else:
            best = self._nearest(root.right, point, depth + 1, best)
            if diff ** 2 < best[0]:
                best = self._nearest(root.left, point, depth + 1, best)
        
        return best
    
    def nearest_neighbor(self, point: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        寻找kd树中与目标点最近的邻居。
        
        Args:
            point: 目标点（坐标）。
        
        Returns:
            Tuple[float, Optional[np.ndarray]]: 最短距离和最近邻点坐标。
        """
        return self._nearest(self.root, point, 0, (float('inf'), None))

# 示例数据
data = np.array([
    [2, 3],
    [5, 4],
    [9, 6],
    [4, 7],
    [8, 1],
    [7, 2]
])

# 构建kd树
kd_tree = KDTree(data)

# 查找最近邻点
point = np.array([3, 4.5])
distance, nearest_point = kd_tree.nearest_neighbor(point)
print(f"最近邻点: {nearest_point}, 距离: {distance}")



# ------------------ 局部敏感哈希 ------------------ #
import numpy as np
from typing import List, Tuple, Dict, Callable

class LSH:
    """
    局部敏感哈希（LSH）类，用于快速近邻搜索。
    
    Attributes:
        k: 哈希函数的数量。
        l: 哈希表的数量。
        hash_tables: 哈希表列表。
        hash_functions: 哈希函数列表。
    """
    
    def __init__(self, k: int, l: int, hash_size: int, input_dim: int):
        """
        初始化LSH。
        
        Args:
            k: 哈希函数的数量。
            l: 哈希表的数量。
            hash_size: 哈希值的大小。
            input_dim: 输入向量的维度。
        """
        self.k = k
        self.l = l
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.hash_tables = [{} for _ in range(l)]
        self.hash_functions = [[self._generate_hash_function(hash_size, input_dim) for _ in range(k)] for _ in range(l)]
    
    def _generate_hash_function(self, hash_size: int, input_dim: int) -> Callable[[np.ndarray], Tuple[int]]:
        """
        生成一个哈希函数。
        
        Args:
            hash_size: 哈希值的大小。
            input_dim: 输入向量的维度。
        
        Returns:
            Callable[[np.ndarray], Tuple[int]]: 哈希函数。
        """
        random_vectors = np.random.randn(hash_size, input_dim)
        return lambda x: tuple((random_vectors @ x) > 0)
    
    def add(self, vec: np.ndarray, label: int):
        """
        将向量添加到哈希表中。
        
        Args:
            vec: 输入向量。
            label: 向量的标签。
        """
        for table, hash_funcs in zip(self.hash_tables, self.hash_functions):
            hash_key = tuple(hash_func(vec) for hash_func in hash_funcs)
            if hash_key in table:
                table[hash_key].append(label)
            else:
                table[hash_key] = [label]
    
    def query(self, vec: np.ndarray, num_neighbors: int = 1) -> List[int]:
        """
        查询与输入向量最相似的邻居。
        
        Args:
            vec: 输入向量。
            num_neighbors: 邻居的数量。
        
        Returns:
            List[int]: 最相似的邻居标签列表。
        """
        candidates = set()
        for table, hash_funcs in zip(self.hash_tables, self.hash_functions):
            hash_key = tuple(hash_func(vec) for hash_func in hash_funcs)
            if hash_key in table:
                candidates.update(table[hash_key])
        
        candidates = list(candidates)
        distances = [np.linalg.norm(vec - candidate) for candidate in candidates]
        nearest_neighbors = [candidates[idx] for idx in np.argsort(distances)[:num_neighbors]]
        return nearest_neighbors

# 示例数据
data = np.random.rand(100, 128)  # 100个128维向量
labels = list(range(100))

# 构建LSH
lsh = LSH(k=10, l=5, hash_size=16, input_dim=128)

# 添加数据到LSH
for vec, label in zip(data, labels):
    lsh.add(vec, label)

# 查询最近邻
query_vec = np.random.rand(128)
neighbors = lsh.query(query_vec, num_neighbors=5)
print(f"查询向量的最近邻：{neighbors}")

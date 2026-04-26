'''
 * @Author       : MatthewZhang
 * @Date         : 2026-03-16 16:14:45
 * @Description  : 
'''
from collections import deque
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorderTraversal(root: Optional[TreeNode]) -> List[int]:
        if not root: return []
        q = list()
        res = []
        while root or q:
            while root:
                q.append(root)
                root = root.left
            res.append(q.pop().val)     # 最左节点的值, 添加并弹出
            if q:
                root = q.pop()              # root指向根节点
                res.append(root.val)
                root = root.right
        return res 

root = TreeNode(
    3,
    TreeNode(9),
    TreeNode(20, TreeNode(15), TreeNode(7))
)

res = inorderTraversal(root)
print(res)


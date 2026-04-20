'''
 * @Author       : MatthewZhang
 * @Date         : 2026-03-16 16:14:45
 * @Description  : 
'''
from collections import deque
from typing import Optional, List


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# def averageOfLevels(root: Optional[TreeNode]) -> List[float]:
#         # 大致初始思路: 用一个变量记录层数, 当遍历到这一层的最后一个节点时, 且下面还有节点, 层数+1,cur_sum, 当遍历同层时记录节点个数, 同时用cur_sum统计和 -> 怎么判断是否遍历到在这一层的最后一个节点?
#         # 关键点不是上面的问题, 而是:每一轮要明确知道当前层一共有多少个节点。
#         if not root:
#             return []
#         q = deque([])
#         res = []
#         # 添加节点
#         q.append(root)
#         while q:
#             num = len(q)
#             cur_sum = 0
#             for i in range(num):
#                 cur_sum += q[0].val
#                 node = q[0]
#                 if node.left: q.append(node.left)
#                 if node.right: q.append(node.right)
#                 q.popleft()
#             # 到达当前层最后一个节点了, 开始计算平均值
#             res.append(cur_sum / num)
#         return res

# root = TreeNode(
#     3,
#     TreeNode(9),
#     TreeNode(20, TreeNode(15), TreeNode(7))
# )

# res = averageOfLevels(root)
# print(res)

import time
def slow_io(name : str) -> float:
    t1 = time.perf_counter()
    time.sleep(2)
    t2 = time.perf_counter()
    print(f"{name} time consumed: {t2 - t1} s")
    return t2 - t1

counter = 0
def my_calculator():
    global counter
    temp = counter          # 读
    time.sleep(0.001)        # 撑开窗口
    counter = temp + 1      # 写

if __name__ == "__main__":
    # 串行
    # s1, s2, s3 = 'A', 'B', 'C'
    # t1 = slow_io(s1)
    # t2 = slow_io(s2)
    # t3 = slow_io(s3)
    # print(f"total time consumed: {t1 + t2 + t3} s")

    # with 本质上等价于帮你自动调用 shutdown(), 如果不用with就得按下面方式写
    # 为什么需要shutdown? -> 站在线程池设计者角度看，默认假设是“你可能还要继续用”，所以它不会因为某一个任务跑完就自作主张销毁自己。
    # executor = ThreadPoolExecutor(max_workers=3)
    # try:
    #     future = executor.submit(task, 10)
    #     print(future.result())
    # finally:
    #     executor.shutdown()

    # max_workers 规定的是并发上限，不是调用次数, submit 决定“有几个任务”, 是非阻塞的, [入参:需要执行的函数, 该函数的入参] 取result()还原为函数返回原始类型, 它还会把子线程里的异常重新抛出来
    # submit的结果算是个中间态, submit是异步的, 意思说f1/2/3会立刻得到这个对象, 添加到futures中
    # from concurrent.futures import ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     t3 = time.perf_counter()
    #     f1 = executor.submit(slow_io, 'A')
    #     f2 = executor.submit(slow_io, 'B')
    #     f3 = executor.submit(slow_io, 'C')
    #     futures = [f1, f2, f3]
    #     for future in futures:
    #         future.result()
    #     t4 = time.perf_counter()
    #     print(f"total time consumed: {t4-t3} s")


    from concurrent.futures import ThreadPoolExecutor
    futures = []
    # 退出 with 时会自动等待线程池任务完成
    with ThreadPoolExecutor(max_workers=100) as executor:
        for i in range(100):
            futures.append(executor.submit(my_calculator))

        for f in futures:
            f.result()
    print(counter)
    # 长 sleep 倾向于“错误但集中”，短 sleep 倾向于“部分正确但更乱”


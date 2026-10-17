"""跳表最小可运行演示 —— 用来看清「每一行代码到底调用了谁」。

跑法：
    python skiplist_demo.py

注意：跳表的层数是**随机**的，所以不固定种子的话，每次 show() 出来的形状都不一样。
下面用 random.seed(3) 固定，保证你看到的和我看到的一致。
"""

import random

MAX_L = 16   # 层数天花板
P = 0.5      # 抛硬币晋升概率


class Node:
    def __init__(self, val, level):
        self.val = val
        self.next = [None] * level


class SkipList:
    def __init__(self):
        self.head = Node(float("-inf"), MAX_L)   # 哨兵/起跑线
        self.level = 1                           # 当前水位

    def _random_level(self):
        lv = 1
        while random.random() < P and lv < MAX_L:
            lv += 1
        return lv

    def insert(self, val):
        # ① 从最高层往下扫，记录每层的前驱
        update = [None] * MAX_L
        cur = self.head
        for i in range(self.level - 1, -1, -1):
            while cur.next[i] is not None and cur.next[i].val < val:
                cur = cur.next[i]
            update[i] = cur

        # ② 抛硬币决定新节点几层高
        lv = self._random_level()

        # ③ 如果新节点比当前所有节点都高，抬升水位
        if lv > self.level:
            for i in range(self.level, lv):
                update[i] = self.head
            self.level = lv

        # ④ 接线
        node = Node(val, lv)
        for i in range(lv):
            node.next[i] = update[i].next[i]
            update[i].next[i] = node

    def search(self, target):
        path = []
        cur = self.head
        for i in range(self.level - 1, -1, -1):
            nxt = cur.next[i]
            while nxt is not None and nxt.val < target:
                path.append(f"  第{i}层: {cur.val} → {nxt.val}   (比{target}小，往右走)")
                cur = nxt
                nxt = cur.next[i]
            if nxt is not None and nxt.val == target:
                path.append(f"  第{i}层: {cur.val} → {nxt.val}   ★命中！")
                return path
            right = nxt.val if nxt is not None else "None"
            path.append(f"  第{i}层: 右边是 {right}，不能再往右 → 下降一层")
        path.append("  结论：不存在")
        return path

    def show(self):
        lines = []
        for i in range(self.level - 1, -1, -1):
            row = [f"第{i}层: "]
            cur = self.head
            while cur is not None:
                row.append(str(cur.val))
                cur = cur.next[i]
            lines.append(" → ".join(row) + " → None")
        return "\n".join(lines)


# ===================== 你问的那段代码 =====================
random.seed(3)          # 固定随机数，保证输出可复现

sl = SkipList()
for v in [1, 3, 4, 5, 6, 7, 8, 9]:
    sl.insert(v)
print(sl.show())

print("\n查找 7 的走法：")
print("\n".join(sl.search(7)))

"""把 insert() 的每一步都打印出来 —— 用真实数字看清「扫描 → 摇号 → 接线」全过程。

跑法：python insert_trace.py
"""

import random

MAX_L = 16
P = 0.5


class Node:
    def __init__(self, val, level):
        self.val = val
        self.next = [None] * level


def nm(node):
    """把一个节点变成能打印的文字"""
    if node is None:
        return "None"
    return str(node.val)


class SkipList:
    def __init__(self):
        self.head = Node(float("-inf"), MAX_L)
        self.level = 1

    def random_level(self):
        lv = 1
        while True:
            r = random.random()
            if r < P and lv < MAX_L:
                print(f"      抛出 {r:.3f} < 0.5 → 正面，升到 {lv + 1} 层")
                lv += 1
            else:
                print(f"      抛出 {r:.3f} ≥ 0.5 → 反面，停在 {lv} 层")
                return lv

    def insert(self, val):
        print(f"\n{'=' * 55}")
        print(f"  插入 {val}   （当前水位 level = {self.level}，可用层 0..{self.level - 1}）")
        print(f"{'=' * 55}")

        # ---------- 前半段：扫描，记录 update ----------
        print("  【第1步】从上往下扫描，记录每层的前驱 update[i]")
        update = [self.head] * MAX_L
        cur = self.head
        for i in range(self.level - 1, -1, -1):
            print(f"    第{i}层: 从 {nm(cur)} 出发")
            while cur.next[i] is not None and cur.next[i].val < val:
                print(f"          {nm(cur)} 的右边是 {nm(cur.next[i])}，{nm(cur.next[i])} < {val} → 往右走")
                cur = cur.next[i]
            why = "右边是 None（到头了）" if cur.next[i] is None else f"右边 {nm(cur.next[i])} ≥ {val}（太大）"
            print(f"          {why} → 停。update[{i}] = {nm(cur)}")
            update[i] = cur

        # ---------- 摇号 ----------
        print("  【第2步】抛硬币决定层数")
        lvl = self.random_level()
        print(f"          → 结果：占 {lvl} 层（出现在第 0..{lvl - 1} 层）")

        # ---------- 接线 ----------
        new = Node(val, lvl)
        print("  【第3步】逐层接线（先接后面，再接前面）")
        for i in range(lvl):
            after = nm(update[i].next[i])
            print(f"    第{i}层: {nm(update[i])} ──→ {after}")
            new.next[i] = update[i].next[i]      # ④ 新节点右手拉住后面那个
            update[i].next[i] = new              # ⑤ 前驱改拉新节点
            print(f"          变成: {nm(update[i])} ──→ {nm(new)} ──→ {after}")

        # ---------- 水位 ----------
        old = self.level
        self.level = max(self.level, lvl)
        print(f"  【第4步】水位: max({old}, {lvl}) = {self.level}"
              + ("   ← 上涨了！" if self.level > old else "   （不变）"))

    def show(self):
        out = []
        for i in range(self.level - 1, -1, -1):
            row, cur = [], self.head
            while cur is not None:
                row.append(nm(cur))
                cur = cur.next[i]
            out.append(f"    第{i}层: " + " → ".join(row) + " → None")
        return "\n".join(out)


random.seed(7)

sl = SkipList()
for v in [3, 1, 7, 5]:
    sl.insert(v)
    print("\n  【当前跳表长这样】")
    print(sl.show())

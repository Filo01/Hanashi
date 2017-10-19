class Node(object):

    def __init__(self, parent, n, contour = None, box=None):
        self.children = []
        self.parent = parent
        self.n = n
        self.contour = contour
        self.box = box

    def add_child(self, node):
        self.children.append(node)

    def __eq__(self, other):
        return self.n == other.n

    def __str__(self):
        return "Node(" + str(self.n) + ")"


class Tree():

    def __init__(self):
        self.root = Node(-2, -1)
        self.indexes = []

    def get(self, n):
        queue = []
        queue.append(self.root)
        while(len(queue)>0):
            node1 = queue.pop(0)
            if node1.n == n:
                return node1
            for child in node1.children:
                queue.append(child)
        return None

    def add(self, node):
        found = self.get(node.n)
        if found:
            return
        parent_found = self.get(node.parent)
        if parent_found:
            parent_found.add_child(node)
            return
        else:
            self.root.add_child(Node(-1, node.n, node.contour, node.box))

    def level_order_traversal(self):
        queue = list()
        queue.append((self.root, 0))
        while len(queue) > 0:
            node, level = queue.pop(0)
            yield (node, level)
            level += 1
            for child in node.children:
                queue.append((child, level))

    def __len__(self):
        n=0
        for node, level in self.level_order_traversal():
            n+=1
        return n



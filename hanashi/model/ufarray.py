# Array based union find data structure

# P: The array, which encodes the set membership of all the elements

class UFarray:
    def __init__(self, size=None):
        # Array which holds label -> set equivalences
        self.P = []
        # Name of the next label, when one is created
        self.label = 0

        # Initialize the array
        if size:
            self.setLabel(size - 1)

    def makeLabel(self):
        r = self.label
        self.label += 1
        self.P.append(r)
        return r

    def setLabel(self, n):
        self.label = n
        if n > len(self.P) - 1:
            for i in range(len(self.P), n + 1):
                self.P.append(i)

    # Makes all nodes "in the path of node i" point to root
    def setRoot(self, i, root):
        while self.P[i] < i:
            j = self.P[i]
            self.P[i] = root
            i = j
        self.P[i] = root

    # Finds the root node of the tree containing node i
    def findRoot(self, i):
        while self.P[i] < i:
            i = self.P[i]
        return i

    # Finds the root of the tree containing node i
    # Simultaneously compresses the tree
    def find(self, i):
        root = self.findRoot(i)
        self.setRoot(i, root)
        return root

    # Joins the two trees containing nodes i and j
    # Modified to be less agressive about compressing paths
    # because performance was suffering some from over-compression
    def union(self, i, j):
        if i != j:
            root = self.findRoot(i)
            rootj = self.findRoot(j)
            if root > rootj: root = rootj
            self.setRoot(j, root)
            self.setRoot(i, root)

    def flatten(self):
        for i in range(1, len(self.P)):
            self.P[i] = self.P[self.P[i]]

    def root_dictionary(self):

        temp = dict()
        for i in range(len(self.P)):
            root = self.P[i]
            children = temp.get(root, list())
            children.append(i)
            temp[root] = children
        keys = list(temp.keys())
        for e in keys:
            if len(temp[e]) == 1 and e != temp[e][0]:
                del temp[e]

        return temp

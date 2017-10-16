from hanashi.model.rectangle import Rectangle

MAX_OBJECTS = 15
MAX_LEVELS = 5


class Quadtree(object):
    level = None
    bounds = None
    objects = []
    nodes = [None] * 4

    def __init__(self, level, bounds):
        """
        init Node object
        :param level: current level
        :param bounds: @Rectangle
        """
        self.level = level
        self.bounds = bounds
        self.objects = []
        self.nodes = [None] * 4

    def clear(self):
        self.objects = []
        for n in self.nodes:
            if n is not None:
                n.clear()

    def split(self):
        x = self.bounds.x
        y = self.bounds.y
        subWidth = self.bounds.width / 2
        subHeight = self.bounds.height / 2

        self.nodes[0] = Quadtree(self.level + 1, Rectangle(x + subWidth, y, subWidth, subHeight))
        self.nodes[1] = Quadtree(self.level + 1, Rectangle(x, y, subWidth, subHeight))
        self.nodes[2] = Quadtree(self.level + 1, Rectangle(x, y + subHeight, subWidth, subHeight))
        self.nodes[3] = Quadtree(self.level + 1, Rectangle(x + subWidth, y + subHeight, subWidth, subHeight))

    def get_index(self, rectangle):
        index = -1
        vertical_midpoint = self.bounds.x + self.bounds.width / 2
        horizontal_midpoint = self.bounds.y + self.bounds.height / 2

        top_quadrants = rectangle.y < horizontal_midpoint and rectangle.y + rectangle.height < horizontal_midpoint
        bottom_quadrants = rectangle.y > horizontal_midpoint

        if rectangle.x < vertical_midpoint and rectangle.x + rectangle.width < vertical_midpoint:
            if top_quadrants:
                index = 1
            elif bottom_quadrants:
                index = 2
        elif rectangle.x > vertical_midpoint:
            if top_quadrants:
                index = 0
            elif bottom_quadrants:
                index = 3

        return index

    def insert(self, rectangle):
        if self.nodes[0] is not None:
            index = self.get_index(rectangle)
            if index != -1:
                self.nodes[index].insert(rectangle)
                return

        self.objects.append(rectangle)

        if len(self.objects) > MAX_OBJECTS and self.level < MAX_LEVELS:
            if self.nodes[0] is None:
                self.split()
            i = 0
            while i < len(self.objects):
                index = self.get_index(self.objects[i])
                if index != -1:
                    self.nodes[index].insert(self.objects.pop(i))
                else:
                    i += 1

    def retrieve(self, returned_objects, rectangle):
        index = self.get_index(rectangle)
        if index != -1 and self.nodes[0] is not None:
            self.nodes[index].retrieve(returned_objects, rectangle)

        for o in self.objects:
            returned_objects.append(o)
        return returned_objects

    def draw(self, draw):
        draw.rectangle(
            (self.bounds.x, self.bounds.y, self.bounds.x + self.bounds.width, self.bounds.y + self.bounds.height),
            outline="green")
        for o in self.objects:
            draw.rectangle((o.x, o.y, o.x + o.width, o.y + o.height), outline="white")
            pass
        for n in self.nodes:
            if n is not None:
                draw = n.draw(draw)

        return draw

    def __str__(self):
        rectangles = " ".join([r.__str__() for r in self.objects])
        nodes = []
        index = 0
        for n in self.nodes:
            if n:
                nodes.append("Index " + str(index) + " " + n.__str__())
            index += 1
        indentation = ["\t"] * self.level
        s = "".join(indentation) + "Level " + str(self.level) + " Rectangles: " + rectangles + "\n" + "\n".join(nodes)
        return s

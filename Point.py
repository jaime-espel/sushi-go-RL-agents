class Point:
    def __init__(self, x=0, y=0, point=None):
        if point:
            if not isinstance(point, Point):
                raise TypeError("point is not an instance of Point")
            self.x = point.x
            self.y = point.y
        else:
            self.x = x
            self.y = y

    def get_point(self):
        return (self.x, self.y)

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_point(self, x, y):
        self.x = x
        self.y = y

    def copy(self, point):
        if isinstance(point, Point):
            self.x = point.x
            self.y = point.y

    def incr_x(self, incr=1):
        self.x += incr

    def incr_y(self, incr=1):
        self.y += incr

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
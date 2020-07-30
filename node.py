class Node:
    class_name = None
    is_leaf = True
    children = []
    data = None
    is_parent = True

    def __init__(self, class_name):
        self.class_name = class_name

    def set_data(self, data):
        self.data = data

    def set_new_child(self, child):
        self.children.append(Node(child))

    def is_leaf(self, is_leaf):
        self.is_leaf

    def is_parent(self, is_parent):
        self.is_parent = is_parent
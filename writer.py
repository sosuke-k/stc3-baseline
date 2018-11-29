

class Writer(object):

    def __init__(self, path):
        self.path = path

    def write(self, values):
        f = open(self.path, "a")
        line = "\t".join(map(str, values)) + "\n"
        f.write(line)
        f.close()

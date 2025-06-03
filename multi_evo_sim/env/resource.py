class Resource:
    """Representa un recurso dentro del mundo."""

    def __init__(self, position, value=1):
        self.position = position
        self.value = value
        self.consumed = False

    def consume(self):
        if not self.consumed:
            self.consumed = True
            return self.value
        return 0

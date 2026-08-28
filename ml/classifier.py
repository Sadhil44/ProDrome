# stub classifier


class StubClassifier:
    def predict(self, window):
        return "MEMORY_LEAK", 0.9


classifier = StubClassifier()

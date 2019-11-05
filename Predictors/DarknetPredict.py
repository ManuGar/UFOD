from Predictors.IPredictor import IPredictor

class DarknetPredict(IPredictor):
    def __init__(self, weights):
        self.weights = weights
    def predict(self):
        pass


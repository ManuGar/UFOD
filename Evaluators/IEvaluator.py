class IEvaluator(object):
    def __init__(self, predictor, dataset_name, dataset_path):
        self.predictor = predictor
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
    def evaluate(self):
        pass

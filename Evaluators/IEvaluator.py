class IEvaluator(object):
    def __init__(self, predictor, dataset_name, dataset_path,model_name):
        self.predictor = predictor
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.model_name = model_name
    def evaluate(self):
        pass

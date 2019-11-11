from objectDetectors.MXNetObjectDetector.MxNetDetector import MxNetDetector, VOCLike
from objectDetectors.MXNetObjectDetector import functions as fn
from Predictors.MxNetPredict import MxNetPredict
from Evaluators.MapEvaluator import MapEvaluator as Map
# from gluoncv import model_zoo, data, utils
from mxnet import gluon, autograd
# from mxnet.gluon.model_zoo import vision as models
# from mxnet.gluon import data, utils
import gluoncv as gcv
import mxnet as mx
import os
import time
import shutil



class SSDMxnet(MxNetDetector):
    def __init__(self, dataset_path, dataset_name, model):
        super(SSDMxnet, self).__init__(dataset_path, dataset_name)
        self.model = model
        # MxNetDetector.__init__(self, dataset_path, dataset_name)

    # def transform(self):
    #     MxNetDetector.transform(self)
    #
    # def organize(self, train_percentage):
    #     MxNetDetector.organize(self, train_percentage)

    # def createModel(self):
    #     pass
    def train(self, framework_path = None, n_gpus = 1):
        # dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
        n_epoch = 10
        classes = fn.readClasses(os.path.join(self.OUTPUT_PATH,"VOC" + self.DATASET_NAME))
        # En este caso debera ser el output path que es donde se guardo el dataset preparado
        # classes = fn.readClasses(os.path.join(self.DATASET,"VOC" + self.DATASET_NAME))
        # MXNET_ENABLE_GPU_P2P = 0
        n_gpu = mx.context.num_gpus()
        # try:
        ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)] if n_gpu == 4 else [mx.gpu(0), mx.gpu(1),
                                                                                   mx.gpu(2)] if n_gpu == 3 else [
                mx.gpu(0), mx.gpu(1)] if n_gpu == 2 else [mx.gpu(0)]
        # except:
        #     ctx = [mx.cpu()]
        # Como en el anterior caso debe ser output_path por que es donde se ha guardado el dataset partido en entrenamiento y test
        dataset = VOCLike(root=self.OUTPUT_PATH, splits=((self.DATASET_NAME, 'train'),),classes=classes)
        # Revisar esto para que funcione con varios modelos que se entrenan igual
        # net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes,
        #                               pretrained_base=False, transfer='voc')
        net = gcv.model_zoo.get_model(self.model, classes=classes,
                                      pretrained_base=False, transfer='voc')
        train_data = fn.get_dataloader(net, dataset, 512, 16, 0)
        net.collect_params().reset_ctx(ctx)
        trainer = gluon.Trainer(
            net.collect_params(), 'sgd',
            {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})

        mbox_loss = gcv.loss.SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')
        for epoch in range(0, n_epoch):
            ce_metric.reset()
            smoothl1_metric.reset()
            # May be useful to calculate exec time
            # tic = time.time()
            btic = time.time()
            net.hybridize(static_alloc=True, static_shape=True)
            for i, batch in enumerate(train_data):
                batch_size = batch[0].shape[0]
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
                with autograd.record():
                    cls_preds = []
                    box_preds = []
                    for x in data:
                        cls_pred, box_pred, _ = net(x)
                        cls_preds.append(cls_pred)
                        box_preds.append(box_pred)
                    sum_loss, cls_loss, box_loss = mbox_loss(
                        cls_preds, box_preds, cls_targets, box_targets)
                    autograd.backward(sum_loss)
                # since we have already normalized the loss, we don't want to normalize
                # by batch-size anymore
                trainer.step(1)
                ce_metric.update(0, [l * batch_size for l in cls_loss])
                smoothl1_metric.update(0, [l * batch_size for l in box_loss])
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                if i % 20 == 0:
                    print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                        epoch, i, batch_size / (time.time() - btic), name1, loss1, name2, loss2))
                btic = time.time()
            if (epoch % 5 == 0):
                net.save_parameters(self.model + '_' + self.DATASET_NAME + '_' + str(epoch) + '.params')
        net.save_parameters(self.model + '_' + self.DATASET_NAME + '_final.params')
        shutil.rmtree(os.path.join(self.OUTPUT_PATH,"VOC"+self.DATASET_NAME))
        # https://github.com/apache/incubator-mxnet/tree/master/example/ssd

    def evaluate(self, framework_path = None):
        mxnetPredict = MxNetPredict(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.model +"_" +self.DATASET_NAME + "_final.params"),
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"),
            self.model)

        map = Map(mxnetPredict, self.DATASET_NAME, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME), self.model)
        map.evaluate()

def main():
    pass

if __name__ == "__main__":
    main()
#!/bin/sh
# Carga de librerías.
module load cuda/10.1.105
module load cudnn/7.6
module load python
# Configuración PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:/home/jheras/ws/holms/models/research:/home/jheras/ws/holms/models/research/slim

# Configuración de parámetros para entrenar.
PIPELINE_CONFIG_PATH=/home/jheras/ws/holms/models/research/pets/faster_rcnn_resnet101_pets.config
MODEL_DIR=/home/jheras/ws/holms/models/research/pets/faster_rcnn_resnet101_coco_11_06_2017/
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
# Instrucción que entrena modelo tensorflow.
#python3 object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES --alsologtostderr --worker_replicas=2 --num_clones=2 --ps_tasks=1
#exit 0

import os

EPOCHS = 1000
BATCH_SIZE = 32
ratio = 0.8
dataset = "/home/minhpv/Desktop/tools/data_gens"
NUM_CLASSES = len(os.listdir("/home/minhpv/Desktop/tools/data_gens"))

label_dict = {
	"id": 0,
	"name": 1,
	"bod": 2,
	"other": 3,
}
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
# configs
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	# Invalid device or cannot modify virtual devices once initialized.
	pass
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
df_train = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train.csv")
df_train["Pawpularity"] = str(df_train["Pawpularity"] / 10)
df_train["Id"] = df_train["Id"] + ".jpg"
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	validation_split=0.2)


class cfg:
	img_size = 384
	train_batch_size = 16
	val_batch_size = 16


# data

train_generator = train_datagen.flow_from_dataframe(
	df_train,
	directory=r'F:\Pycharm_projects\PetFinder\data\train',
	x_col="Id",
	y_col="Pawpularity",
	class_mode="raw",
	target_size=(cfg.img_size, cfg.img_size),
	batch_size=cfg.train_batch_size,
	subset="training")

train_generator_valid = train_datagen.flow_from_dataframe(
	df_train,
	directory=r'F:\Pycharm_projects\PetFinder\data\train',
	x_col="Id",
	y_col="Pawpularity",
	class_mode="raw",
	target_size=(cfg.img_size, cfg.img_size),
	batch_size=cfg.val_batch_size,
	subset="validation")


def make_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((7, 7, 256)))
	assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	assert model.output_shape == (None, 7, 7, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 14, 14, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 28, 28, 1)

	return model

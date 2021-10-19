import pandas as pd, numpy as np, random, os, shutil
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
import sklearn
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import imagesize
import wandb
import yaml
import sys

sys.path.append('../input/swintransformertf')
from swintransformer import SwinTransformer

from vit_keras import vit
from IPython import display as ipd
from glob import glob
from tqdm.notebook import tqdm
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

print('np:', np.__version__)
print('pd:', pd.__version__)
print('sklearn:', sklearn.__version__)
print('tf:', tf.__version__)
print('tfa:', tfa.__version__)
print('w&b:', wandb.__version__)
import wandb

try:
	from kaggle_secrets import UserSecretsClient

	user_secrets = UserSecretsClient()
	api_key = user_secrets.get_secret("WANDB")
	wandb.login(key=api_key)
	anonymous = None
except:
	anonymous = "must"
	print(
		'To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')


class CFG:
	wandb = True
	competition = 'petfinder'
	_wandb_kernel = 'awsaf49'
	debug = False
	exp_name = 'vit_b16+cls-aug1'  # name of the experiment, folds will be grouped using 'exp_name'

	# USE verbose=0 for silent, vebose=1 for interactive, verbose=2 for commit
	verbose = 1 if debug else 0
	display_plot = True

	device = "TPU"  # or "GPU"

	model_name = 'swin_large_384'  # 'vit_b32'

	# USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
	seed = 42

	# NUMBER OF FOLDS. USE 2, 5, 10
	folds = 10

	# FOLDS TO TRAIN
	selected_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	# IMAGE SIZE
	img_size = [512, 512]

	# BATCH SIZE AND EPOCHS
	batch_size = 32
	epochs = 5

	# LOSS
	loss = 'BCE'
	optimizer = 'Adam'

	# CFG.augmentATION
	augment = True
	transform = True

	# TRANSFORMATION
	fill_mode = 'reflect'
	rot = 10.0
	shr = 5.0
	hzoom = 30.0
	wzoom = 30.0
	hshift = 30.0
	wshift = 30.0

	# FLIP
	hflip = True
	vflip = True

	# CLIP [0, 1]
	clip = False

	# LEARNING RATE SCHEDULER
	scheduler = 'exp'  # Cosine

	# Dropout
	drop_prob = 0.75
	drop_cnt = 10
	drop_size = 0.05

	# bri, contrast
	sat = [0.7, 1.3]
	cont = [0.8, 1.2]
	bri = 0.15
	hue = 0.05

	# TEST TIME CFG.augmentATION STEPS
	tta = 1

	tab_cols = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
	            'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
	target_col = ['Pawpularity']


def seeding(SEED):
	np.random.seed(SEED)
	random.seed(SEED)
	os.environ['PYTHONHASHSEED'] = str(SEED)
	#     os.environ['TF_CUDNN_DETERMINISTIC'] = str(SEED)
	tf.random.set_seed(SEED)
	print('seeding done!!!')


seeding(CFG.seed)
if CFG.device == "TPU":
	print("connecting to TPU...")
	try:
		tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
		print('Running on TPU ', tpu.master())
	except ValueError:
		print("Could not connect to TPU")
		tpu = None

	if tpu:
		try:
			print("initializing  TPU ...")
			tf.config.experimental_connect_to_cluster(tpu)
			tf.tpu.experimental.initialize_tpu_system(tpu)
			strategy = tf.distribute.experimental.TPUStrategy(tpu)
			print("TPU initialized")
		except _:
			print("failed to initialize TPU")
	else:
		CFG.device = "GPU"

if CFG.device != "TPU":
	print("Using default strategy for CPU and single GPU")
	strategy = tf.distribute.get_strategy()

if CFG.device == "GPU":
	print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')
BASE_PATH = '/kaggle/input/petfinder-pawpularity-score'
GCS_PATH = KaggleDatasets().get_gcs_path('petfinder-pawpularity-score')


# custom function
class Mish(tf.keras.layers.Activation):
	def __init__(self, activation, **kwargs):
		super(Mish, self).__init__(activation, **kwargs)
		self.__name__ = 'Mish'


def mish(inputs):
	return inputs * tf.math.tanh(tf.math.softplus(inputs))


tf.keras.utils.get_custom_objects().update({'Mish': Mish(mish)})


def get_imgsize(row):
	width, height = imagesize.get(row['image_path'].replace(GCS_PATH, BASE_PATH))
	row['width'] = width
	row['height'] = height
	return row


# Train Data
df = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
df['image_path'] = GCS_PATH + '/train/' + df.Id + '.jpg'
tqdm.pandas(desc='train')
df = df.progress_apply(get_imgsize, axis=1)
display(df.head(2))

# Test Data
test_df = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')
test_df['image_path'] = GCS_PATH + '/test/' + test_df.Id + '.jpg'
tqdm.pandas(desc='test')
test_df = test_df.progress_apply(get_imgsize, axis=1)

display(test_df.head(2))
print('train_files:', df.shape[0])
print('test_files:', test_df.shape[0])
num_bins = int(np.floor(1 + np.log2(len(df))))
df["bins"] = pd.cut(df[CFG.target_col].values.reshape(-1), bins=num_bins, labels=False)

skf = StratifiedKFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["bins"])):
	df.loc[val_idx, 'fold'] = fold
display(df.groupby(['fold', "bins"]).size())


def get_mat(shear, height_zoom, width_zoom, height_shift, width_shift):
	# returns 3x3 transformmatrix which transforms indicies

	# CONVERT DEGREES TO RADIANS
	# rotation = math.pi * rotation / 180.
	shear = math.pi * shear / 180.

	def get_3x3_mat(lst):
		return tf.reshape(tf.concat([lst], axis=0), [3, 3])

	# ROTATION MATRIX
	#     c1   = tf.math.cos(rotation)
	#     s1   = tf.math.sin(rotation)
	one = tf.constant([1], dtype='float32')
	zero = tf.constant([0], dtype='float32')

	#     rotation_matrix = get_3x3_mat([c1,   s1,   zero,
	#                                    -s1,  c1,   zero,
	#                                    zero, zero, one])
	# SHEAR MATRIX
	c2 = tf.math.cos(shear)
	s2 = tf.math.sin(shear)

	shear_matrix = get_3x3_mat([one, s2, zero,
	                            zero, c2, zero,
	                            zero, zero, one])
	# ZOOM MATRIX
	zoom_matrix = get_3x3_mat([one / height_zoom, zero, zero,
	                           zero, one / width_zoom, zero,
	                           zero, zero, one])
	# SHIFT MATRIX
	shift_matrix = get_3x3_mat([one, zero, height_shift,
	                            zero, one, width_shift,
	                            zero, zero, one])

	return K.dot(shear_matrix, K.dot(zoom_matrix,
	                                 shift_matrix))  # K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))


def transform(image, DIM=CFG.img_size):  # [rot,shr,h_zoom,w_zoom,h_shift,w_shift]):
	if DIM[0] != DIM[1]:
		pad = (DIM[0] - DIM[1]) // 2
		image = tf.pad(image, [[0, 0], [pad, pad + 1], [0, 0]])

	NEW_DIM = DIM[0]

	rot = CFG.rot * tf.random.normal([1], dtype='float32')
	shr = CFG.shr * tf.random.normal([1], dtype='float32')
	h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / CFG.hzoom
	w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / CFG.wzoom
	h_shift = CFG.hshift * tf.random.normal([1], dtype='float32')
	w_shift = CFG.wshift * tf.random.normal([1], dtype='float32')

	transformation_matrix = tf.linalg.inv(get_mat(shr, h_zoom, w_zoom, h_shift, w_shift))

	flat_tensor = tfa.image.transform_ops.matrices_to_flat_transforms(transformation_matrix)

	image = tfa.image.transform(image, flat_tensor, fill_mode=CFG.fill_mode)

	rotation = math.pi * rot / 180.

	image = tfa.image.rotate(image, -rotation, fill_mode=CFG.fill_mode)

	if DIM[0] != DIM[1]:
		image = tf.reshape(image, [NEW_DIM, NEW_DIM, 3])
		image = image[:, pad:DIM[1] + pad, :]
	image = tf.reshape(image, [*DIM, 3])
	return image


def dropout(image, DIM=CFG.img_size, PROBABILITY=0.6, CT=5, SZ=0.1):
	# input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
	# output - image with CT squares of side size SZ*DIM removed

	# DO DROPOUT WITH PROBABILITY DEFINED ABOVE
	P = tf.cast(tf.random.uniform([], 0, 1) < PROBABILITY, tf.int32)
	if (P == 0) | (CT == 0) | (SZ == 0):
		return image

	for k in range(CT):
		# CHOOSE RANDOM LOCATION
		x = tf.cast(tf.random.uniform([], 0, DIM[1]), tf.int32)
		y = tf.cast(tf.random.uniform([], 0, DIM[0]), tf.int32)
		# COMPUTE SQUARE
		WIDTH = tf.cast(SZ * min(DIM), tf.int32) * P
		ya = tf.math.maximum(0, y - WIDTH // 2)
		yb = tf.math.minimum(DIM[0], y + WIDTH // 2)
		xa = tf.math.maximum(0, x - WIDTH // 2)
		xb = tf.math.minimum(DIM[1], x + WIDTH // 2)
		# DROPOUT IMAGE
		one = image[ya:yb, 0:xa, :]
		two = tf.zeros([yb - ya, xb - xa, 3], dtype=image.dtype)
		three = image[ya:yb, xb:DIM[1], :]
		middle = tf.concat([one, two, three], axis=1)
		image = tf.concat([image[0:ya, :, :], middle, image[yb:DIM[0], :, :]], axis=0)
		image = tf.reshape(image, [*DIM, 3])

	#     image = tf.reshape(image,[*DIM,3])
	return image


def build_decoder(with_labels=True, target_size=CFG.img_size, ext='jpg'):
	def decode(path):
		file_bytes = tf.io.read_file(path)
		if ext == 'png':
			img = tf.image.decode_png(file_bytes, channels=3)
		elif ext in ['jpg', 'jpeg']:
			img = tf.image.decode_jpeg(file_bytes, channels=3)
		else:
			raise ValueError("Image extension not supported")

		img = tf.image.resize(img, target_size)
		img = tf.cast(img, tf.float32) / 255.0
		img = tf.reshape(img, [*target_size, 3])

		return img

	def decode_with_labels(path, label):
		return decode(path), tf.cast(label, tf.float32) / 100.0

	return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True, dim=CFG.img_size):
	def augment(img, dim=dim):
		img = transform(img, DIM=dim) if CFG.transform else img
		img = tf.image.random_flip_left_right(img) if CFG.hflip else img
		img = tf.image.random_flip_up_down(img) if CFG.vflip else img
		img = tf.image.random_hue(img, CFG.hue)
		img = tf.image.random_saturation(img, CFG.sat[0], CFG.sat[1])
		img = tf.image.random_contrast(img, CFG.cont[0], CFG.cont[1])
		img = tf.image.random_brightness(img, CFG.bri)
		img = dropout(img, DIM=dim, PROBABILITY=CFG.drop_prob, CT=CFG.drop_cnt, SZ=CFG.drop_size)
		img = tf.clip_by_value(img, 0, 1) if CFG.clip else img
		img = tf.reshape(img, [*dim, 3])
		return img

	def augment_with_labels(img, label):
		return augment(img), label

	return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None, batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024,
                  cache_dir="", drop_remainder=False):
	if cache_dir != "" and cache is True:
		os.makedirs(cache_dir, exist_ok=True)

	if decode_fn is None:
		decode_fn = build_decoder(labels is not None)

	if augment_fn is None:
		augment_fn = build_augmenter(labels is not None)

	AUTO = tf.data.experimental.AUTOTUNE
	slices = paths if labels is None else (paths, labels)

	ds = tf.data.Dataset.from_tensor_slices(slices)
	ds = ds.map(decode_fn, num_parallel_calls=AUTO)
	ds = ds.cache(cache_dir) if cache else ds
	ds = ds.repeat() if repeat else ds
	if shuffle:
		ds = ds.shuffle(shuffle, seed=CFG.seed)
		opt = tf.data.Options()
		opt.experimental_deterministic = False
		ds = ds.with_options(opt)
	ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
	ds = ds.batch(batch_size, drop_remainder=drop_remainder)
	ds = ds.prefetch(AUTO)
	return ds


def display_batch(batch, size=2):
	imgs, tars = batch
	plt.figure(figsize=(size * 5, 5))
	for img_idx in range(size):
		plt.subplot(1, size, img_idx + 1)
		plt.title(f'{CFG.target_col[0]}: {tars[img_idx].numpy()[0]}', fontsize=15)
		plt.imshow(imgs[img_idx, :, :, :])
		plt.xticks([])
		plt.yticks([])
	plt.tight_layout()
	plt.show()


fold = 0
fold_df = df.query('fold==@fold')[:1000]
paths = fold_df.image_path.tolist()
labels = fold_df[CFG.target_col].values
ds = build_dataset(paths, labels, cache=False, batch_size=CFG.batch_size * REPLICAS,
                   repeat=True, shuffle=True, augment=True)
ds = ds.unbatch().batch(20)
batch = next(iter(ds))
display_batch(batch, 5)


def RMSE(y_true, y_pred, denormalize=True):
	if denormalize:
		# denormalizing
		y_true = y_true * 100.0
		y_pred = y_pred * 100.0
	# rmse
	loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.subtract(y_true, y_pred))))
	return loss


RMSE.__name__ = 'rmse'

import efficientnet.tfkeras as efn
from vit_keras import vit, utils, visualize, layers

name2effnet = {
	'efficientnet_b0': efn.EfficientNetB0,
	'efficientnet_b1': efn.EfficientNetB1,
	'efficientnet_b2': efn.EfficientNetB2,
	'efficientnet_b3': efn.EfficientNetB3,
	'efficientnet_b4': efn.EfficientNetB4,
	'efficientnet_b5': efn.EfficientNetB5,
	'efficientnet_b6': efn.EfficientNetB6,
	'efficientnet_b7': efn.EfficientNetB7,
}


def build_model(model_name=CFG.model_name, DIM=CFG.img_size[0], compile_model=True, include_top=False):
	image_input = tf.keras.layers.Input(shape=(DIM, DIM, 3))
	# img_adjust_layer = tf.keras.layers.experimental.preprocessing.Resizing(DIM, DIM)
	pretrained_model = SwinTransformer(CFG.model_name, include_top=False, pretrained=True, use_tpu=True)
	base = tf.keras.Sequential([pretrained_model,
	                            tf.keras.layers.Dropout(0.2),
	                            tf.keras.layers.Dense(64, activation='Mish'),
	                            tf.keras.layers.Dropout(0.2),
	                            tf.keras.layers.Dense(1, activation='sigmoid')])
	# x = img_adjust_layer(image_input)
	x = base(image_input)
	model = tf.keras.Model(inputs=image_input, outputs=x)
	if compile_model:
		# optimizer
		opt = tf.keras.optimizers.Adam(learning_rate=0.001)
		# loss
		loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01)
		# metric
		rmse = RMSE
		model.compile(optimizer=opt,
		              loss=loss,
		              metrics=[rmse])
	return model


def get_lr_callback(batch_size=8, plot=False):
	lr_start = 0.000005
	lr_max = 0.00000125 * REPLICAS * batch_size
	lr_min = 0.000001
	lr_ramp_ep = 5
	lr_sus_ep = 0
	lr_decay = 0.8

	def lrfn(epoch):
		if epoch < lr_ramp_ep:
			lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

		elif epoch < lr_ramp_ep + lr_sus_ep:
			lr = lr_max

		elif CFG.scheduler == 'exp':
			lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min

		elif CFG.scheduler == 'cosine':
			decay_total_epochs = CFG.epochs - lr_ramp_ep - lr_sus_ep + 3
			decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
			phase = math.pi * decay_epoch_index / decay_total_epochs
			cosine_decay = 0.5 * (1 + math.cos(phase))
			lr = (lr_max - lr_min) * cosine_decay + lr_min
		return lr

	if plot:
		plt.figure(figsize=(10, 5))
		plt.plot(np.arange(CFG.epochs), [lrfn(epoch) for epoch in np.arange(CFG.epochs)], marker='o')
		plt.xlabel('epoch');
		plt.ylabel('learnig rate')
		plt.title('Learning Rate Scheduler')
		plt.show()

	lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
	return lr_callback


_ = get_lr_callback(CFG.batch_size, plot=True)
if CFG.wandb:
	def wandb_init(fold):
		config = {k: v for k, v in dict(vars(CFG)).items() if '__' not in k}
		config.update({"fold": int(fold)})
		yaml.dump(config, open(f'/kaggle/working/config fold-{fold}.yaml', 'w'), )
		config = yaml.load(open(f'/kaggle/working/config fold-{fold}.yaml', 'r'), Loader=yaml.FullLoader)
		run = wandb.init(project="petfinder-public",
		                 name=f"fold-{fold}|dim-{CFG.img_size[0]}|model-{CFG.model_name}",
		                 config=config,
		                 anonymous=anonymous,
		                 group=CFG.exp_name
		                 )
		return run


def log_wandb(fold):
	"log best result and grad-cam for error analysis"

	valid_df = df.loc[df.fold == fold].copy()
	if CFG.debug:
		valid_df = valid_df.iloc[:1000]
	valid_df['pred'] = oof_pred[fold].reshape(-1)
	valid_df['diff'] = abs(valid_df.Pawpularity - valid_df.pred)
	valid_df = valid_df[valid_df.fold == fold].reset_index(drop=True)
	vali_df = valid_df.sort_values(by='diff', ascending=False)

	noimg_cols = ['Id', 'fold', 'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group',
	              'Collage', 'Human', 'Occlusion', 'Info', 'Blur',
	              'Pawpularity', 'pred', 'diff']
	# select top and worst 10 cases
	gradcam_df = pd.concat((valid_df.head(10), valid_df.tail(10)), axis=0)
	gradcam_ds = build_dataset(gradcam_df.image_path, labels=None, cache=False, batch_size=1,
	                           repeat=False, shuffle=False, augment=False)
	data = []


oof_pred = [];
oof_tar = [];
oof_val = [];
oof_ids = [];
oof_folds = []
preds = np.zeros((test_df.shape[0], 1))

for fold in np.arange(CFG.folds):
	if fold not in CFG.selected_folds:
		continue
	if CFG.wandb:
		run = wandb_init(fold)
		WandbCallback = wandb.keras.WandbCallback(save_model=False)
	if CFG.device == 'TPU':
		if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)

	# TRAIN AND VALID DATAFRAME
	train_df = df.query("fold!=@fold")
	valid_df = df.query("fold==@fold")

	# CREATE TRAIN AND VALIDATION SUBSETS
	train_paths = train_df.image_path.values;
	train_labels = train_df[CFG.target_col].values.astype(np.float32)
	valid_paths = valid_df.image_path.values;
	valid_labels = valid_df[CFG.target_col].values.astype(np.float32)
	test_paths = test_df.image_path.values

	# SHUFFLE IMAGE AND LABELS
	index = np.arange(len(train_paths))
	np.random.shuffle(index)
	train_paths = train_paths[index]
	train_labels = train_labels[index]

	if CFG.debug:
		train_paths = train_paths[:2000];
		train_labels = train_labels[:2000]
		valid_paths = valid_paths[:1000];
		valid_labels = valid_labels[:1000]

	print('#' * 25);
	print('#### FOLD', fold)
	print('#### IMAGE_SIZE: (%i, %i) | MODEL_NAME: %s | BATCH_SIZE: %i' %
	      (CFG.img_size[0], CFG.img_size[1], CFG.model_name, CFG.batch_size * REPLICAS))
	train_images = len(train_paths)
	val_images = len(valid_paths)
	if CFG.wandb:
		wandb.log({'num_train': train_images,
		           'num_valid': val_images})
	print('#### NUM_TRAIN %i | NUM_VALID: %i' % (train_images, val_images))

	# BUILD MODEL
	K.clear_session()
	with strategy.scope():
		model = build_model(CFG.model_name, DIM=CFG.img_size[0], compile_model=True)

	# DATASET
	train_ds = build_dataset(train_paths, train_labels, cache=True, batch_size=CFG.batch_size * REPLICAS,
	                         repeat=True, shuffle=True, augment=CFG.augment)
	val_ds = build_dataset(valid_paths, valid_labels, cache=True, batch_size=CFG.batch_size * REPLICAS,
	                       repeat=False, shuffle=False, augment=False)

	print('#' * 25)
	# SAVE BEST MODEL EACH FOLD
	sv = tf.keras.callbacks.ModelCheckpoint(
		'fold-%i.h5' % fold, monitor='val_rmse', verbose=CFG.verbose, save_best_only=True,
		save_weights_only=False, mode='min', save_freq='epoch')
	callbacks = [sv, get_lr_callback(CFG.batch_size)]
	if CFG.wandb:
		callbacks.append(WandbCallback)
	# TRAIN
	print('Training...')
	history = model.fit(
		train_ds,
		epochs=CFG.epochs if not CFG.debug else 2,
		callbacks=callbacks,
		steps_per_epoch=len(train_paths) / CFG.batch_size // REPLICAS,
		validation_data=val_ds,
		# class_weight = {0:1,1:2},
		verbose=CFG.verbose
	)

	# Loading best model for inference
	print('Loading best model...')
	model.load_weights('fold-%i.h5' % fold)

	# PREDICT OOF USING TTA
	print('Predicting OOF with TTA...')
	ds_valid = build_dataset(valid_paths, labels=None, cache=False, batch_size=CFG.batch_size * REPLICAS * 2,
	                         repeat=True, shuffle=False, augment=True if CFG.tta > 1 else False)
	ct_valid = len(valid_paths);
	STEPS = CFG.tta * ct_valid / CFG.batch_size / 2 / REPLICAS
	pred = model.predict(ds_valid, steps=STEPS, verbose=CFG.verbose)[:CFG.tta * ct_valid, ]
	oof_pred.append(np.mean(pred.reshape((ct_valid, -1, CFG.tta), order='F'), axis=-1) * 100.0)

	# GET OOF TARGETS AND idS
	oof_tar.append(valid_df[CFG.target_col].values[:(1000 if CFG.debug else len(valid_df))])
	oof_folds.append(np.ones_like(oof_tar[-1], dtype='int8') * fold)
	oof_ids.append(valid_df.Id.values)

	# PREDICT TEST USING TTA
	print('Predicting Test with TTA...')
	ds_test = build_dataset(test_paths, labels=None, cache=True,
	                        batch_size=(CFG.batch_size * 2 if len(test_df) > 8 else 1) * REPLICAS,
	                        repeat=True, shuffle=False, augment=True if CFG.tta > 1 else False)
	ct_test = len(test_paths);
	STEPS = 1 if len(test_df) <= 8 else (CFG.tta * ct_test / CFG.batch_size / 2 / REPLICAS)
	pred = model.predict(ds_test, steps=STEPS, verbose=CFG.verbose)[:CFG.tta * ct_test, ]
	preds[:ct_test, :] += np.mean(pred.reshape((ct_test, -1, CFG.tta), order='F'),
	                              axis=-1) / CFG.folds * 100  # not meaningful for DIBUG = True

	# REPORT RESULTS
	y_true = oof_tar[-1];
	y_pred = oof_pred[-1]
	rmse = RMSE(y_true.astype(np.float32), y_pred, denormalize=False).numpy()
	oof_val.append(np.min(history.history['val_rmse']))
	print('#### FOLD %i OOF RMSE without TTA = %.3f, with TTA = %.3f' % (fold, oof_val[-1], rmse))

	if CFG.wandb:
		log_wandb(fold)  # log result to wandb
		wandb.run.finish()  # finish the run
		display(ipd.IFrame(run.url, width=1080, height=720))  # show wandb dashboard
# COMPUTE OVERALL OOF RMSE
oof = np.concatenate(oof_pred);
true = np.concatenate(oof_tar);
ids = np.concatenate(oof_ids);
folds = np.concatenate(oof_folds)
rmse = RMSE(true.astype(np.float32), oof, denormalize=False)
print('Overall OOF RMSE with TTA = %.3f' % rmse)
# SAVE OOF TO DISK
columns = ['Id', 'fold', 'true', 'pred']
df_oof = pd.DataFrame(np.concatenate([ids[:, None], folds[:, 0:1], true, oof], axis=1), columns=columns)
df_oof.to_csv('oof.csv', index=False)
df_oof.head()
import seaborn as sns

sns.set(style='dark')

plt.figure(figsize=(10 * 2, 6))

plt.subplot(1, 2, 1)
sns.kdeplot(x=train_df[CFG.target_col[0]], color='b', shade=True);
sns.kdeplot(x=df_oof.pred.values, color='r', shade=True);
plt.grid('ON')
plt.xlabel(CFG.target_col[0]);
plt.ylabel('freq');
plt.title('KDE')
plt.legend(['train', 'oof'])

plt.subplot(1, 2, 2)
sns.histplot(x=train_df[CFG.target_col[0]], color='b');
sns.histplot(x=df_oof.pred.values, color='r');
plt.grid('ON')
plt.xlabel(CFG.target_col[0]);
plt.ylabel('freq');
plt.title('Histogram')
plt.legend(['train', 'oof'])

plt.tight_layout()
plt.show()
pred_df = pd.DataFrame({'Id': test_df.Id,
                        'Pawpularity': preds.reshape(-1)})
sub_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/sample_submission.csv')
del sub_df['Pawpularity']
sub_df = sub_df.merge(pred_df, on='Id', how='left')
sub_df.to_csv('submission.csv', index=False)
sub_df.head(2)

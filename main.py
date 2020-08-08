import argparse
import os
from dataset import DataManager
from Captioner import Captioner
from model import *
parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--type', dest='type_of_use',default='None',
                        help='type of use train,test or caption',
                        type=str)
parser.add_argument('--checkpoint_dir', dest='Checkpoint_folder', type=str, default='checkpoints')
parser.add_argument('--cnnmodel', dest='CNN_MODEL', type=str, default='inception')
parser.add_argument('--image_folder', dest='image_folder',type=str, help='Image folder',default='None')
parser.add_argument('--caption_file', dest='caption_file',type=str, help='Caption file',default='None')
parser.add_argument('--feature_extraction',dest='features_extraction',type=str, help='Whether image features have to be extracted',default='true')
parser.add_argument('--batch_size',dest='batch_size',type=int, help='Batch size',default=128)
parser.add_argument('--buffer_size', dest='buffer_size', type=int, help='Buffer size',default=1000)
parser.add_argument('--to_caption', dest='image_path', type=str, help='Image path for Captioning',default='None')
parser.add_argument('--nb_epochs', dest='nb_epochs', type=int, help='Number of epochs',default=50)
args = parser.parse_args()
assert args.features_extraction in ['true','false','True','False']
if args.features_extraction in ['true','True']:
  f = True
elif args.features_extraction in ['false','False']:
  f = False
assert args.type_of_use in ['train','test','caption']
if args.type_of_use !='caption':
	assert args.image_folder!='None'
elif args.type_of_use == 'caption':
	assert args.image_path!='None'
assert args.caption_file!='None'
if not os.path.isdir(args.Checkpoint_folder):
	os.mkdir(args.Checkpoint_folder)
data = DataManager(batch_size = args.batch_size,
	buffer_size=args.buffer_size,
	cnn_model=args.CNN_MODEL,
	features_extraction=f,
	IMAGE_FOLDER=args.image_folder,
	captions_filename=args.caption_file)
embedding_dim = 512
lstm_units = 512
encoder = CNN_ENCODER(embedding_dim,data.img_features)
decoder = RNN_DECODER(embedding_dim,lstm_units,data.vocab_size,data.max_length)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')
ckpt = tf.train.Checkpoint(encoder = encoder,
	decoder = decoder,
	optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, args.Checkpoint_folder, max_to_keep=5)
start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])*5
  ckpt.restore(ckpt_manager.latest_checkpoint)
if start_epoch == 0 and args.type_of_use == 'train':
	print("Starting Training from scratch...")
elif start_epoch == 0 and args.type_of_use == 'test':
	assert start_epoch!=0 , "No checkpoints to use"
elif start_epoch == 0 and args.type_of_use == 'caption':
	assert start_epoch!=0 , "No checkpoints to use"
elif start_epoch>0 and args.type_of_use == 'train':
	print("Starting Training from Saved Checkpoint...")
elif start_epoch>0 and args.type_of_use == 'test':
	print("Starting Testing from Saved Checkpoint...")
elif start_epoch>0 and args.type_of_use == 'caption':
	print("Starting Captioning from Saved Checkpoint...")
captioner = Captioner(data,
	encoder,
	decoder,
	start_epoch,
	ckpt_manager,
	optimizer,
	loss_object,
  args.nb_epochs)
if args.type_of_use=='train':
	captioner.train()
if args.type_of_use=='test':
	captioner.test()
if args.type_of_use=='caption':
	captioner.caption_generator(args.image_path)
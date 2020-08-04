import nltk
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import cv2
import matplotlib.pyplot as plt
nltk.download('punkt')
import string
from sklearn.utils import shuffle
import numpy as np
class DataManager(object):
  def __init__(self,cnn_model='inception',captions_filename='Flickr8k.token.txt',
               IMAGE_FOLDER='Flicker8k_Dataset',features_extraction=False,
               batch_size=128,buffer_size=1000):
    """
    Args:
    cnn_model (str) (default:'inception'): Transfer-Learning Model for Feature-Extraction
    captions_filename (str) (default:'Flickr8k.token.txt'): Location of caption data
    IMAGE_FOLDER (str) (default:'Flicker8k_Dataset'): Location of Image_Dataset
    features_extraction (bool) (default:'False'): Whether the features from the images need to be extracted again
                                                  When running the first time set to True
                                                  If features once extracted change back to False so as 
                                                  to save time and memory
    batch_size (int) (default:128): Batch_size of the dataset
    buffer_size (int) (default:1000): Shuffle buffer size for train_dataset  
    """
    self.BATCH_SIZE = batch_size
    self.BUFFER_SIZE = buffer_size
    self.captions_filename = captions_filename
    self.image_folder = IMAGE_FOLDER
    self.image_ids= [i for i in tqdm(os.listdir(self.image_folder))]
    self.cnn = cnn_model
    self.vocab_size=3000
    self.max_length=35
    print("\n\nPreparing text data.....")
    self.prepare_text()
    self.cnn_model()
    if features_extraction:
      if self.cnn == 'inception':
        self.img_features = 2048
        self.img_shape = (299,299)
      elif self.cnn == 'vgg16':
        self.img_features = 512
        self.img_shape=(224,224)
      print("\nExtracting Image Features ....")
      self.prepare_images()
    self.build_dataset()
  def load_image(self,image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    if self.cnn=='inception':
      x = tf.keras.applications.inception_v3.preprocess_input(img)
    elif self.cnn=='vgg16':
      x = tf.keras.applications.vgg16.preprocess_input(img)
    return image_path,x
  def cnn_model(self):
    if self.cnn == 'inception':
      model = tf.keras.applications.InceptionV3(weights='imagenet')
    elif self.cnn == 'vgg16':
      model = tf.keras.applications.VGG16(weights='imagenet')
    new_input = model.input
    hidden_layer = model.layers[-2].output
    self.model_new = tf.keras.models.Model(new_input, hidden_layer)
  def prepare_images(self):
    train_captions = np.array([i[0] for i in self.train_captions])
    train_captions=sorted(set(train_captions))
    image_dataset = tf.data.Dataset.from_tensor_slices(train_captions)
    image_dataset = image_dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)# Feel free to change batch_size according to your system configuration
    for path,img in tqdm(image_dataset):
      batch_features=tf.convert_to_tensor(self.model_new.predict(img))
      batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0],batch_features.shape[1]))
      for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())
  def clean_descriptions(self,desc):
    table = str.maketrans('', '', string.punctuation)
    desc = desc.split()
    desc = [word.lower() for word in desc]
    desc = [w.translate(table) for w in desc]
    desc = [word for word in desc if len(word)>1]
    desc = [word for word in desc if word.isalpha()]
    desc_list=  ' '.join(desc)
    return desc_list
  def listing(self,text):
    all_img_path_captions=[]
    for i in text:
      caption=self.clean_descriptions(' '.join(i.split()[1:]).strip().lower())
      image_id=i.split()[0][:-2]
      if image_id[-4:]!='.jpg':
        image_id=image_id[:-2]
      if image_id in self.image_ids:
        path=self.image_folder+'/'+image_id
      all_img_path_captions.append((path,caption))
    return all_img_path_captions
  def preprocess_captions(self):
    cap=open(self.captions_filename)
    self.train_captions=self.listing(cap)
    words={}
    print("\nPreparing Vocabulary ...")
    max=0
    for batch in tqdm(self.train_captions):
      path,sentence = batch
      if len(sentence.split())>=max:
        max=len(sentence.split())
      for w in nltk.tokenize.word_tokenize(sentence.lower()):
        words[w] = words.get(w, 0) + 1.0
    assert self.vocab_size<=len(words.keys())
    self.max_length = max +2
    word_counts = sorted(list(words.items()),
                         key=lambda x: x[1],
                         reverse=True)
    #print(word_counts)
    self.words=['<start>']
    self.word2ix={}
    self.word2ix['<start>']=1
    for i in range(self.vocab_size):
       word , frequency = word_counts[i]
       if frequency>=5:
        self.words.append(word)
        self.word2ix[word] = i + 2
        max = i + 2
    self.words.append('<end>')
    self.word2ix['<end>'] = max+1
    self.ix2word={self.word2ix[i]:i for i in self.word2ix}
    #print(len(self.words))
    self.vocab_size= len(self.words)+1
  def ixing(self,caption):
    words=caption.split()
    word_idxs = []
    for w in words:
      try:
        word_idxs.append(self.word2ix[w])
      except:
        pass      
    return word_idxs
  def prepare_text(self):
    self.preprocess_captions()
    image_path=[]
    ixing=[]
    masks=[]
    for batch in tqdm(self.train_captions):
      path,caption=batch
      captions = '<start> '+caption.lower().strip()+' <end>'
      caption_ix= self.ixing(captions)
      caption_ixing = np.zeros(self.max_length,dtype=np.int64)
      caption_masks = np.zeros(self.max_length)
      caption_ix_len = len(caption_ix)
      caption_ixing[:caption_ix_len] = np.array(caption_ix)
      caption_masks[:caption_ix_len] = 1.0
      image_path.append(path)
      ixing.append(caption_ixing)
      masks.append(caption_masks)
    self.path = np.array(image_path)
    self.ixing=np.array(ixing)
    self.masks=np.array(masks)
  def map_func(self,image_name,sentence,mask):
    img_tensor = np.load(image_name.decode('utf-8')+'.npy')
    sentence = tf.cast(sentence,tf.int32)
    mask = tf.cast(mask,tf.float32)
    return img_tensor, sentence,mask
  def build_dataset(self):
    train_path,val_path,train_ixing,val_ixing,train_masks,val_masks\
    = train_test_split(self.path,
                       self.ixing,
                       self.masks,
                       test_size=0.2,
                       random_state=0)
    train_path,test_path,train_ixing,test_ixing,train_masks,test_masks\
    = train_test_split(train_path,
                       train_ixing,
                       train_masks,
                       test_size=0.25,
                       random_state=0)
    self.train_steps = len(train_path)//self.BATCH_SIZE
    self.val_steps = len(val_path)//1
    train_dataset = tf.data.Dataset.from_tensor_slices((train_path, train_ixing,train_masks))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_path, val_ixing,val_masks))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_path, test_ixing,test_masks))
    train_dataset = train_dataset.map(lambda item1, item2,item3: tf.numpy_function(
        self.map_func, [item1, item2,item3], [tf.float32, tf.int32,tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(lambda item1, item2,item3: tf.numpy_function(
        self.map_func, [item1, item2,item3], [tf.float32, tf.int32,tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(lambda item1, item2,item3: tf.numpy_function(
        self.map_func, [item1, item2,item3], [tf.float32, tf.int32,tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = train_dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    self.val_dataset = val_dataset.batch(self.BATCH_SIZE)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    self.test_dataset = test_dataset.batch(1)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
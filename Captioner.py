import tensorflow as tf
from sklearn.model_selection import train_test_split
class Captioner(DataManager):
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
    self.val_dataset = val_dataset.batch(1)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    self.test_dataset = test_dataset.batch(1)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

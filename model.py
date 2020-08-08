import tensorflow as tf
class CNN_ENCODER(tf.keras.Model):
  """
  Args:
    embedding_dim: embedding_dim
    image_features: image features value
  """
  def __init__(self,embedding_dim,image_features):
    super(CNN_ENCODER,self).__init__()
    self.fc=tf.keras.layers.Dense(embedding_dim,input_shape=(image_features,))
    self.drop=tf.keras.layers.Dropout(0.5)
    self.repeat_vector = tf.keras.layers.RepeatVector(1)
  def call(self,x,is_train=False):
    x = self.fc(x)
    x = tf.nn.relu(x)
    if is_train:
      x = self.drop(x)
    x = self.repeat_vector(x)
    return x
class RNN_DECODER(tf.keras.Model):
  """
  Args:
    embedding_dim: embedding_dim
    lstm_units: lstm units for LSTM
    vocab_size: vocabulary size
    max_length: maximum length of a caption
  """   
  def __init__(self,embedding_dim,lstm_units,vocab_size,max_length):
    super(RNN_DECODER,self).__init__()
    self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length)
    self.drop = tf.keras.layers.Dropout(0.5)
    self.lstm = tf.keras.layers.LSTM(lstm_units,return_sequences=True)
    self.fc1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(lstm_units))
    self.bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))
    self.fc2 = tf.keras.layers.Dense(vocab_size)
  def call(self,x,features,is_train = False):
    x = self.embedding(x)
    x = self.lstm(x)
    x = self.fc1(x)
    if is_train:
      x = self.drop(x)
    x = tf.concat([features,x],axis = -1 , name='concat')
    x = self.bidirectional(x)
    x = self.fc2(x)
    x = tf.nn.softmax(x)
    return x

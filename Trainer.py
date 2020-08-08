import tensorflow as tf
class Trainer(object):
  """
  Args:
    data: from DataManager
    encoder: encoder model 
    decoder: decoder model
    start_epoch: 0 if training for the first time else from restored checkpoint
    loss: loss function
    optimizer: optimizer
    cpkt_manager: for saving checkpoints
    nb_epochs default = 50: total number of epochs
  """ 
  def __init__(self,data,encoder,decoder,start_epoch,cpkt,optimizer,loss,nb_epochs = 50):
    self.data = data
    self.loss = loss
    self.optimizer = optimizer
    self.encoder = encoder
    self.decoder = decoder
    self.start_epoch = start_epoch
    self.cpkt_manager = cpkt
    self.nb_epochs = nb_epochs
  def loss_function(self,real,preds,masks):
    loss_ = self.loss(real,preds)
    loss_*=masks
    return tf.reduce_mean(loss_)
  @tf.function
  def train_step(self,img_tensor,sentence,masks):
    loss = 0.0
    dec_input = tf.expand_dims([self.data.word2ix['<start>']] * sentence.shape[0], 1)
    with tf.GradientTape() as tape:
      features = self.encoder(img_tensor)
      for i in range(1,sentence.shape[1]):
        predictions = self.decoder(dec_input,features,is_train=True)
        loss += self.loss_function(sentence[:,i],predictions,masks[:,i])
        dec_input = tf.expand_dims(sentence[:, i], 1)      
    total_loss = (loss / int(sentence.shape[1]))
    train_variables = encoder.trainable_variables+decoder.trainable_variables
    gradients = tape.gradient(loss, train_variables)
    self.optimizer.apply_gradients(zip(gradients,train_variables))
    return loss, total_loss
  def val_step(self,img_tensor,sentence,masks):
    loss = 0.0
    dec_input = tf.expand_dims([self.data.word2ix['<start>']] * sentence.shape[0], 1)
    features = self.encoder(img_tensor)
    for i in range(1,sentence.shape[1]):
      predictions = self.decoder(dec_input,features)
      loss += self.loss_function(sentence[:,i],predictions,masks[:,i])
      dec_input = tf.expand_dims(sentence[:, i], 1)      
    total_loss = (loss / int(sentence.shape[1]))
    return loss, total_loss
  def test_step(self,img_tensor,sentence):
    dec_input = tf.expand_dims([self.data.word2ix['<start>']], 0)
    result = []
    features = self.encoder(img_tensor)
    for i in range(self.data.max_length):
      predictions = self.decoder(dec_input, features,is_train=False)
      predicted_id = tf.math.argmax(predictions, 1).numpy()[0]
      result.append(self.data.ix2word[predicted_id])
      if self.data.ix2word[predicted_id] == '<end>':
        return ' '.join(result)
      dec_input = tf.expand_dims([predicted_id], 0)
    return ' '.join(result)

      
from nltk.translate.bleu_score import corpus_bleu 
from meteor import meteor_score
import cv2
import time
from tqdm import tqdm
import numpy as np
from Trainer import Trainer
import tensorflow as tf
import matplotlib.pyplot as plt
class Captioner(Trainer):
  """
  Inherited from Trainer in Trainer.py
  pls refer to that for input arguments
  """
  def train(self):
    train_plot , val_plot = [] , []
    for epoch in range(self.start_epoch,self.nb_epochs):
      start=time.time()
      total_loss , total_valloss= 0.0 , 0.0
      for (batch, (img_tensor, sentence,mask)) in tqdm(enumerate(self.data.train_dataset)):
        batch_loss, t_loss = self.train_step(img_tensor, sentence,mask)
        total_loss += t_loss
        if batch % 100 == 0:
          print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(sentence.shape[1])))
      for (batch, (img_tensor, sentence,mask)) in tqdm(enumerate(self.data.val_dataset)):
        batch_loss, t_loss = self.val_step(img_tensor, sentence,mask)
        total_valloss += t_loss
        if batch % 100 == 0:
          print ('Epoch {} Batch {} ValLoss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(sentence.shape[1])))
      train_plot.append(total_loss / self.data.train_steps)
      val_plot.append(total_valloss / self.data.val_steps)
      if (epoch+1) % 5 == 0:
        self.cpkt_manager.save()
      print ('Epoch {} Loss {:.6f} ValLoss {:6f}'.format(epoch + 1,
                                         total_loss/self.data.train_steps,total_valloss/self.data.val_steps))
      print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
      if len(val_plot)>1:
        if val_plot[-1]>val_plot[-2]:
          print("\nEarly Stopping....")
          break
    plt.plot(train_plot)
    plt.plot(val_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()
  def test(self):
    test_loss = 0.0
    meteor = []
    bleu_ref = []
    bleu_hyp = []
    for (img_name,img_tensor,sentence,mask) in tqdm(self.data.test_dataset):
      batch_loss, t_loss = self.val_step(img_tensor,sentence,mask)
      result = self.test_step(img_tensor,sentence)
      result = result.strip()[7:-5].strip()
      img_name = img_name.numpy()[0].decode('utf8')
      ref = self.data.caption_dict[img_name]
      meteor.append(meteor_score(ref,result))
      bleu_ref.append(ref)
      bleu_hyp.append(result)
      test_loss += t_loss
    print("Test Loss {}\nBleu Score:\n".format(test_loss))
    print('BLEU-1: %f' % corpus_bleu(bleu_ref, bleu_hyp, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(bleu_ref, bleu_hyp, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(bleu_ref, bleu_hyp, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(bleu_ref, bleu_hyp, weights=(0.25, 0.25, 0.25, 0.25)))
    print('Meteor: {}'.format(np.mean(meteor)))
  def caption_generator(self,image_path):
    img_path = tf.convert_to_tensor(image_path)
    img_path , img_tensor = self.data.load_image(img_path)
    batch_features = tf.convert_to_tensor(self.data.model_new.predict(tf.reshape(img_tensor,(-1,self.data.img_shape[0],self.data.img_shape[1],3))))
    print(batch_features.shape)
    result = self.test_step(tf.reshape(batch_features,(1,self.data.img_features)),img_path)
    print('ID {} , \nCaption:\nMax Search:{}'.format(image_path, result))
    plt.imshow(cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB))
    plt.title(' '.join(result.split()[:-1]))
    plt.axis("off")
    plt.show()

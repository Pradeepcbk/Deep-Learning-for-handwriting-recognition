import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
    
    model.compile(optimizer = 'adam', 
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy']
                  )
    return model
        
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=1)

model = create_model()
model.fit(x_train, y_train, epochs = 3, callbacks = [cp_callback], validation_data = (x_test,y_test), verbose=0)
model.summary()
#model.load_weights(checkpoint_path)
# =============================================================================
# #val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss)
# print(val_acc)
# =============================================================================

predictions = model.predict(x_test)
print(np.argmax(predictions[0]))
#model.summary()

'''
#fit the model with the training data
model.fit(x_train, y_train, epochs = 3)

val_loss, val_acc = model.evaluate(x_test, y_test)

#print the validation loss and accuracy found in the test data - to make sure that the model doesnt overfit
print(val_loss)
print(val_acc)

#predict the first test data
predictions = model.predict(x_test)
print(np.argmax(predictions[0]))

#check what was the input
plt.imshow(x_test[0],cmap = plt.cm.binary)
plt.show()
'''
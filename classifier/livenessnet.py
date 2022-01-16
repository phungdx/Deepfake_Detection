import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model,regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_path = ['../input/celebfaces/DatasetCELEB_train/DatasetCELEB_train/train',
             '../input/celebfaces/DatasetCELEB_train/DatasetCELEB_train/val',
            '../input/celebfaces/DatasetCELEB_test/DatasetCELEB_test']

img_height,img_width = 160,160
batch_size=32
datagen = ImageDataGenerator(
    rescale= 1./255,
    rotation_range = 30,
    horizontal_flip = True,
    fill_mode = 'nearest',)

train_generator = datagen.flow_from_directory(
    data_path[0],
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle = True,
    seed=123)

validation_generator = datagen.flow_from_directory(
    data_path[1],
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=123)


test_datagen = ImageDataGenerator(rescale= 1./255)
test_generator = test_datagen.flow_from_directory(
    data_path[2],
    target_size=(img_height, img_width),
    class_mode='binary',
    batch_size=batch_size,
    shuffle = False,
    seed=123)


train_generator.class_indices

import math
def get_labels_from_dataIterator(dataset,batch_size):
    number_of_examples = len(dataset.filenames)
    number_of_generator_calls = math.ceil(number_of_examples / (1.0 * batch_size))
    # 1.0 above is to skip integer division

    labels = []

    for i in range(0,int(number_of_generator_calls)):
        labels.extend(dataset[i][1])
        
    return np.array(labels)


import matplotlib.pyplot as plt
my_model = keras.models.load_model('/kaggle/working/classifier.h5')
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
test_labels = get_labels_from_dataIterator(test_generator,batch_size)
probs = my_model.predict(test_generator)
auc = roc_auc_score(test_labels, probs)
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plot_roc_curve(fpr, tpr)
print("AUC-ROC :",auc)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-30T14:54:13.900388Z","iopub.execute_input":"2021-12-30T14:54:13.900652Z","iopub.status.idle":"2021-12-30T14:54:13.907576Z","shell.execute_reply.started":"2021-12-30T14:54:13.900617Z","shell.execute_reply":"2021-12-30T14:54:13.905843Z"}}
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % (best_thresh))

# # calculate the g-mean for each threshold
# gmeans = np.sqrt(tpr * (1-fpr))
# # locate the index of the largest g-mean
# ix = np.argmax(gmeans)
# print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

# %% [code] {"execution":{"iopub.status.busy":"2021-12-30T14:54:13.90913Z","iopub.execute_input":"2021-12-30T14:54:13.909685Z","iopub.status.idle":"2021-12-30T14:54:13.919043Z","shell.execute_reply.started":"2021-12-30T14:54:13.909649Z","shell.execute_reply":"2021-12-30T14:54:13.918318Z"}}
# Calculate the G-mean
gmean = np.sqrt(tpr * (1 - fpr))

# Find the optimal threshold
index = np.argmax(gmean)
thresholdOpt = round(thresholds[index], ndigits = 4)
gmeanOpt = round(gmean[index], ndigits = 4)
fprOpt = round(fpr[index], ndigits = 4)
tprOpt = round(tpr[index], ndigits = 4)
print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))

# %% [code] {"execution":{"iopub.status.busy":"2021-12-30T14:54:13.921543Z","iopub.execute_input":"2021-12-30T14:54:13.922181Z","iopub.status.idle":"2021-12-30T14:54:13.929877Z","shell.execute_reply.started":"2021-12-30T14:54:13.922143Z","shell.execute_reply":"2021-12-30T14:54:13.929041Z"}}
# from sklearn.metrics import auc, roc_curve
# fpr, tpr, threshold = roc_curve(test, true, pos_label=1)

fnr = 1 - tpr
fpr_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
fnr_eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
eer = min(fpr_eer, fnr_eer)

# print("tpr = ", tpr)
# print("fpr = ", fpr)
# print("threshold = ", threshold)
print("threshold at eer = ", eer_threshold)
print("eer = ", eer)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-30T11:06:48.244983Z","iopub.execute_input":"2021-12-30T11:06:48.245289Z","iopub.status.idle":"2021-12-30T11:06:48.252659Z","shell.execute_reply.started":"2021-12-30T11:06:48.245259Z","shell.execute_reply":"2021-12-30T11:06:48.2516Z"}}
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
def mobileNet():
    inputs = keras.Input(shape=(img_height, img_width,3))
    
    #ImageNet weights
    base_model = MobileNetV2(
                            input_shape=(img_height, img_width,3),
                            include_top=False,
                            weights='imagenet')
    base_model.trainable = True
    
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    #Use the generated model
    base_model = base_model(inputs)
    #Add the fully-connected layers
    base_model = layers.GlobalAveragePooling2D()(base_model)
#     y = layers.Dense(1000, activation = 'relu')(y)
#     y = layers.Dense(1000, activation = 'relu')(y)
#     y = layers.Dropout(0.5)(y)
    outputs = layers.Dense(1, activation='sigmoid')(base_model)

    model = Model(inputs = inputs, outputs = outputs)
    
    return model



METRICS=[
         keras.metrics.BinaryAccuracy(name='accuracy'),
         keras.metrics.Precision(name='precision'),
         keras.metrics.Recall(name='recall'),
         keras.metrics.AUC(name='auc')
]

def scheduler(epoch,lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.99

lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler,
                                                    verbose=1)

new_model = mobileNet()

# new_model = keras.models.load_model('../input/models/classifier.h5')

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=4,
                                              restore_best_weights = True,
                                              verbose = 1)
checkpoint = keras.callbacks.ModelCheckpoint('/kaggle/working/classifier.h5',
                                            save_best_only=True,
#                                             lr_scheduler,
                                            monitor='val_loss',
#                                             class_weight=train_class_weights,
                                            verbose = 1)



new_model.compile(
                  optimizer = keras.optimizers.Adam(0.001),
#                   optimizer = keras.optimizers.RMSprop(0.0001),
                  loss = keras.losses.BinaryCrossentropy(),
#                   loss = keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
history = new_model.fit(train_generator,
                        epochs=35,
                        validation_data=validation_generator,
                        callbacks=[earlyStopping,checkpoint],
                        initial_epoch =  history.epoch[-1],
                        verbose=1)

# new_model.save('/kaggle/working/classifier.h5')

# %% [code] {"execution":{"iopub.status.busy":"2021-12-30T14:59:07.098794Z","iopub.execute_input":"2021-12-30T14:59:07.099822Z","iopub.status.idle":"2021-12-30T14:59:07.682089Z","shell.execute_reply.started":"2021-12-30T14:59:07.099768Z","shell.execute_reply":"2021-12-30T14:59:07.681391Z"}}
from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('/kaggle/working/loss.jpg')
plt.show()
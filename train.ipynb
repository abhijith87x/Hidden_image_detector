{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "28a21dae-0b51-4a1c-83fe-a9e9e2ecb75b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.preprocessing import image\n",
    "from tensorflow.keras import layers,models\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "import os\n",
    "import numpy as np\n",
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "95c0c901-e1c2-4d3a-9813-455c6304f260",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_height,img_width = 128,128\n",
    "batch_size = 32\n",
    "\n",
    "test_datagen  = ImageDataGenerator(rescale = 1./255)\n",
    "\n",
    "early_stop = EarlyStopping(\n",
    "    monitor='val_loss',\n",
    "    patience=3,\n",
    "    restore_best_weights=True\n",
    ")\n",
    "\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255,\n",
    "    rotation_range=30,\n",
    "    zoom_range=0.3,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    brightness_range=[0.7,1.3],\n",
    "    validation_split=0.2\n",
    ")\n",
    "\n",
    "\n",
    "val_datagen  = ImageDataGenerator(rescale = 1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b90ee6df-5718-455d-823e-18db4e8b6319",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 3126 images belonging to 2 classes.\n",
      "Found 348 images belonging to 2 classes.\n",
      "Found 348 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "train_generator = train_datagen.flow_from_directory(\n",
    "    'painting_dataset/train',\n",
    "    target_size=(img_height,img_width),\n",
    "    batch_size = batch_size,\n",
    "    class_mode = 'binary'\n",
    ")\n",
    "\n",
    "test_generator = test_datagen.flow_from_directory(\n",
    "    'painting_dataset/test',\n",
    "    target_size=(128,128),\n",
    "    batch_size=32,\n",
    "    class_mode='binary',\n",
    "    shuffle=False  \n",
    ")\n",
    "val_generator = val_datagen.flow_from_directory(\n",
    "    'painting_dataset/val',\n",
    "    target_size=(img_height, img_width),\n",
    "    batch_size=batch_size,\n",
    "    class_mode='binary'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ff5a21b7-7328-4da4-b756-759120bf1e91",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = models.Sequential([\n",
    "    layers.Conv2D(32,(3,3),activation=\"relu\",input_shape=(img_height,img_width,3)),\n",
    "    layers.MaxPooling2D((2,2)),\n",
    "    layers.Conv2D(64, (3,3), activation=\"relu\"),\n",
    "    layers.MaxPooling2D((2,2)),\n",
    "    layers.Conv2D(128,(3,3),activation=\"relu\"),\n",
    "    layers.MaxPooling2D((2,2)),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(64, activation='relu'),\n",
    "    layers.Dropout(0.5),\n",
    "    layers.Dense(1,activation=\"sigmoid\")\n",
    "])\n",
    "model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "b723b294-3151-4f93-af33-8a5ce901396b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "98/98 [==============================] - 22s 216ms/step - loss: 0.6512 - accuracy: 0.6424 - val_loss: 0.5076 - val_accuracy: 0.7414\n",
      "Epoch 2/10\n",
      "98/98 [==============================] - 22s 219ms/step - loss: 0.6172 - accuracy: 0.6660 - val_loss: 0.5624 - val_accuracy: 0.6954\n",
      "Epoch 3/10\n",
      "98/98 [==============================] - 21s 218ms/step - loss: 0.6075 - accuracy: 0.6916 - val_loss: 0.4703 - val_accuracy: 0.8132\n",
      "Epoch 4/10\n",
      "98/98 [==============================] - 22s 221ms/step - loss: 0.5806 - accuracy: 0.7095 - val_loss: 0.5059 - val_accuracy: 0.7270\n",
      "Epoch 5/10\n",
      "98/98 [==============================] - 23s 232ms/step - loss: 0.5659 - accuracy: 0.7230 - val_loss: 0.5295 - val_accuracy: 0.7328\n",
      "Epoch 6/10\n",
      "98/98 [==============================] - 23s 231ms/step - loss: 0.5585 - accuracy: 0.7300 - val_loss: 0.5264 - val_accuracy: 0.7011\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x19a9180d810>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(train_generator,validation_data = val_generator,epochs = 10,callbacks=[early_stop])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "78f780dc-0625-4b13-ae89-8337f8010a5c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "11/11 [==============================] - 1s 43ms/step - loss: 0.5259 - accuracy: 0.7529\n",
      "Test Accuracy: 0.7528735399246216\n",
      "Test Loss: 0.5258991718292236\n"
     ]
    }
   ],
   "source": [
    "test_loss, test_acc = model.evaluate(test_generator)\n",
    "\n",
    "print(\"Test Accuracy:\", test_acc)\n",
    "print(\"Test Loss:\", test_loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "47a21d0d-408d-4bcc-898f-ed360f7ea23c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Testing Image: painting_dataset/test\\no_hidden\\t3_l5f9bw.jpg\n",
      "1/1 [==============================] - 0s 24ms/step\n",
      "Prediction : Normal Painting\n"
     ]
    }
   ],
   "source": [
    "\n",
    "test_folder = \"painting_dataset/test\"\n",
    "\n",
    "# Collect all test images\n",
    "all_images = []\n",
    "\n",
    "for root, dirs, files in os.walk(test_folder):\n",
    "    for file in files:\n",
    "        if file.lower().endswith((\".png\", \".jpg\", \".jpeg\")):\n",
    "            all_images.append(os.path.join(root, file))\n",
    "\n",
    "# Randomly choose image\n",
    "img_path = random.choice(all_images)\n",
    "\n",
    "print(\"Testing Image:\", img_path)\n",
    "\n",
    "# Load and preprocess image\n",
    "img = image.load_img(img_path, target_size=(128,128))\n",
    "img_array = image.img_to_array(img)/255.0\n",
    "img_array = np.expand_dims(img_array, axis=0)\n",
    "\n",
    "# Prediction\n",
    "prediction = model.predict(img_array)\n",
    "\n",
    "if prediction[0][0] > 0.5:\n",
    "    print(\"Prediction : Normal Painting\")\n",
    "else:\n",
    "    print(\"Prediction : Hidden Painting\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6c45165d-c98c-4133-b542-65a8a3f17104",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\keras\\src\\engine\\training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
      "  saving_api.save_model(\n"
     ]
    }
   ],
   "source": [
    "model.save(\"hidden_image_detector.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9979734-dcfa-4b72-8c5f-540baa754f7a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

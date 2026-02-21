{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "340e8a82-2750-415c-8995-18eb98e334f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 48ms/step\n",
      "NO Hidden Image Detected\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import cv2\n",
    "model = tf.keras.models.load_model(\"model\\hidden_image_detector.h5\")\n",
    "IMG_SIZE = 128\n",
    "\n",
    "def predict_image(image_path):\n",
    "    img = cv2.imread(r\"D:\\random_images\\t3_l5f9bw.jpg\")\n",
    "    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))\n",
    "    img = img / 255.0\n",
    "    img = np.expand_dims(img, axis=0)\n",
    "    prediction = model.predict(img)[0][0]\n",
    "    \n",
    "    if prediction > 0.5:\n",
    "        print(\"NO Hidden Image Detected\")\n",
    "    else:\n",
    "        print(\"Hidden Image Detected\")\n",
    "\n",
    "predict_image(\"test.jpg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "526ce037-5c01-41c8-ba29-69ca618d109d",
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

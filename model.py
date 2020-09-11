#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[ ]:



from tensorflow.keras.applications.vgg19 import VGG19


# In[ ]:


model=VGG19(weights="imagenet")


# In[ ]:


model.save('vgg19.h5')


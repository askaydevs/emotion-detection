#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()


# In[5]:


emotions_rules = {'fearful': {
                                'scared': ['helpless', 'frightened'],
                                'anxious': ['overwhelmed','worried'],
                                'insecure': ['inadequate', 'inferior'],
                                'weak': ['worthless', 'insignificant'],
                                'rejected': ['excluded', 'persecuted'],
                                'threatened': ['nervous', 'exposed']
                             },
                  'angry': {'let down': ['betrayed', 'resentful'],
                            'humiliated': ['disrespected', 'ridiculed'],
                            'bitter': ['Indignant', 'Violated'],
                            'mad': ['furious', 'jealous'],
                            'aggressive': ['provoked', 'hostile'],
                            'frustrated': ['infuriated', 'annoyed'],
                            'distant': ['withdrawn', 'numb'],
                            'critical': ['sceptical', 'dismissive']
                           },
                  'disgusted': {'disapproving': ['judgemental', 'embarrassed'],
                                'disappointed': ['appalled', 'revolted'],
                                'awful': ['nauseated', 'detestable'],
                                'repelled': ['horrified', 'hesitant']
                               },
                  'sad': {'hurt': ['embarrassed', 'disappointed'],
                          'depressed': ['inferior', 'empty'],
                          'guilty': ['remorseful', 'ashamed'],
                          'despair': ['grief', 'powerless'],
                          'vulnerable': ['victimised', 'fragile'],
                          'lonely': ['isolated', 'abandoned']
                         },
                  'happy': {'playful': ['aroused', 'cheeky'],
                            'content': ['free', 'joyful'],
                            'interested': ['curious', 'inquisitive'],
                            'proud': ['successful', 'confdent'],
                            'accepted': ['respected', 'valued'],
                            'powerful': ['courageous', 'creative'],
                            'peaceful': ['thankful', 'loving'],
                            'trusting': ['intimate', 'sensitive'],
                            'optimistic': ['hopeful', 'inspired']
                           },
                  'surprised': {'startled':['shocked', 'dismayed'],
                                'confused': ['disillusioned', 'perplexed'],
                                'amazed': ['astonished', 'awe'],
                                'excited':['eager', 'energetic']
                               },
                  'bad': {'bored': ['indifferent', 'apathetic'],
                          'busy': ['pressured', 'rushed'],
                          'stressed': ['overwhelmed', 'out of controll'],
                          'tired': ['sleepy', 'unfocussed']
                         }
                 }


# In[12]:


emotions_rules['fearful'].values()


# In[19]:


all_emotions = []
for k, v in emotions_rules.items():
    emotions = [item for individual in v.values() for item in individual]
    all_emotions.extend(emotions)


# In[22]:


all_emotions = list(set(all_emotions))


# In[24]:


len(all_emotions)


# In[25]:


from transformers import pipeline
ZERO_SHOT = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# In[38]:


def classify_emotion(text, labels=None, CLASSIFIER=None):
    result = CLASSIFIER(text, labels)
    label = result['labels'][0]
    confidence = result['scores'][0]
    return label, confidence


# In[35]:


data = pd.read_csv('/Users/shubhamsingh/Downloads/out_v3_11_frustration_1.csv')


# In[36]:


data.columns


# In[ ]:


data['emotion_manual'] = data['clean_paraphrases'].progress_apply(lambda x: classify_emotion(x, labels=all_emotions, CLASSIFIER=ZERO_SHOT))


# In[ ]:





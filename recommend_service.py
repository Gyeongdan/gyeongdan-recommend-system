import warnings
warnings.filterwarnings('ignore')
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import numpy as np
from lightfm.cross_validation import random_train_test_split
import os
from scipy.sparse import csr_matrix
import pandas as pd

class ArticleData:
    def __init__(self, article_id, category):
        self.article_data = pd.DataFrame({
            'article_id' : article_id,
            'Economy and Business' : [0], 
            'Politics and Society' : [0], 
            'Technology and Culture' : [0], 
            'Sports and Leisure' : [0], 
            'Opinion and Analysis' : [0]
            })
        self.article_data.iloc[0][category] = 1

class InteractionData:
    def __init__(self, user_id, article_id, duration_time):
        self.interaction_data = pd.DataFrame({
            'user_id' : [user_id],
            'article_id' : [article_id],
            'duration_time' : [duration_time]
        })

class UserData:
    def __init__(self, user_id, age, sex):
        self.user_id = user_id
        self.age = age
        self.sex = sex
        self.user_data = pd.DataFrame({
            'user_id' :[user_id],
            'age' : [age],
            'sex' : [sex]
        })
    def get_user_data(self):
        self.user_data['age_bin'] = pd.cut(self.user_data['age'], bins=[0,31,42,57,np.inf], labels= ['<= 31', '32 - 42', '43 - 57', '>= 58'])
        self.user_data = pd.get_dummies(self.user_data.drop(columns = ['age']))
        self.user_data = self.user_data.astype(float) 
        return self.user_data
        
class RecommendService:
    def __init__(self):
        self.set_user_data('data/user_data.csv')
        self.set_article_data('data/article_data.csv')
        self.set_interaction_data('data/interaction_data.csv')
        
    def set_user_data(self, user_data_path):
        self.user_data_path = user_data_path
        self.user_data = pd.read_csv(user_data_path)
    
    def set_article_data(self, article_data_path):
        self.article_data_path = article_data_path
        self.article_data = pd.read_csv(article_data_path)
        
    def set_interaction_data(self, interaction_data_path):
        self.interaction_data_path = interaction_data_path
        self.interaction_data = pd.read_csv(interaction_data_path)
        
    def make_dataset(self):
        self.user_data['age_bin'] = pd.cut(self.user_data['age'], bins=[0,31,42,57,np.inf], labels= ['<= 31', '32 - 42', '43 - 57', '>= 58'])
        self.user_data = pd.get_dummies(self.user_data.drop(columns = ['age']))
        self.user_features_col = self.user_data.drop(columns =['user_id']).columns.values
        self.user_feat = self.user_data.drop(columns =['user_id']).to_dict(orient='records')
        
        self.item_features = self.article_data
        self.item_features_col = self.item_features.drop(columns=['article_id']).columns.values
        self.item_feat = self.item_features.drop(columns =['article_id']).to_dict(orient='records')
        
        self.dataset = Dataset()
        self.dataset.fit(users=[x for x in self.user_data['user_id']], items=[x for x in self.article_data['article_id']], item_features=self.item_features_col, user_features=self.user_features_col)
        
        self.item_features = self.dataset.build_item_features((x,y) for x,y in zip(self.item_features['article_id'], self.item_feat))
        self.user_features = self.dataset.build_user_features((x,y) for x,y in zip(self.user_data['user_id'], self.user_feat))
        
        (self.interactions, self.weights) = self.dataset.build_interactions((x, y)
                                                    for x,y in zip(self.interaction_data['user_id'], self.interaction_data['article_id']))
        
        num_users, num_items = self.dataset.interactions_shape()
        print('Num users: {}, num_items {}.'.format(num_users, num_items))
        
    def make_model(self, n_components:int = 30, loss:str = 'warp', epoch:int = 30, num_thread:int = 4):
        self.n_components = n_components
        self.loss = 'warp'
        self.epoch = epoch
        self.num_thread = num_thread
        self.model = LightFM(no_components= self.n_components, loss=self.loss, random_state = 1616)
        
    def fit_model(self): 
        self.make_dataset()
        self.make_model()
        # self.train, self.test = random_train_test_split(self.interactions,test_percentage=0.2, random_state=779)
        # self.train_w, self.test_w = random_train_test_split(self.weights, test_percentage=0.2, random_state=779)
        self.model.fit(self.interactions,  user_features= self.user_features, item_features= self.item_features, epochs=self.epoch,num_threads = self.num_thread, sample_weight = self.weights)
        
    def get_top_n_articles(self, user_id:int, article_num:int):
        item_ids = np.arange(self.interactions.shape[1])  # 예측할 아이템 ID 배열

        predictions = self.model.predict(user_id, item_ids)
        top_items = self.article_data.iloc[np.argsort(-predictions)[:article_num]]
        return top_items
    
    def similar_items(self, item_id, N=10):
        item_bias ,item_representations = self.model.get_item_representations(features=self.item_features)

        scores = item_representations.dot(item_representations[item_id, :])
        best = np.argpartition(scores, -N)[-N:]
        
        return self.article_data.iloc[best]
    
    def items_for_new_user(self, new_user_data:UserData, N:int):
        new_user = new_user_data.get_user_data()
        print(new_user)
        new_user = csr_matrix(new_user)
        scores_new_user = self.model.predict(user_ids = 0,item_ids = np.arange(self.interactions.shape[1]), user_features=new_user)
        top_items_new_user = self.article_data.iloc[np.argsort(-scores_new_user)]
        return top_items_new_user[:N]
    
    def add_user(self, user_data:UserData):
        df = pd.read_csv(self.user_data_path)
        df = pd.concat([df, user_data.user_data], ignore_index=True)
        df.to_csv(self.user_data_path, index=False)
        print("user is added")
        
    def add_interaction_data(self, interaction_data:InteractionData):
        df = pd.read_csv(self.interaction_data_path)
        df = pd.concat([df, interaction_data.interaction_data])
        df.to_csv(self.interaction_data_path, index=False)
        print("interactin is added")
        
    def add_article_data(self, article_data:ArticleData):
        df = pd.read_csv(self.article_data_path)
        df = pd.concat([df, article_data.article_data])
        df.to_csv(self.article_data_path, index=False)
        print("article is added")

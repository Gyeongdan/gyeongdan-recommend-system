import warnings
warnings.filterwarnings('ignore')
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import numpy as np
from lightfm.cross_validation import random_train_test_split
import os
from scipy.sparse import csr_matrix, coo_matrix
import pandas as pd

from datetime import datetime

class ArticleDataInfo:
    def __init__(self, article_id, category, created_at):
        self.article_data = pd.DataFrame({
            'article_id' : article_id,
            'Economy and Business' : [0], 
            'Politics and Society' : [0], 
            'Technology and Culture' : [0], 
            'Sports and Leisure' : [0], 
            'Opinion and Analysis' : [0],
            'created at' : [created_at]
            })
        self.article_data.iloc[0][category] = 1

class InteractionDataInfo:
    def __init__(self, user_id, article_id, duration_time):
        self.interaction_data = pd.DataFrame({
            'user_id' : [user_id],
            'article_id' : [article_id],
            'duration_time' : [duration_time]
        })

class RecommendService:
    def __init__(self):
        self.set_user_datas('data/user_classification.csv')
        self.set_article_datas('data/article_data.csv')
        self.set_interaction_datas('data/interaction_data.csv')
        
    def set_user_datas(self, user_data_path):
        self.user_data_path = user_data_path
        self.user_datas = pd.read_csv(user_data_path)
    
    def set_article_datas(self, article_data_path):
        self.article_data_path = article_data_path
        self.article_datas = pd.read_csv(article_data_path)
        
    def set_interaction_datas(self, interaction_data_path):
        self.interaction_data_path = interaction_data_path
        self.interaction_datas = pd.read_csv(interaction_data_path)
        
    def make_dataset(self):
        self.user_datas = pd.get_dummies(self.user_datas)
        self.user_features_col = self.user_datas.drop(columns =['classification_id']).columns.values
        self.user_feat = self.user_datas.drop(columns =['classification_id']).to_dict(orient='records')
        
        self.item_features = self.article_datas
        self.item_features_col = self.item_features.drop(columns=['article_id', 'created at']).columns.values
        self.item_feat = self.item_features.drop(columns =['article_id', 'created at']).to_dict(orient='records')
        
        self.dataset = Dataset()
        self.dataset.fit(users=[x for x in self.user_datas['classification_id']], items=[x for x in self.article_datas['article_id']], item_features=self.item_features_col, user_features=self.user_features_col)
        
        self.item_features = self.dataset.build_item_features((x,y) for x,y in zip(self.item_features['article_id'], self.item_feat))
        self.user_features = self.dataset.build_user_features((x,y) for x,y in zip(self.user_datas['classification_id'], self.user_feat))
        
        (self.interactions, self.weights) = self.dataset.build_interactions((x, y, z * self.get_time_weight(y))
                                                    for x,y, z in zip(
                                                        self.interaction_datas['classification_id'], 
                                                        self.interaction_datas['article_id'], 
                                                        self.interaction_datas['duration_time']))
    
        num_users, num_items = self.dataset.interactions_shape()
        print('Num users: {}, num_items {}.'.format(num_users, num_items))
        
    def make_model(self, n_components:int = 30, loss:str = 'warp', epoch:int = 30, num_thread:int = 4):
        self.n_components = n_components
        self.loss = loss
        self.epoch = epoch
        self.num_thread = num_thread
        self.model = LightFM(no_components= self.n_components, loss=self.loss, random_state = 1616)
        
    def fit_model(self): 
        self.make_dataset()
        self.make_model()
        self.model.fit(self.interactions, user_features= self.user_features, item_features= self.item_features, epochs=self.epoch,num_threads = self.num_thread, sample_weight = self.weights)
        
    def get_top_n_articles(self, user_id:int, article_num:int):
        item_ids = np.arange(self.interactions.shape[1])  # 예측할 아이템 ID 배열

        predictions = self.model.predict(user_id, item_ids)
        top_items = self.article_datas.iloc[np.argsort(-predictions)[:article_num]]
        return top_items
    
    def similar_items(self, item_id, N=10):
        item_bias ,item_representations = self.model.get_item_representations(features=self.item_features)

        scores = item_representations.dot(item_representations[item_id, :])
        best = np.argpartition(scores, -N)[-N:]
        
        return self.article_datas.iloc[best]
    
    # def items_for_new_user(self, new_user_data:UserDataInfo, N:int):
    #     new_user = new_user_data.get_user_data()
    #     print(new_user)
    #     new_user = csr_matrix(new_user)
    #     scores_new_user = self.model.predict(user_ids = 0,item_ids = np.arange(self.interactions.shape[1]), user_features=new_user)
    #     top_items_new_user = self.article_datas.iloc[np.argsort(-scores_new_user)]
    #     return top_items_new_user[:N]
    def get_time_weight(self, article_id):
        today = datetime.now().date()
        date_obj = datetime.strptime(self.article_datas[self.article_datas['article_id'] == article_id]['created at'].iloc[0], "%Y-%m-%d").date()
        difference = today - date_obj
        return max(1 - ((difference.days//30)/5), 0)
            
        
    def add_interaction_data(self, interaction_data:InteractionDataInfo):
        df = pd.read_csv(self.interaction_data_path)
        df = pd.concat([df, interaction_data.interaction_data])
        df.to_csv(self.interaction_data_path, index=False)
        print("interactin is added")
        
    def add_article_data(self, article_data:ArticleDataInfo):
        df = pd.read_csv(self.article_data_path)
        df = pd.concat([df, article_data.article_data])
        df.to_csv(self.article_data_path, index=False)
        print("article is added")

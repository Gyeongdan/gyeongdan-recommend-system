from user_classification import user_data_to_classification_id
from recommend_service import RecommendService
import pandas as pd
import asyncio

async def main():
    user_data_path = 'data/user_data_classified.csv'
    user_datas = pd.read_csv(user_data_path)
    recommendService = RecommendService()
    await recommendService.fit_model()
    user_id = 1
    print(user_data_to_classification_id(
            user_datas.iloc[user_id]['sex'],
            user_datas.iloc[user_id]['issue finder'],
            user_datas.iloc[user_id]['lifestyle consumer'],
            user_datas.iloc[user_id]['entertainer'],
            user_datas.iloc[user_id]['tech specialist'],
            user_datas.iloc[user_id]['professionals']
        ))
    print(recommendService.get_top_n_articles(
        user_data_to_classification_id(
            user_datas.iloc[user_id]['sex'],
            user_datas.iloc[user_id]['issue finder'],
            user_datas.iloc[user_id]['lifestyle consumer'],
            user_datas.iloc[user_id]['entertainer'],
            user_datas.iloc[user_id]['tech specialist'],
            user_datas.iloc[user_id]['professionals']
        ), 5))

if __name__ == '__main__':
    main()
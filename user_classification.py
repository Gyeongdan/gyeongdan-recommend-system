from enum import Enum
import pandas as pd

class ClassificationItem(Enum):
    ISSUE_FINDER = 'issue finder'
    LIFESTYLE_CONSUMER = 'lifestyle consumer'
    ENTERTAINER = 'entertainer'
    TECH_SPECIALIST = 'tech specialist'
    PROFESSIONALS = 'professionals'
    
def user_data_to_classification_id(sex, issue_finder, lifestyle_consumer, entertainer, tech_specialist, professionals):
    
    target_features = [[issue_finder, ClassificationItem.ISSUE_FINDER], 
                       [lifestyle_consumer, ClassificationItem.LIFESTYLE_CONSUMER],
                       [entertainer, ClassificationItem.ENTERTAINER],
                       [tech_specialist, ClassificationItem.TECH_SPECIALIST], 
                       [professionals, ClassificationItem.PROFESSIONALS]]
    target_features.sort(key=lambda x: x[0], reverse=True)
    target_features[0:3]
    data = {
        'classification_id': range(1, 21),
        'sex': ['F']*10 + ['M']*10,
        'issue finder': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        'lifestyle consumer': [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        'entertainer': [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        'tech specialist': [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1],
        'professionals': [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    filtered_df = df[
        (df[target_features[0][1].value] == 1) &
        (df[target_features[1][1].value] == 1) &
        (df[target_features[2][1].value] == 1) &
        (df['sex'] == sex)
        ]
    return (int) (filtered_df.iloc[0]['classification_id'])
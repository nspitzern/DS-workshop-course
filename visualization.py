from collections import defaultdict

import pandas as pd
import numpy as np


def get_scores_matrics(results, *args):
    data = defaultdict(list)
    
    for model_results in results:
        data['RMSE'].append(model_results['RMSE'])
        data['MAE'].append(model_results['MAE'])
        data['R^2'].append(model_results['R^2'])
    
    scores = pd.DataFrame(data=data, index=[*args])
    return scores.style.apply(_highlight)


def _highlight(data):
    if data.name == 'R^2':
        is_max = data == data.max()
        return ['background: lightgreen' if cell else '' for cell in is_max]
    else:
        is_min = data == data.min()
        return ['background: lightgreen' if cell else '' for cell in is_min]
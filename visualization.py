from collections import defaultdict

import pandas as pd
import numpy as np

from IPython.display import display_html
from itertools import chain,cycle


def display_side_by_side(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2 style="text-align: center;">{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)


def get_scores_matrics(results, *args):
    data = defaultdict(list)
    
    data['MAPE'].append(results['MAPE'])
    data['RMSE'].append(results['RMSE'])
    data['MAE'].append(results['MAE'])
    data['R^2'].append(results['R^2'])
    
    scores = pd.DataFrame(data=data, index=[*args])
    return scores.style.apply(_highlight)


def _highlight(data):
    if data.name == 'R^2':
        is_max = data == data.max()
        return ['background: lightgreen' if cell else '' for cell in is_max]
    else:
        is_min = data == data.min()
        return ['background: lightgreen' if cell else '' for cell in is_min]
    
    
def show_shap_feature_importance(shap_values, X):
    vals = np.abs(shap_values.values).mean(0)
    feature_names = X.columns

    feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                     columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],
                                  ascending=False, inplace=True)
    feature_importance.head(50)
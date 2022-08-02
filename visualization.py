import pandas as pd

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


def get_scores_matrics(results):
    scores = pd.DataFrame(data={'MSE': [results['lin_reg']['MSE'], 
                                        results['reg_tree']['MSE'], 
                                        results['forest_reg']['MSE'], 
                                        results['ada_reg']['MSE'], 
                                        results['xg_reg_model']['MSE'], 
                                        results['dnn_model']['MSE']
                                       ],
                           'RMSE': [results['lin_reg']['RMSE'], 
                                    results['reg_tree']['RMSE'], 
                                    results['forest_reg']['RMSE'], 
                                    results['ada_reg']['RMSE'], 
                                    results['xg_reg_model']['RMSE'], 
                                    results['dnn_model']['RMSE']
                                   ],
                           'MAE': [results['lin_reg']['MAE'], 
                                   results['reg_tree']['MAE'], 
                                   results['forest_reg']['MAE'], 
                                   results['ada_reg']['MAE'], 
                                   results['xg_reg_model']['MAE'], 
                                   results['dnn_model']['MAE']
                                  ],
                           'R^2': [results['lin_reg']['R^2'], 
                                   results['reg_tree']['R^2'], 
                                   results['forest_reg']['R^2'], 
                                   results['ada_reg']['R^2'], 
                                   results['xg_reg_model']['R^2'], 
                                   results['dnn_model']['R^2']
                                  ]
                            },
                     index=['Linear Regression', 'Decision Tree', 'Random Forest', 'AdaBoosting', 'XGBoost', 'NN'])
    return scores.style.apply(_highlight)


def _highlight(data):
    if data.name == 'R^2':
        is_max = data == data.max()
        return ['background: lightgreen' if cell else '' for cell in is_max]
    else:
        is_min = data == data.min()
        return ['background: lightgreen' if cell else '' for cell in is_min]
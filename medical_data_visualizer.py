import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2))> 25).astype(int)

# 3
def isnt_one(x):
    if x == 1:
        return 0
    else:
        return 1
    
df['cholesterol'] = df.apply(lambda row: isnt_one(row['cholesterol']), axis=1)
df['gluc'] = df.apply(lambda row: isnt_one(row['gluc']), axis=1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 
                     var_name='variable', 
                     value_name='value',
                     ignore_index=True)

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7
    chart = sns.catplot(x='variable', y='total', hue='value', data=df_cat, kind='bar', col='cardio')

    # 8
    fig = chart.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr().round(1)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10, 8)) 

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={"shrink": .8})

    # 16
    fig.savefig('heatmap.png')
    return fig

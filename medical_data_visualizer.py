import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('../projeto03_boilerplate-medical-data-visualizer/medical_examination.csv')

# 2
value_overweight = []

for weight, height in zip(df['weight'], df['height']):
	BMI = weight / ((height / 100)**2)
	if BMI > 25:
		x = 1
	else:
		x = 0
	value_overweight.append(x)

df["overweight"] = value_overweight

# 3 
df['cholesterol'] = df['cholesterol'].apply(lambda value: 0 if value == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda value: 0 if value == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], var_name='variable', value_name='value')


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7
    catplot = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar',
        height=5,
        aspect=1.2
    )


    # 8
    fig = catplot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
	
    height_low = df['height'].quantile(0.025)
    height_high = df['height'].quantile(0.975)
    weight_low = df['weight'].quantile(0.025)
    weight_high = df['weight'].quantile(0.975)
 
    df_heat = df[
        (df['height'] >= height_low) & 
        (df['height'] <= height_high) & 
        (df['weight'] >= weight_low) & 
        (df['weight'] <= weight_high)
    ]
    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 8))

    # Plotar a matriz de correlação com um heatmap usando seaborn
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


    # 14
    fig, ax = plt.gcf(), plt.gca()

    # 15



    # 16
    fig.savefig('heatmap.png')
    return fig

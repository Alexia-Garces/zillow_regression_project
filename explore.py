import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score

def tax_distribution_viz(df):
    los_angeles_tax_dist = df[df.county_name == "Los Angeles"].tax_rate
    orange_tax_dist = df[df.county_name == "Orange"].tax_rate
    ventura_tax_dist = df[df.county_name == "Ventura"].tax_rate

    plt.figure(figsize=(16,14))

    plt.subplot(3,1,1)
    sns.distplot(los_angeles_tax_dist, bins=10, kde=True, rug=True, color='#5302a3')
    plt.xlim(0, 2)
    plt.ylim(0, 8)
    plt.title("Los Angeles County Tax Distribution", fontsize=20)

    plt.subplot(3,1,2)
    sns.distplot(orange_tax_dist, bins=10, kde=True, rug=True, color='#8b0aa5')
    plt.xlim(0, 2)
    plt.ylim(0, 8)
    plt.title("Orange County Tax Distribution", fontsize=20)

    plt.subplot(3,1,3)
    sns.distplot(ventura_tax_dist, bins=10, kde=True, rug=True, color='#b83289')
    plt.xlim(0, 2)
    plt.ylim(0, 8)
    plt.title("Ventura County Tax Distribution", fontsize=20)

    plt.tight_layout()

    plt.show()

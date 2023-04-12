import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os
import pandas as pd


def plot_curve_scores(save_path, x_list, y_list, class_list):
    # 每个类的准确率变化曲线
    x = []
    y = []
    class_all = []
    markers = []
    for y_class_elem, class_name in zip(y_list, class_list):
        x.extend([x_elem for x_elem in x_list])
        y.extend([y_elem * 100 for y_elem in y_class_elem])
        class_all.extend([class_name] * len(x_list))
        markers.append("o")
    df = pd.DataFrame({
        "#Samples": x,
        "Type of Score": class_all,
        "Score": y,
    })
    plt.figure(num=f'PLOT_CURVE_SCORES', figsize=(11.7, 9.27))
    sns.set_style('darkgrid')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax = sns.lineplot(x="#Samples", y="Score", hue="Type of Score", style="Type of Score", data=df, dashes=False, palette="bright", markers=markers)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.grid(linestyle='--')
    plt.savefig(os.path.join(save_path, f'plot_curve_scores.png'))
    plt.clf()

import torch
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd

# Bond index plotting colors.
# ['Brazil', 'Chile',    'Colombia', 'Peru',     'Panama',   'Mexico']
# ['#009C3B','#0032A0',  '#FCD116',  '#D91023',  '#964B00',  '#FFB52E']
# ['green',  'blue',     'yellow',   'red','     brown',     'orange']


def prediction_plotting(
    prediction: torch.tensor,
    target: torch.tensor,
    num_plot: int,
    grid: tuple = (2, 3),
    specific_plots: list = None,
    random_walk: int = None,
    fig_size: tuple = (12, 8),
):
    """
    Plotting predictions against true targets.
    :param prediction: torch.tensor, same dimensions as target.
    :param target: torch.tensor, same dimensions as prediction.
    :param num_plot: int, number of plots.
    :param random_walk: int, number of lags for the RW.
    :param grid: tuple, define grid of plots.
    :param specific_plots: list, specific plots for plotting.
    :fig_size: tuple, size of figure
    """
    # Squeeze input if more than two dim.
    def squeezing_input(inpt):
        if len(inpt.shape) > 2:
            inpt = inpt.squeeze()
        if len(inpt.shape) == 1:
            inpt = inpt.unsqueeze(1)
        return inpt

    prediction = squeezing_input(prediction)
    target = squeezing_input(target)

    if num_plot > (grid[0] * grid[1]):
        raise Exception("Number of plot does not fit into grid")
    elif num_plot < (grid[0] * grid[1]):
        print("Consider making number of plots fit with grid")

    # Sample
    if specific_plots:
        if len(specific_plots) != num_plot:
            raise Exception("Specific plotting list has to have the length of num_plot")
        else:
            plotting_sample = specific_plots
    elif num_plot == prediction.shape[1]:
        plotting_sample = range(prediction.shape[1])
    else:
        plotting_sample = random.sample(range(1, prediction.shape[1]), num_plot)
        plotting_sample.sort()

    plt.subplots(figsize=fig_size)
    num = 0
    legends = ["Target", "Prediction"]
    if random_walk != None:
        legends.append("Random Walk")
    for _, idx_value in enumerate(plotting_sample):
        num += 1
        plt.subplot(grid[0], grid[1], num)
        plt.plot(target[:, idx_value])
        plt.plot(prediction[:, idx_value])
        if random_walk != None:
            A = [target[0, idx_value].tolist()] * random_walk
            B = target[0:-random_walk, idx_value].tolist()
            plt.plot(A + B)
        if num == 1:
            plt.legend(legends, loc=(0.5, 1), ncol=2)
        plt.title(
            "Timeseries #:" + str(idx_value), loc="left", fontsize=10, fontweight=0
        )


def plotly_spaghetti_plot(
    exp_dict: dict,
    nodes_from: list,
    nodes_to: list,
    name_list: list,
    yaxis_title: str = None,
    plot_title: str = None,
    legend_title: str = None,
    line_color_list: list = None,
    diag_only: bool = True,
    smoothing_factor: int = 200,
    save_file_as: str = None,
):
    fig = go.Figure()
    exp_dict_keys = list(exp_dict.keys())

    number_of_series = exp_dict[exp_dict_keys[0]].shape[1]

    x_range = np.arange(exp_dict[exp_dict_keys[0]][:, 0, 0].shape[0]) - smoothing_factor

    # Add horizontal line
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=np.ones(x_range[-1]) / number_of_series,
            showlegend=False,
            line=dict(color="rgb(101, 101, 101)", dash="dash", width=2),
        )
    )

    for i in nodes_from:
        for j in nodes_to:

            if diag_only:
                if i != j:
                    continue

            # CONSTRUCT MEAN
            concate = exp_dict[exp_dict_keys[0]][:, i, j].unsqueeze(dim=1)
            for exp in range(1, len(exp_dict)):
                add_tensor = exp_dict[exp_dict_keys[exp]][:, i, j].unsqueeze(dim=1)
                concate = torch.cat((concate, add_tensor), dim=1)
            # SMOOTHING
            y_smoothed = (
                pd.Series(torch.mean(concate, dim=1).numpy())
                .rolling(window=smoothing_factor)
                .mean()
                .iloc[smoothing_factor:]
                .values
            )
            y_smoothed_std = (
                pd.Series(torch.std(concate, dim=1).numpy())
                .rolling(window=smoothing_factor)
                .mean()
                .iloc[smoothing_factor:]
                .values
            )

            fig.add_trace(
                go.Scatter(
                    name=name_list[i],
                    x=x_range,
                    y=y_smoothed,
                    line_color=None if line_color_list == None else line_color_list[i],
                    legendgroup=f"series{i}",
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Upper Bound",
                    x=x_range,
                    y=y_smoothed + y_smoothed_std,
                    mode="lines",
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    legendgroup=f"series{i}",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Lower Bound",
                    x=x_range,
                    y=y_smoothed - y_smoothed_std,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.1)",
                    fill="tonexty",
                    legendgroup=f"series{i}",
                    showlegend=False,
                )
            )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title=legend_title,
        title=plot_title,
        xaxis_title="Time",
        yaxis_title=yaxis_title,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", showgrid=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black", showgrid=False)

    if save_file_as != None:
        fig.write_image(save_file_as + ".png", width=1200, height=400, scale=2)

    fig.show()


def multiple_heatmap_plot(
    exp_dict: dict,
    vmin: float = 0.0,
    vmax: float = 1.0,
    col_map: str = "magma_r",
    save_file_as: str = None,
):

    num_of_exp = len(exp_dict)
    exp_dict_keys = list(exp_dict.keys())

    fig, axn = plt.subplots(2, num_of_exp, sharex=True, sharey=True, figsize=(20, 8))
    cbar_ax = fig.add_axes([0.91, 0.07, 0.02, 0.86])

    for i, ax in enumerate(axn.flat):
        if i < num_of_exp:
            ax.set_title("Experiment " + str(i + 1))
            data = np.round(torch.mean(exp_dict[exp_dict_keys[i]], dim=0).numpy(), 2)
        else:
            data = np.round(
                torch.std(exp_dict[exp_dict_keys[i - num_of_exp]], dim=0).numpy(), 2
            )

        sns.heatmap(
            data,
            ax=ax,
            cbar=i == 0,
            vmin=vmin,
            vmax=vmax,
            cmap=col_map,
            annot=True,
            linewidths=3,
            cbar_ax=None if i else cbar_ax,
        )

        if i == 0:
            ax.set_ylabel("Mean", fontsize=18)
        if i == num_of_exp:
            ax.set_ylabel("Standard deviation", fontsize=18)

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    if save_file_as != None:
        plt.savefig("save_file_as.png", bbox_inches="tight")
    plt.show()


def heatmap(
    data,
    row_labels,
    col_labels,
    label_size,
    ax=None,
    cbar_kw={},
    cbarlabel="",
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=label_size)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=label_size)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            # text = ax.text(j, i, data[i, j],
            #          ha="center", va="center", color="w")

            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            # texts.append(text)

    return texts


def plot_confusion_matrix(tn, fp, fn, tp, labels, color_min=1, color_max=500):
    # cm : confusion matrix list(list)
    # labels : name of the data list(str)
    # title : title for the heatmap
    cm = [[fn, tp], [tn, fp]]
    y_reverse_labels = [labels[-i] for i in range(1, len(labels) + 1)]
    data = go.Heatmap(z=cm, y=y_reverse_labels, x=labels, colorscale="Blues")
    annotations = []
    order = [tn, fp, fn, tp]
    annot = [[fn, tn], [tp, fp]]
    for i, row in enumerate(annot):
        for j, value in enumerate(row):

            fonti = "white" if value > 200 else "black"
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": fonti},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False,
                }
            )
    # Disse conditions skal bare fjernes

    layout = {
        "yaxis": {"title": "Real value"},
        "xaxis": {"title": "Predicted value"},
        "annotations": annotations,
    }

    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(showscale=False)
    fig.data[0].update(zmin=color_min, zmax=color_max)
    fig.update_xaxes(side="top")

    fig.update_layout(
        autosize=False,
        width=400,
        height=400,
        margin=dict(l=0, r=0, b=0, t=0, pad=4),
        font=dict(family="Serif", size=40),
    )

    return fig

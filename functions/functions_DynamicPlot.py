import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from PIL import Image
import os
import math
import numpy as np
import sys

sys.path.append("/Users/danieleraimondi/bitcoin_datascience/functions")
from fetch_data import fetch_crypto_data

def load_data():
    df = fetch_crypto_data("btc")
    df = df[["time", "PriceUSD"]].dropna()
    return df


def create_bitcoin_price_animation(df, sampling_interval=30):
    """
    Create an animated plot of Bitcoin price over time.

    This function takes a DataFrame containing Bitcoin price data and creates
    an interactive, animated plot using Plotly. The animation shows the Bitcoin
    price trend over time, with options to play/pause the animation and switch
    between linear and logarithmic scales.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing at least two columns:
                           'time' (timestamp) and 'PriceUSD' (float)
    sampling_interval (int): Number of data points to skip between each frame
                             of the animation. Higher values result in faster
                             animation but lower resolution. Default is 20.

    Returns:
    plotly.graph_objects.Figure: An interactive Plotly figure object

    Note:
    The function assumes that the input DataFrame has 'time' and 'PriceUSD' columns.
    It will remove any rows with NaN values in the 'PriceUSD' column.
    """

    # Convert 'time' column to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Sort the DataFrame by date
    df = df.sort_values("time")

    # Remove rows with NaN in PriceUSD
    df = df.dropna(subset=["PriceUSD"])

    # Create the plot
    fig = make_subplots(rows=1, cols=1)

    # Add the initial trace
    fig.add_trace(
        go.Scatter(
            x=[df["time"].iloc[0]],
            y=[df["PriceUSD"].iloc[0]],
            mode="lines",
            name="Bitcoin Price",
            line=dict(color="#F7931A", width=2),  # Bitcoin orange color
        )
    )

    # Configure the layout
    fig.update_layout(
        title={
            "text": "Bitcoin Price Over Time",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=24, color="#333333"),
        },
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showticklabels=True,
            showgrid=True,
            gridcolor="lightgray",
            title="",
        ),
        yaxis=dict(
            showticklabels=True,
            showgrid=True,
            gridcolor="lightgray",
            title="",
            type="linear",  # Initial arithmetic scale
        ),
        height=800,  # Increase plot height
    )

    # Create frames for animation with data sampling
    frames = [
        go.Frame(data=[go.Scatter(x=df["time"][:k], y=df["PriceUSD"][:k])])
        for k in range(2, len(df), sampling_interval)
    ]

    # Add frames to the figure object
    fig.frames = frames

    # Add animation controls and scale change button
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,  # Moved to the left
                y=-0.05,  # Moved below the plot
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            ),
            dict(
                type="buttons",
                direction="left",
                x=0.5,
                y=1.05,
                xanchor="center",
                yanchor="top",
                pad=dict(t=0, r=10),
                showactive=True,
                buttons=[
                    dict(
                        label="Linear",
                        method="relayout",
                        args=[{"yaxis.type": "linear"}],
                    ),
                    dict(
                        label="Log",
                        method="relayout",
                        args=[{"yaxis.type": "log"}],
                    ),
                ],
            ),
        ]
    )

    return fig


def create_bitcoin_price_gif(df, scale_type, sampling_interval=30, duration=75):
    """
    Create an animated GIF of Bitcoin price over time.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing 'time' and 'PriceUSD' columns
    scale_type (str): 'log' for logarithmic scale, 'linear' for linear scale
    sampling_interval (int): Number of data points to skip between each frame. Default is 30.
    duration (int): Duration of each frame in milliseconds. Default is 75.

    Returns:
    None. Saves the generated GIF in the output directory.
    """
    # Convert 'time' column to datetime and sort the dataframe
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    # Remove rows with missing price data
    df = df.dropna(subset=["PriceUSD"])

    # Create a subplot figure
    fig = make_subplots(rows=1, cols=1)

    # Add initial trace to the figure
    fig.add_trace(
        go.Scatter(
            x=[df["time"].iloc[0]],
            y=[df["PriceUSD"].iloc[0]],
            mode="lines",
            name="Bitcoin Price",
            line=dict(color="#F7931A", width=2),  # Bitcoin orange color
        )
    )

    # Calculate minimum and maximum price for y-axis scaling
    min_price = df["PriceUSD"].min()
    max_price = df["PriceUSD"].max()

    if scale_type == "log":
        # For logarithmic scale
        min_log = math.floor(math.log10(min_price))
        max_log = math.ceil(math.log10(max_price))
        # Create tick values and labels for logarithmic scale
        tick_values = [10**i for i in range(min_log, max_log + 1)]
        tick_text = [f"${10**i:,}" for i in range(min_log, max_log + 1)]
        # Configure y-axis dictionary for logarithmic scale
        yaxis_dict = dict(
            type="log",
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            showticklabels=True,
            showgrid=True,
            gridcolor="lightgray",
            title="Price (USD)",
        )
    else:
        # For linear scale
        range_price = max_price - min_price
        # Calculate step size aiming for about 5 major ticks
        step = 10 ** math.floor(math.log10(range_price / 5))
        # Create tick values and labels for linear scale
        tick_values = [
            i * step
            for i in range(
                math.floor(min_price / step), math.ceil(max_price / step) + 1
            )
        ]
        tick_text = [f"${val:,.0f}" for val in tick_values]
        # Configure y-axis dictionary for linear scale
        yaxis_dict = dict(
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            showticklabels=True,
            showgrid=True,
            gridcolor="lightgray",
            title="Price (USD)",
            type=scale_type,
        )

    # Update the layout of the figure
    fig.update_layout(
        title={
            "text": f"Bitcoin Price Over Time ({scale_type.capitalize()} Scale)",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=24, color="#333333"),
        },
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showticklabels=True,
            showgrid=True,
            gridcolor="lightgray",
            title="",
        ),
        yaxis=yaxis_dict,
        height=800,
        width=1200,
    )

    # Create frames for animation
    frames = [
        go.Frame(data=[go.Scatter(x=df["time"][:k], y=df["PriceUSD"][:k])])
        for k in range(2, len(df), sampling_interval)
    ]

    # Add frames to the figure
    fig.frames = frames

    # Create temporary directory for storing frame images
    if not os.path.exists("temp_images"):
        os.makedirs("temp_images")

    # Generate and save individual frames as images
    for i, frame in enumerate(fig.frames):
        fig.update(data=frame.data)
        pio.write_image(fig, f"temp_images/frame_{i:03d}.png")

    # Load saved images
    images = []
    for file_name in sorted(os.listdir("temp_images")):
        if file_name.endswith(".png"):
            file_path = os.path.join("temp_images", file_name)
            images.append(Image.open(file_path))

    # Remove the first image (it's empty due to how frames are created)
    images = images[1:]

    # Save the GIF
    images[0].save(
        f"../output/Dynamic_Plot/Dynamic_BTC_Plot_{scale_type}.gif",
        save_all=True,
        append_images=images[1:] + [images[-1]] * 10,  # Repeat last frame 10 times
        optimize=False,
        duration=duration,  # Each frame displays for the specified duration
        loop=1,  # GIF loops once (plays 1 time)
    )

    # Clean up temporary images
    for file_name in os.listdir("temp_images"):
        os.remove(os.path.join("temp_images", file_name))
    os.rmdir("temp_images")

    print(f"GIF created successfully: Dynamic_BTC_Plot_{scale_type}.gif")


def create_fixed_bitcoin_chart(df):
    """
    Create an interactive Bitcoin price chart using Plotly.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing 'time' and 'PriceUSD' columns

    Returns:
    plotly.graph_objects.Figure: An interactive Plotly figure object
    """
    # Convert 'time' column to datetime and handle potential errors
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Sort the DataFrame by date and remove rows with NaN in PriceUSD
    df = df.sort_values("time").dropna(subset=["PriceUSD", "time"])

    # Create a numeric column for the x-axis
    df["x_numeric"] = (df["time"] - df["time"].min()).dt.total_seconds()

    # Create the figure
    fig = make_subplots(rows=1, cols=1)

    # Add the complete trace
    fig.add_trace(
        go.Scatter(
            x=df["x_numeric"],
            y=df["PriceUSD"],
            mode="lines",
            name="Bitcoin Price",
            line=dict(color="#F7931A", width=2),
        )
    )

    # Function to format the x-axis
    def format_date(x):
        try:
            return pd.to_datetime(x, unit="s", origin=df["time"].min()).strftime("%Y")
        except ValueError:
            return ""  # Return empty string for invalid dates

    # Configure the layout
    fig.update_layout(
        title={
            "text": "Bitcoin Price Over Time",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=24, color="#333333"),
        },
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showticklabels=True,
            showgrid=True,
            gridcolor="lightgray",
            title="",
            type="linear",
            tickmode="array",
            tickvals=np.linspace(df["x_numeric"].min(), df["x_numeric"].max(), 10),
            ticktext=[
                format_date(x)
                for x in np.linspace(df["x_numeric"].min(), df["x_numeric"].max(), 10)
            ],
        ),
        yaxis=dict(
            showticklabels=True,
            showgrid=True,
            gridcolor="lightgray",
            title="",
            type="linear",
        ),
        height=800,
    )

    # Add animation controls and scale change buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=-0.05,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            ),
            dict(
                type="buttons",
                direction="left",
                x=0.3,
                y=1.05,
                xanchor="center",
                yanchor="top",
                pad=dict(t=0, r=10),
                showactive=True,
                buttons=[
                    dict(
                        label="Y Linear",
                        method="relayout",
                        args=[{"yaxis.type": "linear"}],
                    ),
                    dict(
                        label="Y Log", method="relayout", args=[{"yaxis.type": "log"}]
                    ),
                ],
            ),
            dict(
                type="buttons",
                direction="left",
                x=0.7,
                y=1.05,
                xanchor="center",
                yanchor="top",
                pad=dict(t=0, r=10),
                showactive=True,
                buttons=[
                    dict(
                        label="X Linear",
                        method="relayout",
                        args=[{"xaxis.type": "linear"}],
                    ),
                    dict(
                        label="X Log",
                        method="relayout",
                        args=[
                            {
                                "xaxis.type": "log",
                                "xaxis.tickmode": "array",
                                "xaxis.tickvals": np.logspace(
                                    np.log10(df["x_numeric"].min()),
                                    np.log10(df["x_numeric"].max()),
                                    10,
                                ),
                                "xaxis.ticktext": [
                                    format_date(x)
                                    for x in np.logspace(
                                        np.log10(df["x_numeric"].min()),
                                        np.log10(df["x_numeric"].max()),
                                        10,
                                    )
                                ],
                            }
                        ],
                    ),
                ],
            ),
        ]
    )

    # Create frames for animation
    frames = [
        go.Frame(data=[go.Scatter(x=df["x_numeric"][:k], y=df["PriceUSD"][:k])])
        for k in range(2, len(df), len(df) // 100)  # Use 100 frames for animation
    ]

    # Add frames to the figure
    fig.frames = frames

    return fig

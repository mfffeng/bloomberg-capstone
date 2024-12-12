# %%
import os
import pickle

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

# %%
# Avoid shenenigans with backend settings from keras 3
os.environ["KERAS_BACKEND"] = "tensorflow"

# %%
st.set_page_config(
    layout="wide",
    page_title="Stock Price Anomaly Detection",
    page_icon="📈",
)
st.title("Anomaly Detection in Stock Prices using Autoencoders")

# %%
WIN_SIZE = 7


# %%
def create_windows(data, win_size):
    X = []
    for i in range(len(data) - win_size):
        X.append(data[i : i + win_size])
    return np.array(X), np.array(X)[:, :, 0]


# %%

# Note: Streamlit's cache will fail randomly for unknown reasons, so I switched to
# storing the returned values in a pickle file and loading them back in the next run.

# @st.cache_data
# def process_and_predict():
#     sp500 = pickle.load(open("./sp500.pickle", "rb"))
#     with open("history_vol.pickle", "rb") as f:
#         history = pickle.load(f)
#     ae = load_model("autoencoder_vol.keras")
#     training_windows = []
#     training_labels = []
#     testing_windows = {}
#     testing_labels = {}
#     scalars = {}
#     for symbol, stock in sp500.items():
#         stock.set_index(stock.columns[0], inplace=True)
#         stock.index = pd.to_datetime(stock.index)
#         stock["log_return"] = np.log(
#             stock["adjusted_close"] / stock["adjusted_close"].shift(1)
#         )
#         stock["log_return"] = stock["log_return"].fillna(0)

#         stock["rolling_std"] = stock["log_return"].rolling(WIN_SIZE).std().fillna(0)

#         stock = stock[["log_return", "volume", "rolling_std"]]
#         stock = stock.astype(np.float32)

#         train = stock.loc[stock[stock.index < "2021-01-01"].index]
#         test = stock.loc[stock[stock.index >= "2021-01-01"].index]

#         scaler = StandardScaler()

#         if not train.empty:
#             train_scaled = scaler.fit_transform(train)
#             test_scaled = scaler.transform(test)
#             train_windows, train_labels = create_windows(train_scaled, WIN_SIZE)
#             test_windows, test_labels = create_windows(test_scaled, WIN_SIZE)
#             training_windows.append(train_windows)
#             training_labels.append(train_labels)
#             testing_windows[symbol] = test_windows
#             testing_labels[symbol] = test_labels
#         else:
#             # If the stock doesn't have enough data to cover one window size, skip it
#             if len(test) < WIN_SIZE:
#                 continue
#             test_scaled = scaler.fit_transform(test)
#             test_windows, test_labels = create_windows(test_scaled, WIN_SIZE)
#             testing_windows[symbol] = test_windows
#             testing_labels[symbol] = test_labels
#             scaler.fit(test)

#         scalars[symbol] = scaler

#     training_windows = np.concatenate(training_windows)
#     training_labels = np.concatenate(training_labels)
#     # print(
#     #     training_windows.shape,
#     #     training_labels.shape,
#     #     len(testing_windows),
#     #     len(testing_labels),
#     # )
#     predictions = ae.predict(training_windows[: int(0.8 * len(training_windows))])
#     # predictions = ae.predict(training_windows)
#     return (
#         sp500,
#         training_windows,
#         training_labels,
#         testing_windows,
#         testing_labels,
#         history,
#         ae,
#         predictions,
#     )


# # %%
# (
#     sp500,
#     training_windows,
#     training_labels,
#     testing_windows,
#     testing_labels,
#     history,
#     ae,
#     train_predictions,
# ) = process_and_predict()

# with open("processed_data.pickle", "wb") as f:
#     pickle.dump(
#         (
#             sp500,
#             training_windows,
#             training_labels,
#             testing_windows,
#             testing_labels,
#             history,
#             ae,
#             train_predictions,
#         ),
#         f,
#     )

with open("processed_data.pickle", "rb") as f:
    (
        sp500,
        training_windows,
        training_labels,
        testing_windows,
        testing_labels,
        history,
        ae,
        train_predictions,
    ) = pickle.load(f)

st.subheader("S&P 500 Raw Data")
selected_stock = st.selectbox(
    "Select a stock", sorted(list(sp500.keys())), key="raw_data"
)
st.write(sp500[selected_stock].iloc[:, :6])
train_mae_loss = np.mean(
    np.abs(train_predictions - training_labels[: int(0.8 * len(training_windows))]),
    axis=1,
)


# %%
st.subheader("Model Training History")
fig = go.Figure()

fig.add_trace(go.Scatter(y=history["loss"], mode="lines", name="Training Loss"))
fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines", name="Validation Loss"))

fig.update_layout(
    title="Training and Validation Loss",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    legend=dict(x=0, y=1),
)

st.plotly_chart(fig)

# %%
THRESHOLD = np.percentile(train_mae_loss, 90)
st.markdown(f"Global Threshold: {THRESHOLD}")


# %%
# @st.cache_data
# def predict_val():
#     val_predictions = ae.predict(training_windows[int(0.8 * len(training_windows)) :])
#     val_mae_loss = np.mean(
#         np.abs(val_predictions - training_labels[int(0.8 * len(training_windows)) :]),
#         axis=1,
#     )
#     return val_mae_loss


# val_mae_loss = predict_val()

# with open("val_mae_loss.pickle", "wb") as f:
#     pickle.dump(val_mae_loss, f)

with open("val_mae_loss.pickle", "rb") as f:
    val_mae_loss = pickle.load(f)

# %%
st.subheader("Training & Validation Set MAE Loss Distribution")
# fig = px.histogram(
#     train_mae_loss[train_mae_loss < THRESHOLD],
#     nbins=100,
#     name="Training MAE Loss",
# )
fig = go.Figure(
    data=[
        go.Histogram(
            x=train_mae_loss[train_mae_loss < THRESHOLD],
            nbinsx=50,
            name="Training MAE Loss",
            histnorm="probability",
            # histnorm="density",
            opacity=0.75,
        ),
        go.Histogram(
            x=val_mae_loss[val_mae_loss < THRESHOLD],
            nbinsx=50,
            name="Validation MAE Loss",
            histnorm="probability",
            # histnorm="density",
            opacity=0.6,
        ),
    ]
)
fig.update_layout(barmode="overlay")
st.plotly_chart(fig)

# %%
st.subheader("Anomaly Detection Threshold")
st.markdown("This applies to all subsequent visualizations.")
threshold_mod = st.selectbox(
    "Choose a threshold mode:",
    ["Global (from the training set)", "Local (one for every stock)"],
)

# %%
st.subheader("Test Set MAE Statistics")
selected_stock_for_anomaly = st.selectbox(
    "Select a stock", sorted(list(testing_windows.keys())), key="test_stats"
)


test_predictions = ae.predict(testing_windows[selected_stock_for_anomaly])
test_mae_loss = np.mean(
    np.abs(test_predictions - testing_labels[selected_stock_for_anomaly]),
    axis=1,
)
fig = px.line(
    x=sp500[selected_stock_for_anomaly]
    .loc[sp500[selected_stock_for_anomaly].index >= "2021-01-01"][:-WIN_SIZE]
    .index,
    y=test_mae_loss,
    title=f"Anomalies Detected on {selected_stock_for_anomaly}",
    labels={"x": "Date", "y": "MAE Loss"},
)
fig.add_hline(
    y=THRESHOLD
    if threshold_mod == "Global (from the training set)"
    else np.percentile(test_mae_loss, 90),
    line_dash="dash",
    annotation_text="Threshold",
    annotation_position="top left",
    line_color="red",
)
st.plotly_chart(fig)


# %%
st.subheader("Anomaly Detection")
selected_stock_for_anomaly = st.selectbox(
    "Select a stock", sorted(list(testing_windows.keys())), key="anomaly_detection"
)
test_predictions = ae.predict(testing_windows[selected_stock_for_anomaly])
test_mae_loss = np.mean(
    np.abs(test_predictions - testing_labels[selected_stock_for_anomaly]),
    axis=1,
)

fig_loss = go.Figure(
    data=[
        go.Histogram(
            x=train_mae_loss[
                train_mae_loss
                < (
                    THRESHOLD
                    if threshold_mod == "Global (from the training set)"
                    else np.percentile(train_mae_loss, 90)
                )
            ],
            nbinsx=50,
            name="Training MAE Loss",
            histnorm="probability",
            # histnorm="density",
            opacity=0.75,
        ),
        go.Histogram(
            x=test_mae_loss[
                test_mae_loss
                < (
                    THRESHOLD
                    if threshold_mod == "Global (from the training set)"
                    else np.percentile(test_mae_loss, 90)
                )
            ],
            nbinsx=50,
            name=f"Test MAE Loss for {selected_stock_for_anomaly}",
            histnorm="probability",
            # histnorm="density",
            opacity=0.6,
        ),
    ]
)

fig_loss.update_layout(barmode="overlay")
st.plotly_chart(fig_loss)


plot_data = sp500[selected_stock_for_anomaly].loc[
    sp500[selected_stock_for_anomaly].index >= "2021-01-01"
]
anomalies = plot_data[:-WIN_SIZE][
    test_mae_loss > THRESHOLD
    if threshold_mod == "Global (from the training set)"
    else test_mae_loss > np.percentile(test_mae_loss, 90)
]

fig_price = px.line(
    plot_data,
    x=plot_data.index,
    y="adjusted_close",
    title=f"Stock Price and Anomalies for {selected_stock_for_anomaly}",
)
fig_price.add_trace(
    go.Scatter(
        x=anomalies.index,
        y=plot_data.loc[anomalies.index, "adjusted_close"],
        mode="markers",
        marker=dict(color="red", size=3),
        name="Anomaly",
    )
)

fig_returns = px.line(
    plot_data,
    x=plot_data.index,
    y="log_return",
    title=f"Stock Log Returns and Anomalies for {selected_stock_for_anomaly}",
)
fig_returns.add_trace(
    go.Scatter(
        x=anomalies.index,
        y=plot_data.loc[anomalies.index, "log_return"],
        mode="markers",
        marker=dict(color="red", size=3),
        name="Anomaly",
    )
)

st.plotly_chart(fig_price)
st.plotly_chart(fig_returns)

st.markdown("List of Anomalies")
anomalies_data = anomalies.loc[:, ["adjusted_close", "log_return"]]
anomalies_data["return%"] = (np.exp(anomalies_data["log_return"]) - 1) * 100
anomalies_data["period_mae"] = test_mae_loss[
    test_mae_loss
    > (
        THRESHOLD
        if threshold_mod == "Global (from the training set)"
        else np.percentile(test_mae_loss, 90)
    )
]
anomalies_data["threshold"] = (
    THRESHOLD
    if threshold_mod == "Global (from the training set)"
    else np.percentile(test_mae_loss, 90)
)
st.write(anomalies_data)

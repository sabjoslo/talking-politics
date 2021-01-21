#!/usr/bin/python3

"""Creates the figures for Study 1 and Study 3a.
"""

import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# # Study 1
# df = pd.read_csv("../experiments/conditional-variation/study-1/figure.csv")
# df.rename(columns=dict(rmetric="logodds", ratings_mean="mean", ratings_std="se"
#          ), inplace=True)

# Study 3a
df = pd.read_csv("../experiments/conditional-variation/study-3a/figure.csv")

df["mean"] += 1

df_d = df.loc[df.logodds < 0]
df_r = df.loc[df.logodds > 0]

df["size_"] = 100

fig = px.scatter(df, x="logodds", y="mean", color="logodds",
                 color_continuous_scale=["blue","white","red"],
                 color_continuous_midpoint=0, size="size_", text="word",
                 error_y="se")

x_d = np.append([np.min(df_d["logodds"])-1,0],df_d["logodds"])
mod = sm.OLS(df_d["mean"], sm.add_constant(df_d["logodds"]))
res = mod.fit()
fig.add_trace(
    go.Scatter(
        x=x_d,
        y=res.predict(sm.add_constant(x_d)),
        mode="lines",
        line=dict(
            color="blue"
        )
    )
)

x_r = np.append([np.max(df_r["logodds"])+1,0],df_r["logodds"])
mod = sm.OLS(df_r["mean"], sm.add_constant(df_r["logodds"]))
res = mod.fit()
fig.add_trace(
    go.Scatter(
        x=x_r,
        y=res.predict(sm.add_constant(x_r)),
        mode="lines",
        line=dict(
            color="red"
        )
    )
)

fig.data[0].error_y.thickness = 1
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(
    font=dict(size=14),
    plot_bgcolor="rgba(200,200,200,.1)",
    shapes=[ dict(type="line", xref="x", x0=0, x1=0, yref="paper", y0=0, y1=1,
                  line=dict(dash="dash", width=.5)
                 ),
             dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=3.5,
                  y1=3.5, line=dict(dash="dash", width=.5)
                 )
           ],
    showlegend=False
)
fig.update_traces(marker=dict(line=dict(color="black")),
                  textposition="middle right")
fig.update_xaxes(title="$logodds_R$", title_font=dict(size=30))
# For Study 3a
fig.update_xaxes(range=(-1.5,1.5))
fig.update_yaxes(title="Participant rating", title_font=dict(size=30))

fig.show()

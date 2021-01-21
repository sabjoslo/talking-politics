#!/usr/bin/python3

import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_csv("../experiments/conditional-variation/study-2/figure.csv")

df["label"] = [ "{}/{}".format(rw,dw) for rw,dw in zip(df.rw,df.dw) ]
df.sort_values(by=["acc"], inplace=True)
df[4:81].label = ""
df["label"] = df["label"].replace("father/mother", "")
df["label"] = df["label"].replace("support/oppose", "")
df["label"] = df["label"].replace("provides/eliminates", "")
df.rename(columns=dict(acc="Accuracy"), inplace=True)

fig = px.scatter(df, x="logodds_r", y="logodds_d", color="Accuracy",
                 color_continuous_scale=["rgba(0,0,0,0)","rgba(0,0,0,.75)"],
                 size="Accuracy", text="label")
fig.add_annotation(x=df.loc[df.rw == "father"].logodds_r.values[0],
                   y=df.loc[df.dw == "mother"].logodds_d.values[0],
                   text="father/mother", axref="x", ayref="y", ax=.85, ay=0,
                   showarrow=True, arrowhead=1)
fig.add_annotation(x=df.loc[df.rw == "support"].logodds_r.values[0],
                   y=df.loc[df.dw == "oppose"].logodds_d.values[0],
                   text="support/oppose", axref="x", ayref="y", ax=-.1, ay=-.75,
                   showarrow=True, arrowhead=1)
fig.add_annotation(x=df.loc[df.rw == "provides"].logodds_r.values[0],
                   y=df.loc[df.dw == "eliminates"].logodds_d.values[0],
                   text="provides/eliminates", axref="x", ayref="y", ax=.75,
                   ay=-1.25, showarrow=True, arrowhead=1)
fig.update_layout(plot_bgcolor="white",
                  title=dict(text=r"$logodds_R(w_R)$", font=dict(size=30), x=.5,
                             xanchor="center"), width=1000, height=1000)
fig.update_traces(marker=dict(line=dict(color="black")),
                  textposition="top center")
fig.update_xaxes(anchor="free", position=1, ticklabelposition="inside",
                 showgrid=False, showline=True, linecolor="black",
                 range=(-.1,4.2), title_text="")
fig.update_yaxes(title_text=r"$logodds_R(w_D)$", title_font=dict(size=30),
                 showgrid=False, showline=True, linecolor="black",
                 range=(-3.1,.1))
fig.add_hline(y=-3.1)
fig.add_vline(x=4.2)
fig.show()

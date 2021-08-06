#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv(os.path.expanduser("fig1b.csv"))
df = df.groupby(["spi","dpe"]).agg(["mean","count"]).reset_index()
fig = go.Figure(
    go.Scatter(
       x=df.spi,
       y=df.dpe,
       mode="markers",
       marker=dict(
           color=df["avb","mean"],
           size=5*df["avb","count"],
           colorscale=["darkgreen","lightgrey","yellow"],
           cmid=0,
           colorbar=dict(
               title="Magnitude of AVB",
               title_font_size=15
           )
       ),
   )    
)
fig.update_xaxes(title="Strength of party identity (SPI)", title_font_size=25,
                 gridcolor="lightgrey", gridwidth=.5, 
                 zerolinecolor="lightgrey", zerolinewidth=.5,
                 linecolor="black", mirror=True
                )
fig.update_yaxes(title="Degree of political engagement (DPE)",
                 title_font_size=25, gridcolor="lightgrey", gridwidth=.5,
                 zerolinecolor="lightgrey", zerolinewidth=.5,
                 linecolor="black", mirror=True
                )
fig.update_layout(plot_bgcolor="white", width=1000, height=750)
fig.show()
fig.write_image("figure1b.png")

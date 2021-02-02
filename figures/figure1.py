#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np





sns.histplot(x=dat, stat="density", hue=dat>0, palette="seismic", legend=False)
sns.kdeplot(x=dat, color="black", linewidth=2)
plt.xlabel(r"$logodds_R$")
plt.show()
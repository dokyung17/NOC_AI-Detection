import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

for feature in features:

    if feature not in df.columns:
        continue

    plt.figure(figsize=(8,4))

    real = df[df["class"]=="Real"][feature]
    fake = df[df["class"]=="Fake"][feature]

    x_real = np.ones(len(real))*0
    x_fake = np.ones(len(fake))*1

    plt.scatter(
        x_real,
        real,
        alpha=0.7,
        label="Real"
    )

    plt.scatter(
        x_fake,
        fake,
        alpha=0.7,
        label="Fake"
    )

    plt.xticks([0,1],["Real","Fake"])
    plt.ylabel(feature)
    plt.title(feature)

    plt.legend()

    plt.tight_layout()

    plt.savefig(f"plots/{feature}_scatter.png", dpi=300)
    plt.close()

print("Done.")
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("beam_results.csv")

# Graph 1: Beam size vs BLEU
plt.figure()
plt.plot(df["beam_size"], df["bleu"], marker="o")
plt.xlabel("Beam size")
plt.ylabel("BLEU score")
plt.title("Effect of Beam Size on BLEU")
plt.xticks(df["beam_size"])
plt.grid(True)
plt.savefig("beam_size_vs_bleu.png", dpi=300, bbox_inches="tight")

# Graph 2: Beam size vs generation time
plt.figure()
plt.plot(df["beam_size"], df["time_seconds"], marker="o")
plt.xlabel("Beam size")
plt.ylabel("Time in seconds")
plt.title("Effect of Beam Size on Generation Time")
plt.xticks(df["beam_size"])
plt.grid(True)
plt.savefig("beam_size_vs_time.png", dpi=300, bbox_inches="tight")

print("Saved beam_size_vs_bleu.png and beam_size_vs_time.png")

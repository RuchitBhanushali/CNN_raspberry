import json, pandas as pd, matplotlib.pyplot as plt, pathlib

root = pathlib.Path(__file__).parent
bench = json.load(open(root/"bench_results.json"))
offline = {d["model"]: d["accuracy"]*100 for d in json.load(open(root/"acc_results.json"))["results"]}
live    = {d["model"]: d["accuracy"]*100 for d in json.load(open(root/"live_acc.json"))}

df = pd.DataFrame(bench).set_index("model")
df["offline_acc"] = df.index.map(offline)
df["live_acc"]    = df.index.map(live)
df.to_csv(root/"metrics_table.csv")          # handy for LaTeX table

# -------- FPS + Latency bar chart --------
ax = df[["fps", "latency_ms"]].plot.bar(rot=0)
ax.set_ylabel("FPS  (higher is better) /  Latency ms  (lower is better)")
plt.tight_layout()
plt.savefig(root/"fig_performance.png", dpi=300)

# -------- Accuracy bar chart --------
ax2 = df[["offline_acc", "live_acc"]].plot.bar(rot=0, color=["tab:purple","tab:green"])
ax2.set_ylabel("Accuracy (%)")
plt.ylim(0, 100); plt.tight_layout()
plt.savefig(root/"fig_accuracy.png", dpi=300)
print("✓ wrote fig_performance.png, fig_accuracy.png and metrics_table.csv")

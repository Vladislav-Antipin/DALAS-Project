import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap

# --- Config (edit these numbers/text) ---
stages = [
    dict(name="Discovery + Preclinical", dur=4.5, cost=400, pass_rate=None, note="Very high attrition"),
    dict(name="Phase I", dur=1.0, cost=60, pass_rate=0.65, note="Safety, PK/PD"),
    dict(name="Phase II", dur=1.5, cost=200, pass_rate=0.35, note="PoC + dose finding\n(highest failure)"),
    dict(name="Phase III", dur=3.0, cost=600, pass_rate=0.72, note="Confirmatory efficacy\n+ safety at scale"),
    dict(name="FDA Review (NDA/BLA)", dur=1.0, cost=75, pass_rate=0.90, note="Label + manufacturing\n+ benefit–risk"),
]

title = "Drug development timeline (US): typical duration, cost, and stage-wise success rates (illustrative)"
subtitle = (
    "Durations/costs vary widely by indication and modality; costs are rough out-of-pocket phase costs for a "
    "typical small-molecule program. Overall cost per approved drug is much higher once failures and cost of "
    "capital are included."
)

# --- Build cumulative timeline ---
starts = [0.0]
for s in stages[:-1]:
    starts.append(starts[-1] + s["dur"])
total_years = starts[-1] + stages[-1]["dur"]

# --- Figure ---
fig, ax = plt.subplots(figsize=(14, 5.6))
ax.set_xlim(0, total_years)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("Time (years)")
ax.set_title(title, fontsize=14, pad=18)

# Subtitle: reduce wrap width or comment out if you want a super-clean slide
ax.text(
    0, 1.02,
    "\n".join(textwrap.wrap(subtitle, 120)),
    transform=ax.transAxes,
    fontsize=10,
    va="bottom"
)

bar_y = 0.60
bar_h = 0.16

# Draw stage bars + labels on separate rows
for s, x0 in zip(stages, starts):
    rect = patches.Rectangle((x0, bar_y - bar_h/2), s["dur"], bar_h, linewidth=1)
    ax.add_patch(rect)

    x_mid = x0 + s["dur"]/2

    # Stage name (top row)
    ax.text(x_mid, bar_y + 0.17, s["name"], ha="center", va="bottom", fontsize=11)

    # Duration (under name)
    ax.text(x_mid, bar_y + 0.135, f"{s['dur']:.1f}y", ha="center", va="bottom", fontsize=9)

    # Cost (inside bar)
    ax.text(x_mid, bar_y, f"≈ ${s['cost']}M", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # Stage success (below bar)
    pr = "—" if s["pass_rate"] is None else f"{int(round(s['pass_rate']*100))}%"
    ax.text(x_mid, bar_y - 0.17, f"Stage success: {pr}", ha="center", va="top", fontsize=10)

    # Notes (lowest row)
    note = "\n".join(textwrap.wrap(s["note"], 22))
    ax.text(x_mid, bar_y - 0.25, note, ha="center", va="top", fontsize=9)

# Vertical separators at phase boundaries
for x in starts[1:]:
    ax.axvline(x, linewidth=0.8, linestyle="--", alpha=0.6)

# X ticks at boundaries + end
xticks = starts + [total_years]
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x:.1f}" for x in xticks])

# Clean spines
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

# “To market” arrow
ax.annotate("", xy=(total_years, 0.88), xytext=(0, 0.88),
            arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(total_years, 0.90, "Market", ha="right", va="bottom", fontsize=11)

plt.tight_layout()

# Save
out_path = "clinical_trials_timeline_v2.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
print("Saved:", out_path)
plt.show()
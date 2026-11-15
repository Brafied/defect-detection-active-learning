"""Regenerate both report figures from the pilot's saved predictions and history."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

folder = Path(__file__).resolve().parent
history = pd.read_csv(folder / "epoch_metrics.csv")
pred = pd.read_csv(folder / "test_predictions.csv")
y, score = pred.label, pred.probability
assert len(history) == 10 and len(pred) == 440 and pred.image.is_unique
matrix = confusion_matrix(y, (score >= 0.5).astype(int), labels=[0, 1])
assert list(matrix.ravel()) == list(history.iloc[-1][["tn","fp","fn","tp"]].astype(int))
ap, auc = average_precision_score(y, score), roc_auc_score(y, score)
assert abs(ap - history.iloc[-1].average_precision) < 1e-10
assert abs(auc - history.iloc[-1].roc_auc) < 1e-10

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.right":False,
                     "axes.spines.top":False,"axes.titleweight":"bold","axes.labelcolor":"#243746",
                     "axes.edgecolor":"#829099","xtick.color":"#394953","ytick.color":"#394953"})
blue, green, gray = "#1c5282", "#16705f", "#647681"
def save(fig, name):
    fig.savefig(folder/name, dpi=220, facecolor="white")
    plt.close(fig)
fig, (loss, cm) = plt.subplots(1, 2, figsize=(8.1,2.62), layout="constrained")
loss.plot(history.epoch,history.train_loss,marker="o",ms=4,color=blue,lw=2)
loss.set(title="A   Training loss", xlabel="Epoch", ylabel="Binary cross-entropy", xticks=[1,3,5,7,10])
loss.grid(axis="y",alpha=.2)
tn,fp,fn,tp = matrix.ravel()
labels = [["Correct normal","False alarm"],["Missed defect","Detected defect"]]
cm.set_xlim(0,2); cm.set_ylim(2,0); cm.set_xticks([.5,1.5],["Normal","Defective"]); cm.set_yticks([.5,1.5],["Normal","Defective"]); cm.set_xlabel("Predicted label"); cm.set_ylabel("True label"); cm.set_title("B   Predictions at threshold 0.5")
for s in cm.spines.values(): s.set_visible(False)
cm.tick_params(length=0)
for i in range(2):
    for j in range(2):
        cm.add_patch(plt.Rectangle((j,i),1,1,fc="#e3efe9" if i==j else "#f6e9e2",ec="white",lw=2))
        cm.text(j+.5,i+.44,str(matrix[i,j]),ha="center",va="center",weight="bold",size=16)
        cm.text(j+.5,i+.70,labels[i][j],ha="center",va="center",size=9)
save(fig,"figure_1_training_and_decisions.png")
fig, (pr,roc) = plt.subplots(1,2,figsize=(8.1,2.62),layout="constrained")
prec,rec,_=precision_recall_curve(y,score)
pr.step(rec[::-1],prec[::-1],where="pre",lw=1.8,color=green,label=f"Average precision {ap:.3f}")
pr.axhline(y.mean(),color=gray,ls="--",label=f"Defect prevalence {y.mean():.3f}")
pr.set(title="A   Precision–recall",xlabel="Recall",ylabel="Precision")
fpr,tpr,_=roc_curve(y,score)
roc.plot(fpr,tpr,color=blue,lw=1.8,label=f"ROC AUC {auc:.3f}")
roc.plot([0,1],[0,1],"--",color=gray,label="Chance")
roc.set(title="B   Receiver operating characteristic",xlabel="False positive rate",ylabel="True positive rate")
for ax in (pr,roc):
    ax.set(xlim=(0,1.01),ylim=(0,1.03)); ax.legend(frameon=True,framealpha=.9,fontsize=8,loc="upper right" if ax is pr else "lower right"); ax.grid(alpha=.13)
save(fig,"figure_2_precision_recall_roc.png")

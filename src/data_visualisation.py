
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score, roc_curve


def plot_confusion_matrix(test_labels, test_predicted):
    cm = confusion_matrix(test_labels, test_predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["0", "1"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion matrix")
    plt.show()
    
    print(classification_report(test_labels, test_predicted,
                                target_names=["0", "1"]))
    

def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(True)
    plt.show()

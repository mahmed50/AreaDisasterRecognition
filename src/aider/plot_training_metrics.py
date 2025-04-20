import matplotlib.pyplot as plt

def plot_accuracy_loss_from_vm_metrics(metrics, title='Training Metrics'):
    accuracy = [epoch['train_acc'] for epoch in metrics]
    loss = [epoch['train_loss'] for epoch in metrics]
    epochs = [epoch['epoch'] for epoch in metrics]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(epochs, accuracy, label='Accuracy', marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(epochs, loss, label='Loss', marker='x', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(title)
    plt.tight_layout()
    plt.show()

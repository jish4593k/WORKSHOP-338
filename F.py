import sys
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + 2 + 0.1 * np.random.randn(100)


X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()

sns.set(style="whitegrid")
sns.scatterplot(x=X.squeeze(), y=y, label="Original Data")
sns.lineplot(x=X.squeeze(), y=model(X_tensor).detach().numpy().squeeze(), color="red", label="Model Prediction")
plt.title("Simple Linear Regression")
plt.legend()

# GUI class for displaying the plot
class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.canvas = MplCanvas(self.central_widget, width=5, height=4, dpi=100)
        self.layout.addWidget(self.canvas)

        self.plot_data()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.show()
    sys.exit(app.exec_())

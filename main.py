import sys
import warnings
# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Suppress SIP deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class SklearnGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Scikit-Learn GUI')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, 1)
        
        # Right panel for display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, 2)
        
        # Create tabs for right panel
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)
        
        # Create different tabs
        self.data_tab = QWidget()
        self.model_tab = QWidget()
        self.results_tab = QWidget()
        self.visualization_tab = QWidget()
        
        self.tabs.addTab(self.data_tab, "Data View")
        self.tabs.addTab(self.model_tab, "Model Training")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.visualization_tab, "Visualization")
        
        # Setup data tab
        self.setup_data_tab()
        
        # Setup model tab
        self.setup_model_tab()
        
        # Setup results tab
        self.setup_results_tab()
        
        # Setup visualization tab
        self.setup_visualization_tab()
        
        # Data loading section
        load_group = QGroupBox("Data Loading")
        load_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load CSV File")
        self.load_button.clicked.connect(self.load_csv)
        load_layout.addWidget(self.load_button)
        
        self.sample_data_button = QPushButton("Load Sample Data")
        self.sample_data_button.clicked.connect(self.load_sample_data)
        load_layout.addWidget(self.sample_data_button)
        
        self.data_info_label = QLabel("No data loaded")
        load_layout.addWidget(self.data_info_label)
        
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)
        
        # Data preprocessing section
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QVBoxLayout()
        
        self.target_var_combo = QComboBox()
        preprocess_layout.addWidget(QLabel("Target Variable:"))
        preprocess_layout.addWidget(self.target_var_combo)
        
        self.preprocess_button = QPushButton("Preprocess Data")
        self.preprocess_button.clicked.connect(self.preprocess_data)
        preprocess_layout.addWidget(self.preprocess_button)
        
        preprocess_group.setLayout(preprocess_layout)
        left_layout.addWidget(preprocess_group)
        
        # Model selection section
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        # Model type selection
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "Classification",
            "Regression",
            "Clustering"
        ])
        self.model_type_combo.currentTextChanged.connect(self.update_model_options)
        model_type_layout.addWidget(self.model_type_combo)
        model_layout.addLayout(model_type_layout)
        
        # Model algorithm selection
        self.model_algo_combo = QComboBox()
        model_layout.addWidget(QLabel("Algorithm:"))
        model_layout.addWidget(self.model_algo_combo)
        
        # Test size slider
        test_size_layout = QHBoxLayout()
        test_size_layout.addWidget(QLabel("Test Size:"))
        self.test_size_slider = QSlider(Qt.Horizontal)
        self.test_size_slider.setRange(10, 50)
        self.test_size_slider.setValue(20)
        self.test_size_label = QLabel("20%")
        self.test_size_slider.valueChanged.connect(
            lambda v: self.test_size_label.setText(f"{v}%")
        )
        test_size_layout.addWidget(self.test_size_slider)
        test_size_layout.addWidget(self.test_size_label)
        model_layout.addLayout(test_size_layout)
        
        # Train button
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        model_layout.addWidget(self.train_button)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # Evaluation section
        eval_group = QGroupBox("Evaluation")
        eval_layout = QVBoxLayout()
        
        self.evaluate_button = QPushButton("Evaluate Model")
        self.evaluate_button.clicked.connect(self.evaluate_model)
        eval_layout.addWidget(self.evaluate_button)
        
        self.cross_val_button = QPushButton("Cross Validation")
        self.cross_val_button.clicked.connect(self.cross_validation)
        eval_layout.addWidget(self.cross_val_button)
        
        self.predict_button = QPushButton("Make Predictions")
        self.predict_button.clicked.connect(self.make_predictions)
        eval_layout.addWidget(self.predict_button)
        
        eval_group.setLayout(eval_layout)
        left_layout.addWidget(eval_group)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Initialize model options
        self.update_model_options()
        
    def setup_data_tab(self):
        layout = QVBoxLayout(self.data_tab)
        
        # Create table view for data
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)
        
        # Data statistics
        self.data_stats_text = QTextEdit()
        self.data_stats_text.setReadOnly(True)
        layout.addWidget(QLabel("Data Statistics:"))
        layout.addWidget(self.data_stats_text)
        
    def setup_model_tab(self):
        layout = QVBoxLayout(self.model_tab)
        
        # Model parameters
        self.params_text = QTextEdit()
        self.params_text.setReadOnly(True)
        layout.addWidget(QLabel("Model Parameters:"))
        layout.addWidget(self.params_text)
        
        # Training log
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.training_log)
        
    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        # Confusion matrix (for classification)
        self.confusion_matrix_text = QTextEdit()
        self.confusion_matrix_text.setReadOnly(True)
        layout.addWidget(QLabel("Confusion Matrix:"))
        layout.addWidget(self.confusion_matrix_text)
        
    def setup_visualization_tab(self):
        layout = QVBoxLayout(self.visualization_tab)
        
        # Matplotlib figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
    def update_model_options(self):
        model_type = self.model_type_combo.currentText()
        self.model_algo_combo.clear()
        
        if model_type == "Classification":
            self.model_algo_combo.addItems([
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Support Vector Machine",
                "Gradient Boosting"
            ])
        elif model_type == "Regression":
            self.model_algo_combo.addItems([
                "Linear Regression",
                "Random Forest Regressor",
                "Gradient Boosting Regressor"
            ])
        elif model_type == "Clustering":
            self.model_algo_combo.addItems([
                "K-Means",
                "PCA (Dimensionality Reduction)"
            ])
    
    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.display_data()
                self.update_data_info()
                self.populate_target_vars()
                self.status_bar.showMessage(f"Loaded {len(self.data)} rows from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def load_sample_data(self):
        # Load iris dataset as sample
        from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
        
        sample_data, ok = QInputDialog.getItem(
            self, "Select Sample Dataset", "Dataset:",
            ["Iris (Classification)", "Diabetes (Regression)", "Breast Cancer (Classification)"], 0, False
        )
        
        if ok and sample_data:
            if "Iris" in sample_data:
                data = load_iris()
                self.data = pd.DataFrame(data.data, columns=data.feature_names)
                self.data['target'] = data.target
            elif "Diabetes" in sample_data:
                data = load_diabetes()
                self.data = pd.DataFrame(data.data, columns=data.feature_names)
                self.data['target'] = data.target
            elif "Breast Cancer" in sample_data:
                data = load_breast_cancer()
                self.data = pd.DataFrame(data.data, columns=data.feature_names)
                self.data['target'] = data.target
            
            self.display_data()
            self.update_data_info()
            self.populate_target_vars()
            self.status_bar.showMessage(f"Loaded {sample_data} dataset with {len(self.data)} rows")
    
    def display_data(self):
        if self.data is not None:
            self.data_table.setRowCount(self.data.shape[0])
            self.data_table.setColumnCount(self.data.shape[1])
            self.data_table.setHorizontalHeaderLabels(self.data.columns)
            
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    item = QTableWidgetItem(str(self.data.iat[i, j]))
                    self.data_table.setItem(i, j, item)
    
    def update_data_info(self):
        if self.data is not None:
            buffer = StringIO()
            self.data.info(buf=buffer)
            info_str = buffer.getvalue()
            
            stats = self.data.describe().to_string()
            
            self.data_stats_text.setText(f"Dataset Shape: {self.data.shape}\n\n"
                                       f"Info:\n{info_str}\n\n"
                                       f"Statistics:\n{stats}")
    
    def populate_target_vars(self):
        if self.data is not None:
            self.target_var_combo.clear()
            self.target_var_combo.addItems(self.data.columns.tolist())
            # Default to last column as target
            self.target_var_combo.setCurrentIndex(len(self.data.columns) - 1)
    
    def preprocess_data(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        try:
            # Separate features and target
            target_col = self.target_var_combo.currentText()
            X = self.data.drop(columns=[target_col])
            y = self.data[target_col]
            
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            
            # Encode categorical variables if any
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
            
            # Encode target if categorical
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Split data
            test_size = self.test_size_slider.value() / 100.0
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            
            self.status_bar.showMessage(
                f"Preprocessing complete. Train: {len(self.X_train)}, Test: {len(self.X_test)}"
            )
            
            # Update training log
            self.training_log.append("Data preprocessing completed successfully.")
            self.training_log.append(f"Training set: {len(self.X_train)} samples")
            self.training_log.append(f"Test set: {len(self.X_test)} samples")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preprocessing failed: {str(e)}")
    
    def train_model(self):
        if self.X_train is None:
            QMessageBox.warning(self, "Warning", "Please preprocess data first!")
            return
        
        model_type = self.model_type_combo.currentText()
        algorithm = self.model_algo_combo.currentText()
        
        try:
            if model_type == "Classification":
                if algorithm == "Logistic Regression":
                    self.model = LogisticRegression(max_iter=1000)
                elif algorithm == "Decision Tree":
                    self.model = DecisionTreeClassifier(random_state=42)
                elif algorithm == "Random Forest":
                    self.model = RandomForestClassifier(random_state=42)
                elif algorithm == "Support Vector Machine":
                    self.model = SVC(probability=True)
                elif algorithm == "Gradient Boosting":
                    self.model = GradientBoostingClassifier(random_state=42)
            
            elif model_type == "Regression":
                if algorithm == "Linear Regression":
                    self.model = LinearRegression()
                elif algorithm == "Random Forest Regressor":
                    from sklearn.ensemble import RandomForestRegressor
                    self.model = RandomForestRegressor(random_state=42)
                elif algorithm == "Gradient Boosting Regressor":
                    from sklearn.ensemble import GradientBoostingRegressor
                    self.model = GradientBoostingRegressor(random_state=42)
            
            elif model_type == "Clustering":
                if algorithm == "K-Means":
                    self.model = KMeans(n_clusters=3, random_state=42)
                elif algorithm == "PCA":
                    self.model = PCA(n_components=2)
            
            # Train model
            self.model.fit(self.X_train, self.y_train)
            
            # Display model parameters
            params = str(self.model.get_params())
            self.params_text.setText(f"Model: {algorithm}\n\nParameters:\n{params}")
            
            # Update training log
            self.training_log.append(f"Model trained: {algorithm}")
            self.training_log.append(f"Training completed successfully.")
            
            self.status_bar.showMessage(f"Model trained: {algorithm}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed: {str(e)}")
    
    def evaluate_model(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model trained!")
            return
        
        model_type = self.model_type_combo.currentText()
        algorithm = self.model_algo_combo.currentText()
        
        try:
            results = ""
            
            if model_type == "Classification":
                y_pred = self.model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                report = classification_report(self.y_test, y_pred)
                cm = confusion_matrix(self.y_test, y_pred)
                
                results = f"Model: {algorithm}\n"
                results += f"Accuracy: {accuracy:.4f}\n\n"
                results += "Classification Report:\n" + report
                
                self.confusion_matrix_text.setText(str(cm))
                
                # Plot confusion matrix
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                cax = ax.matshow(cm, cmap=plt.cm.Blues)
                self.figure.colorbar(cax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center')
                
                self.canvas.draw()
                
            elif model_type == "Regression":
                y_pred = self.model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                results = f"Model: {algorithm}\n"
                results += f"Mean Squared Error: {mse:.4f}\n"
                results += f"R² Score: {r2:.4f}\n\n"
                
                # Plot predictions vs actual
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.scatter(self.y_test, y_pred, alpha=0.5)
                ax.plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title('Predictions vs Actual')
                self.canvas.draw()
            
            elif model_type == "Clustering":
                if algorithm == "K-Means":
                    labels = self.model.predict(self.X_test)
                    results = f"Model: {algorithm}\n"
                    results += f"Number of clusters: {self.model.n_clusters}\n"
                    results += f"Inertia: {self.model.inertia_:.4f}\n"
                    
                    # Plot clusters
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    
                    if self.X_test.shape[1] >= 2:
                        scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1], 
                                           c=labels, cmap='viridis', alpha=0.6)
                        ax.set_xlabel('Feature 1')
                        ax.set_ylabel('Feature 2')
                        ax.set_title('K-Means Clustering')
                        self.figure.colorbar(scatter)
                    else:
                        # Use PCA to reduce dimensions for visualization
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(self.X_test)
                        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                           c=labels, cmap='viridis', alpha=0.6)
                        ax.set_xlabel('PC1')
                        ax.set_ylabel('PC2')
                        ax.set_title('K-Means Clustering (PCA reduced)')
                        self.figure.colorbar(scatter)
                    
                    self.canvas.draw()
                
                elif algorithm == "PCA":
                    X_pca = self.model.transform(self.X_test)
                    results = f"Model: {algorithm}\n"
                    results += f"Explained variance ratio: {self.model.explained_variance_ratio_}\n"
                    results += f"Total explained variance: {sum(self.model.explained_variance_ratio_):.4f}\n"
                    
                    # Plot PCA
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
                    ax.set_xlabel('Principal Component 1')
                    ax.set_ylabel('Principal Component 2')
                    ax.set_title('PCA Visualization')
                    self.canvas.draw()
            
            self.results_text.setText(results)
            self.tabs.setCurrentWidget(self.results_tab)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Evaluation failed: {str(e)}")
    
    def cross_validation(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model trained!")
            return
        
        model_type = self.model_type_combo.currentText()
        
        # Skip cross-validation for clustering models
        if model_type == "Clustering":
            QMessageBox.information(self, "Info", "Cross validation is not applicable for clustering models.")
            return
        
        try:
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
            
            results = "Cross Validation Results (5-fold):\n"
            results += f"Scores: {cv_scores}\n"
            results += f"Mean: {cv_scores.mean():.4f}\n"
            results += f"Std: {cv_scores.std():.4f}\n"
            
            self.results_text.append("\n" + results)
            
            # Plot CV scores
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.bar(range(1, 6), cv_scores)
            ax.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
            ax.set_xlabel('Fold')
            ax.set_ylabel('Score')
            ax.set_title('Cross Validation Scores')
            ax.legend()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cross validation failed: {str(e)}")
    
    def make_predictions(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model trained!")
            return
        
        # Create dialog for input features
        dialog = QDialog(self)
        dialog.setWindowTitle("Make Predictions")
        layout = QVBoxLayout(dialog)
        
        # Create input fields based on number of features
        n_features = self.X_train.shape[1]
        input_fields = []
        
        for i in range(n_features):
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(f"Feature {i+1}:"))
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("Enter value")
            input_fields.append(line_edit)
            hbox.addWidget(line_edit)
            layout.addLayout(hbox)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() == QDialog.Accepted:
            try:
                # Collect input values
                features = []
                for field in input_fields:
                    value = field.text()
                    if value:
                        features.append(float(value))
                    else:
                        features.append(0.0)
                
                # Make prediction
                features_array = np.array(features).reshape(1, -1)
                prediction = self.model.predict(features_array)
                
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(features_array)
                    result = f"Prediction: {prediction[0]}\nProbabilities: {proba[0]}"
                else:
                    result = f"Prediction: {prediction[0]}"
                
                QMessageBox.information(self, "Prediction Result", result)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    
    window = SklearnGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

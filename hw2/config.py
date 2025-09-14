# Конфигурация проекта
DATA_PATH = 'Fish.csv'
TARGET_COL = 'Weight'
CATEGORICAL_COLS = ['Species']
NUMERICAL_COLS = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_SAVE_PATH = 'best_model.pkl'
MODELS = [
'LinearRegression',
'Ridge',
'RandomForest',
'GradientBoosting'
]
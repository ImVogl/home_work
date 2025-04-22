"""
В данном скрипте содержиться бизнес логика обучения произвольной модели.
"""

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class ModelTrainer:
    """
    Класс обучения произвольной модели.
    """

    def __init__(self, X, y, model) -> None:
        """
        Инициализация класса.
        :param X: Словарь с наборами векторов входных параметров (Обучающая ('train'), тестовая ('test') и проверочная ('valid')).
        :param y: Словарь с результирующими метками, которые соответствуют значениям из X.
        """
        self.X_train, self.y_train = X['train'], y['train']
        self.X_test, self.y_test = X['test'], y['test']
        self.X_valid, self.y_valid = X['valid'], y['valid']
        self.model = model
        self.r2_score = { 'test': None, 'valid': None}
        self.mae = { 'test': None, 'valid': None}
        self.rmse = { 'test': None, 'valid': None}
        self.mape = { 'test': None, 'valid': None}

    def fit(self) -> None:
        """
        Запуск обучения.
        """
        self.model.fit(self.X_train, self.y_train)
        
        y_pred = self.model.predict(self.X_test)
        self.mae['test'] = mean_absolute_error(self.y_test, y_pred)
        self.rmse['test'] = root_mean_squared_error(self.y_test, y_pred)
        self.mape['test'] = mean_absolute_percentage_error(self.y_test, y_pred)
        self.r2_score['test'] = r2_score(self.y_test, y_pred)

        y_pred = self.model.predict(self.X_valid)
        self.mae['valid'] = mean_absolute_error(self.y_valid, y_pred)
        self.rmse['valid'] = root_mean_squared_error(self.y_valid, y_pred)
        self.mape['valid'] = mean_absolute_percentage_error(self.y_valid, y_pred)
        self.r2_score['valid'] = r2_score(self.y_valid, y_pred)

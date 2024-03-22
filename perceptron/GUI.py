import numpy as np
import tkinter as tk
from tkinter import messagebox

class GUI:
    def __init__(self, root, nn, inputs_mean, inputs_std):
        self.root = root
        self.nn = nn
        self.inputs_mean = inputs_mean
        self.inputs_std = inputs_std
        self.root.title("Классификация транспортных средств")

        self.labels = ['Ширина (м):', 'Длина (м):', 'Высота (м):', 'Масса (т):', 'Количество мест (шт):',
                       'Привод (0 - передний, 1 - задний, 2 - полный):', 'Клиренс (м):',
                       'Объём двигателя (л):', 'Лошадинные силы (шт):',
                       'Тип топлива (0 - бензин, 1 - дизель, 2 - гибрид, 3 - электро):',
                       'Количество колес (шт):', 'Тип размера (Little, Middle, Big):']
        self.entries = []

        for label_text in self.labels:
            label = tk.Label(root, text=label_text)
            label.pack()
            entry = tk.Entry(root)
            entry.pack()
            self.entries.append(entry)

        self.button = tk.Button(root, text="Предположить", command=self.predict)
        self.button.pack()

    def predict(self):
        inputs = []
        for entry in self.entries:
            entry_value = entry.get()
            try:
                inputs.append(float(entry_value))
            except ValueError:
                inputs.append(entry_value)

        category_mapping = {"Big": 0, "Middle": 1, "Little": 2}

        category_column_index = len(inputs) - 1
        category = inputs[category_column_index]
        if category in category_mapping:
            inputs[category_column_index] = category_mapping[category]
        else:
            inputs[category_column_index] = len(category_mapping)

        inputs = (np.array(inputs, dtype=float) - self.inputs_mean) / self.inputs_std
        output = self.nn.feedforward(inputs)
        classes = ['Легковой автомобиль', 'Грузовой автомобиль', 'Автобус']
        print(output)
        prediction = classes[np.argmax(output)]
        messagebox.showinfo("Результат", f"Предполагаемый тип автомобиля: {prediction}")

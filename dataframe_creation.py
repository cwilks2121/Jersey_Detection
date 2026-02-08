import pandas as pd

class DataFrameCreator:
    def __init__(self, fields=['image', 'number', 'last_name', 'color', 'confidence', 'number_ground_truth', 'accuracy', 'hallucination_rate', 'true_number_of_players']):
        self.fields = fields
        self.df = pd.DataFrame({field: [] for field in self.fields})


    def append_df_from_output(self, json_output, img_path):

        ground_truth = self._compute_ground_truth(img_path)
        num_true_players = len(ground_truth)

        accuracy = self._compute_accuracy(json_output['number'], ground_truth)
        hallucination_rate = self._compute_hallucination_rate(json_output['number'], ground_truth)

        row = {
            "image": img_path,
            "number": json_output["number"],
            "last_name": json_output["last_name"],
            "color": json_output["color"],
            "confidence": json_output["confidence"],
            "number_ground_truth": ground_truth,
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "true_number_of_players": num_true_players,
        }

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
    
    def _compute_ground_truth(self, img_path):
        ground_truth = []
        separated_numbers = img_path.replace("-", ".").split(".")
        for part in separated_numbers:
            if part.isdigit() and len(part) <= 2:
                ground_truth.append(int(part))
        return ground_truth
    
    def _compute_accuracy(self, pred_numbers, true_numbers):
        correct = set(pred_numbers) & set(true_numbers)
        accuracy = len(correct) / len(true_numbers) if true_numbers else 0
        return accuracy
    
    def _compute_hallucination_rate(self, pred_numbers, true_numbers):
        hallucinations = set(pred_numbers) - set(true_numbers)
        hallucination_rate = len(hallucinations) / len(pred_numbers) if pred_numbers else 0
        return hallucination_rate

    def get_raw_df(self):
        return self.df

    def print_df(self):
        print(self.df)
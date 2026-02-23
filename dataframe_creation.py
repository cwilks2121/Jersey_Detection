import pandas as pd

class DataFrameCreator:
    def __init__(self, fields=['image', 'boxed_image','jersey_number', 'last_name', 'jersey_color', 'number_color', 'confidence', 'number_ground_truth', 'correctly_identified_numbers', 'false_positives', 'f1_score', 'hallucination_rate', 'true_number_of_players']):
        self.fields = fields
        self.df = pd.DataFrame({field: [] for field in self.fields})

    def append_df_from_output(self, json_output, img_path):
        jersey_numbers, jersey_colors, last_names, number_colors, confidences = self.extract_numbers_and_colors_from_model_json(json_output)

        ground_truth = self._compute_ground_truth(img_path)
        num_true_players = len(ground_truth)

        f1_score = self._compute_f1_score(jersey_numbers, ground_truth)
        hallucination_rate = self._compute_hallucination_rate(jersey_numbers, ground_truth)

        row = {
            "image": img_path,
            "boxed_image": img_path.replace("images/", "boxed_images/"),
            "jersey_number": jersey_numbers,
            "last_name": last_names,
            "jersey_color": jersey_colors,
            "number_color": number_colors,
            "confidence": confidences,
            "number_ground_truth": ground_truth,
            "correctly_identified_numbers": list(set(jersey_numbers) & set(ground_truth)),
            "false_positives": list(set(jersey_numbers) - set(ground_truth)),
            "f1_score": f1_score,
            "hallucination_rate": hallucination_rate,
            "true_number_of_players": num_true_players,
        }

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
    
    def extract_numbers_and_colors_from_model_json(self, json_output):
        """
        Pull jersey numbers out of the model's JSON output.
        It expects something like: {"jerseys": [{"jersey_number": 19}, {"jersey_number": 8}, ...]}
        Returns: a list of unique jersey numbers (ints), in the same order they appear.
        """
        jersey_numbers = []
        last_names = []
        jersey_colors = []
        number_colors = []
        confidences = []

        # Get the jerseys from the JSON
        jerseys = json_output.get("jerseys", [])

        # Get all the jersey numbers (that are ints)
        for j in jerseys:
            n = j.get("jersey_number", None)
            c = j.get("jersey_color", None)
            l = j.get("last_name", None)
            nc = j.get("number_color", None)
            conf = j.get("confidence", None)
            if n is not None:
                jersey_numbers.append(n)
            if c is not None:
                jersey_colors.append(c)
            if l is not None:
                last_names.append(l)
            if nc is not None:
                number_colors.append(nc)
            if conf is not None:
                confidences.append(conf)

        return jersey_numbers, jersey_colors, last_names, number_colors, confidences
    
    def _compute_ground_truth(self, img_path):
        ground_truth = []
        separated_numbers = img_path.replace("-", ".").split(".")
        for part in separated_numbers:
            if part.isdigit() and len(part) <= 2:
                ground_truth.append(int(part))
        return ground_truth
    
    def _compute_f1_score(self, pred_numbers, true_numbers):
        pred_set = set(pred_numbers)
        true_set = set(true_numbers)

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return f1
    
    def _compute_hallucination_rate(self, pred_numbers, true_numbers):
        hallucinations = set(pred_numbers) - set(true_numbers)
        hallucination_rate = len(hallucinations) / len(pred_numbers) if pred_numbers else 0
        return hallucination_rate

    def get_raw_df(self):
        return self.df

    def print_df(self):
        print(self.df)
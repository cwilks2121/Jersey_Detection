def compute_accuracy_and_hallucination(df):
    weighted_correct = df["accuracy"] * df["true_number_of_players"]
    overall_accuracy = weighted_correct.sum() / df["true_number_of_players"].sum()

    weighted_hallucination = df["hallucination_rate"] * df["true_number_of_players"]
    overall_hallucination_rate = weighted_hallucination.sum() / df["true_number_of_players"].sum()

    return overall_accuracy, overall_hallucination_rate 
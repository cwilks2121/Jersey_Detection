def compute_f1_and_hallucination(df):
    weighted_f1 = df["f1_score"] * df["true_number_of_players"]
    overall_f1 = weighted_f1.sum() / df["true_number_of_players"].sum()

    weighted_hallucination = df["hallucination_rate"] * df["true_number_of_players"]
    overall_hallucination_rate = weighted_hallucination.sum() / df["true_number_of_players"].sum()

    return overall_f1, overall_hallucination_rate 
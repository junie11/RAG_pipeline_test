class Evaluation:
    def __init__(self):
        self.f1_scores = []

    def compute_f1(self, predictions, references):
        from collections import Counter
        for pred, truth in zip(predictions, references):
            pred_tokens = pred.split()
            truth_tokens = truth.split()
            common = Counter(pred_tokens) & Counter(truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                self.f1_scores.append(0)
                continue
            precision = num_same / len(pred_tokens)
            recall = num_same / len(truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            self.f1_scores.append(f1)
        average_f1 = sum(self.f1_scores) / len(self.f1_scores) if self.f1_scores else 0
        return average_f1

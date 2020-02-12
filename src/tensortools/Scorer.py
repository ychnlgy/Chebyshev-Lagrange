import sys

class Scorer:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.true_p = 0.0
        self.true_n = 0.0
        self.pred_p = 0.0
        self.total_p = 0.0
        self.total_n = 0.0

    def update(self, yh, y):
        _, I = yh.max(dim=1)
        pred = I == 1
        true = y == 1
        true_p = pred & true
        true_n = (~pred) & (~true)
        self.true_p += true_p.long().sum().item()
        self.true_n += true_n.long().sum().item()
        self.pred_p += pred.long().sum().item()
        self.total_p += true.long().sum().item()
        self.total_n += (~true).long().sum().item()

    def peek(self):
        sens = self.calc_sens()
        spec = self.calc_spec()
        acc = self.calc_acc()
        if self.verbose:
            print("Sens/Spec/Acc: %.5f/%.5f/%.5f" % (sens, spec, acc))
        return (sens+spec)/2.0, acc

    def calc_precision(self):
        if self.pred_p == 0:
            return 0
        return self.true_p/self.pred_p

    def calc_f1(self):
        recall = self.calc_sens()
        precise = self.calc_precision()
        if precise + recall == 0:
            return 0
        return 2*precise*recall/(precise+recall)
    
    def calc_acc(self):
        correct = self.true_p + self.true_n
        total = self.total_p + self.total_n
        return correct/total

    def calc_sens(self):
        if self.total_p == 0:
            return 0
        return self.true_p/self.total_p

    def calc_spec(self):
        if self.total_n == 0:
            return 0
        return self.true_n/self.total_n

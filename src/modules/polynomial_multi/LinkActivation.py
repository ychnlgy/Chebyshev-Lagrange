from . import RegActivation

class LinkActivation(RegActivation):

    # === PROTECTED ===

    def calc_weight(self, slc, *args):
        if slc is self.leftslice: # left or <-1
            w = self.basis.grad_neg1()
        else: # right or >1
            assert slc is self.rightslice
            w = self.basis.grad_pos1()
        p = self.weight*w.view(1, 1, len(w), 1, 1)
        return p.sum(dim=2).view(-1, 1)

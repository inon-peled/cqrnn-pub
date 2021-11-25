from loss import tilted_loss
from loss import yu_stander


def get_loss_function(which):
    return {
        'tl': tilted_loss.make_loss_function_tilted_loss,
        'ys': yu_stander.make_loss_function_yu_standr_nll,
        'ys_multi': yu_stander.make_loss_function_yu_standr_nll_multi
    }[which]

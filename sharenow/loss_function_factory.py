from tilted_loss import make_loss_function_tilted_loss, make_loss_function_tilted_loss_multi
from yu_stander import make_loss_function_yu_standr_nll, make_loss_function_yu_standr_nll_multi
from huber_loss import make_huber_tilted_loss, make_huber_multi_tilted_loss

loss_function_factory = {
    'tl': make_loss_function_tilted_loss,
    'tl_multi': make_loss_function_tilted_loss_multi,
    'ys': make_loss_function_yu_standr_nll,
    'ys_multi': make_loss_function_yu_standr_nll_multi,
    'huber': make_huber_tilted_loss,
    "huber_multi": make_huber_multi_tilted_loss
}
